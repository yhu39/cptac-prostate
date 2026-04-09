from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from venn import venn


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _read_config(config_path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.read(config_path)
    return parser


def _get_required_path(config: configparser.ConfigParser, section: str, option: str) -> Path:
    if not config.has_section(section) or not config.has_option(section, option):
        msg = f"Config file is missing [{section}] {option}."
        raise ValueError(msg)
    return Path(_strip_quotes(config.get(section, option)))


def _get_optional_value(
    config: configparser.ConfigParser,
    section: str,
    option: str,
    default: str,
) -> str:
    if not config.has_section(section) or not config.has_option(section, option):
        return default
    return _strip_quotes(config.get(section, option))


def _get_step_names(config: configparser.ConfigParser) -> list[str]:
    if not config.has_section("steps"):
        return []

    step_names: list[str] = []
    for option in sorted(config.options("steps")):
        step_names.append(_strip_quotes(config.get("steps", option)))
    return step_names


def _infer_sep(path: Path) -> str:
    return "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","


def _detect_site_column(columns: pd.Index, preferred: str | None) -> str:
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(["gene_sites", "gene_site"])

    for column in dict.fromkeys(candidates):
        if column in columns:
            return column

    msg = "Could not find a phosphosite column. Expected one of: gene_sites, gene_site."
    raise ValueError(msg)


def _build_combined_output_path(output_dir: Path, reference_path: Path) -> Path:
    ext = reference_path.suffix or ".tsv"
    return output_dir / f"{reference_path.stem}_combined{ext}"


@dataclass
class PhosphoCombinePYConfig:
    input_dir: Path
    pY_path: Path
    new_pY_dir: Path
    new_pY_path: Path
    output_dir: Path
    site_column: str | None = None
    combined_output_name: str | None = None
    summary_name: str = "phospho_combine_pY_summary.txt"
    venn_name: str = "phospho_combine_pY_venn.png"
    overlap_name: str = "phospho_combine_pY_overlap.tsv"

    @property
    def reference_path(self) -> Path:
        return self.input_dir / self.pY_path

    @property
    def new_path(self) -> Path:
        return self.new_pY_dir / self.new_pY_path


@dataclass
class PhosphoCombinePYState:
    cfg: PhosphoCombinePYConfig
    reference_df: pd.DataFrame | None = None
    new_df: pd.DataFrame | None = None
    combined_df: pd.DataFrame | None = None
    site_column: str | None = None
    overlap_counts: dict[str, int] = field(default_factory=dict)
    combined_output_path: Path | None = None


def read_pY(state: PhosphoCombinePYState) -> PhosphoCombinePYState:
    reference_df = pd.read_csv(state.cfg.reference_path, sep=_infer_sep(state.cfg.reference_path))
    new_df = pd.read_csv(state.cfg.new_path, sep=_infer_sep(state.cfg.new_path))

    site_column = _detect_site_column(reference_df.columns, state.cfg.site_column)
    if list(reference_df.columns) != list(new_df.columns):
        msg = "Reference pY table and new pY table do not have identical columns."
        raise ValueError(msg)
    if site_column not in new_df.columns:
        msg = f"Site column '{site_column}' was not found in the new pY table."
        raise ValueError(msg)

    state.reference_df = reference_df
    state.new_df = new_df
    state.site_column = site_column
    return state


def add_new_pY(state: PhosphoCombinePYState) -> PhosphoCombinePYState:
    if state.reference_df is None or state.new_df is None or state.site_column is None:
        msg = "pY tables have not been loaded."
        raise ValueError(msg)

    reference_df = state.reference_df.copy()
    new_df = state.new_df.copy()
    site_column = state.site_column

    reference_sites = reference_df[site_column].astype(str)
    new_sites = new_df[site_column].astype(str)

    reference_unique = set(reference_sites)
    new_unique = set(new_sites)
    overlap_sites = reference_unique & new_unique
    novel_new_sites = new_unique - reference_unique

    novel_new_df = new_df.loc[new_sites.isin(novel_new_sites)].copy()
    novel_new_df = novel_new_df.drop_duplicates(subset=[site_column], keep="first")

    combined_df = pd.concat([reference_df, novel_new_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=[site_column], keep="first")

    state.combined_df = combined_df
    state.overlap_counts = {
        "reference_rows": len(reference_df),
        "new_rows": len(new_df),
        "reference_unique_sites": len(reference_unique),
        "new_unique_sites": len(new_unique),
        "overlap_sites": len(overlap_sites),
        "new_only_sites": len(novel_new_sites),
        "combined_rows": len(combined_df),
    }
    return state


def write_combined_pY(state: PhosphoCombinePYState) -> PhosphoCombinePYState:
    if state.combined_df is None:
        msg = "Combined pY table is not available."
        raise ValueError(msg)

    state.cfg.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = (
        state.cfg.output_dir / state.cfg.combined_output_name
        if state.cfg.combined_output_name
        else _build_combined_output_path(state.cfg.output_dir, state.cfg.pY_path)
    )
    state.combined_df.to_csv(output_path, sep=_infer_sep(output_path), index=False)
    state.combined_output_path = output_path
    return state


def get_summary(state: PhosphoCombinePYState) -> PhosphoCombinePYState:
    if state.reference_df is None or state.new_df is None or state.site_column is None:
        msg = "pY tables have not been loaded."
        raise ValueError(msg)
    if state.combined_df is None or state.combined_output_path is None:
        msg = "Combined pY output is not available."
        raise ValueError(msg)

    output_dir = state.cfg.output_dir
    summary_path = output_dir / state.cfg.summary_name
    venn_path = output_dir / state.cfg.venn_name
    overlap_path = output_dir / state.cfg.overlap_name

    reference_sites = set(state.reference_df[state.site_column].astype(str))
    new_sites = set(state.new_df[state.site_column].astype(str))

    fig, ax = plt.subplots(figsize=(8, 8))
    venn(
        {
            "reference pY": reference_sites,
            "new pY": new_sites,
        },
        ax=ax,
        fontsize=12,
        legend_loc="upper right",
    )
    ax.set_title("Reference vs New pY Site Overlap")
    fig.tight_layout()
    fig.savefig(venn_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    overlap_df = pd.DataFrame(
        [
            {
                "set": "reference_only",
                "count": len(reference_sites - new_sites),
                "sites": ";".join(sorted(reference_sites - new_sites)),
            },
            {
                "set": "overlap",
                "count": len(reference_sites & new_sites),
                "sites": ";".join(sorted(reference_sites & new_sites)),
            },
            {
                "set": "new_only",
                "count": len(new_sites - reference_sites),
                "sites": ";".join(sorted(new_sites - reference_sites)),
            },
        ]
    )
    overlap_df.to_csv(overlap_path, sep="\t", index=False)

    summary_lines = [
        f"reference_path={state.cfg.reference_path}",
        f"new_path={state.cfg.new_path}",
        f"site_column={state.site_column}",
        f"reference_rows={state.overlap_counts['reference_rows']}",
        f"new_rows={state.overlap_counts['new_rows']}",
        f"reference_unique_sites={state.overlap_counts['reference_unique_sites']}",
        f"new_unique_sites={state.overlap_counts['new_unique_sites']}",
        f"overlap_sites={state.overlap_counts['overlap_sites']}",
        f"new_only_sites_added={state.overlap_counts['new_only_sites']}",
        f"combined_rows={state.overlap_counts['combined_rows']}",
        f"combined_output={state.combined_output_path}",
        f"venn_output={venn_path}",
        f"overlap_output={overlap_path}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return state


STEP_FUNCTIONS = {
    "read_pY": read_pY,
    "add_new_pY": add_new_pY,
    "write_combined_pY": write_combined_pY,
    "get_summary": get_summary,
}


def _build_phospho_combine_py_config(config: configparser.ConfigParser) -> PhosphoCombinePYConfig:
    return PhosphoCombinePYConfig(
        input_dir=_get_required_path(config, "input", "input_dir"),
        pY_path=_get_required_path(config, "input", "pY_path"),
        new_pY_dir=_get_required_path(config, "input", "new_pY_dir"),
        new_pY_path=_get_required_path(config, "input", "new_pY_path"),
        output_dir=_get_required_path(config, "output", "output_dir"),
        site_column=_get_optional_value(config, "input", "site_column", "") or None,
        combined_output_name=_get_optional_value(config, "output", "combined_output_name", "") or None,
        summary_name=_get_optional_value(config, "output", "summary_name", "phospho_combine_pY_summary.txt"),
        venn_name=_get_optional_value(config, "output", "venn_name", "phospho_combine_pY_venn.png"),
        overlap_name=_get_optional_value(config, "output", "overlap_name", "phospho_combine_pY_overlap.tsv"),
    )


def run_phospho_combine_py(config_path: Path) -> Path:
    config = _read_config(config_path)
    cfg = _build_phospho_combine_py_config(config)
    state = PhosphoCombinePYState(cfg=cfg)

    for step_name in _get_step_names(config):
        print(step_name)
        if step_name not in STEP_FUNCTIONS:
            msg = f"Unsupported phospho_combine_pY step: {step_name}"
            raise ValueError(msg)
        state = STEP_FUNCTIONS[step_name](state)

    if state.combined_output_path is None:
        msg = "The phospho_combine_pY workflow completed without producing a combined pY table."
        raise ValueError(msg)

    print(f"phospho_combine_pY output written to {state.combined_output_path}")
    return state.combined_output_path
