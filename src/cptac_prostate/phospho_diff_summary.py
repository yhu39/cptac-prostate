from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from venn import venn

from cptac_prostate.global_diff import _infer_sep, _normalize_grade_value, _pick_sample_column


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


def _get_optional_float(
    config: configparser.ConfigParser,
    section: str,
    option: str,
    default: float,
) -> float:
    if not config.has_section(section) or not config.has_option(section, option):
        return default
    return float(_strip_quotes(config.get(section, option)))


def _get_step_names(config: configparser.ConfigParser) -> list[str]:
    if not config.has_section("steps"):
        return []

    step_names: list[str] = []
    for option in sorted(config.options("steps")):
        step_names.append(_strip_quotes(config.get("steps", option)))
    return step_names


@dataclass
class PhosphoDiffSummaryConfig:
    input_dir: Path
    phospho_path: Path
    meta_dir: Path
    meta_path: Path
    output_dir: Path
    sample_id_column: str | None = None
    phosphosite_columns: tuple[str, str, str] = ("gene_site", "Index", "SequenceWindow")
    venn_groups: list[str] = field(default_factory=lambda: ["GG2", "GG3", "GG4", "GG5", "tumor"])
    heatmap_groups: list[str] = field(
        default_factory=lambda: ["normal", "GG1", "GG2", "GG3", "GG4", "GG5", "tumor"]
    )
    heatmap_vmin: float = -0.58
    heatmap_vmax: float = 0.58
    up_site_label: str = "S-U"

    @property
    def data_path(self) -> Path:
        return self.input_dir / self.phospho_path

    @property
    def metadata_path(self) -> Path:
        return self.meta_dir / self.meta_path


@dataclass
class PhosphoDiffSummaryState:
    cfg: PhosphoDiffSummaryConfig
    meta: pd.DataFrame | None = None
    data: pd.DataFrame | None = None
    phosphosite_info: pd.DataFrame | None = None
    sample_id_column: str | None = None
    site_sets: dict[str, set[str]] = field(default_factory=dict)
    overlap_table: pd.DataFrame | None = None
    heatmap_table: pd.DataFrame | None = None


def _build_summary_config(config: configparser.ConfigParser) -> PhosphoDiffSummaryConfig:
    return PhosphoDiffSummaryConfig(
        input_dir=_get_required_path(config, "input", "input_dir"),
        phospho_path=_get_required_path(config, "input", "phospho_path"),
        meta_dir=_get_required_path(config, "input", "meta_dir"),
        meta_path=_get_required_path(config, "input", "meta_path"),
        output_dir=_get_required_path(config, "output", "output_dir"),
        sample_id_column=(
            _get_optional_value(config, "input", "sample_id_column", "")
            or None
        ),
        heatmap_vmin=_get_optional_float(config, "settings", "heatmap_vmin", -0.58),
        heatmap_vmax=_get_optional_float(config, "settings", "heatmap_vmax", 0.58),
    )


def _load_inputs(cfg: PhosphoDiffSummaryConfig) -> PhosphoDiffSummaryState:
    meta = pd.read_csv(cfg.metadata_path, sep=_infer_sep(cfg.metadata_path))
    data = pd.read_csv(cfg.data_path, sep=_infer_sep(cfg.data_path))

    missing_columns = [column for column in cfg.phosphosite_columns if column not in data.columns]
    if missing_columns:
        msg = f"Phosphosite columns were not found in {cfg.data_path}: {missing_columns}"
        raise ValueError(msg)

    phosphosite_info = data.loc[:, list(cfg.phosphosite_columns)].copy().fillna("").astype(str)
    phosphosite_index = phosphosite_info.apply(lambda row: "|".join(row.values.tolist()), axis=1)
    phosphosite_index.name = "Phosphosite.Index"
    phosphosite_info.index = phosphosite_index

    data = data.drop(columns=list(cfg.phosphosite_columns))
    data = data.apply(pd.to_numeric, errors="coerce")
    data.index = phosphosite_index
    data = data[~data.index.duplicated(keep="first")]
    phosphosite_info = phosphosite_info.loc[data.index]

    sample_id_column = _pick_sample_column(meta, data.columns.tolist(), cfg.sample_id_column)
    return PhosphoDiffSummaryState(
        cfg=cfg,
        meta=meta,
        data=data,
        phosphosite_info=phosphosite_info,
        sample_id_column=sample_id_column,
    )


def _read_up_sites(diff_path: Path, up_site_label: str) -> set[str]:
    if not diff_path.exists():
        return set()
    diff = pd.read_csv(diff_path, sep="\t")
    if "Phosphosite.Index" not in diff.columns or "Significance" not in diff.columns:
        return set()
    return set(
        diff.loc[diff["Significance"] == up_site_label, "Phosphosite.Index"]
        .dropna()
        .astype(str)
        .tolist()
    )


def _load_site_sets(output_dir: Path, group_names: list[str], up_site_label: str) -> dict[str, set[str]]:
    site_sets: dict[str, set[str]] = {}
    for group_name in group_names:
        diff_path = output_dir / f"{group_name}_vs_normal_diff.tsv"
        if group_name.casefold() == "tumor":
            diff_path = output_dir / "tumor_vs_normal_diff.tsv"
        site_sets[group_name] = _read_up_sites(diff_path, up_site_label)
    return site_sets


def _ensure_site_sets(state: PhosphoDiffSummaryState) -> dict[str, set[str]]:
    if not state.site_sets:
        state.site_sets = _load_site_sets(
            state.cfg.output_dir,
            state.cfg.venn_groups,
            state.cfg.up_site_label,
        )
    return state.site_sets


def _group_samples(meta: pd.DataFrame, sample_id_column: str, group_name: str) -> list[str]:
    if group_name.casefold() == "normal":
        mask = meta["Tissuetype"].astype(str).str.casefold() == "normal"
    elif group_name.casefold() == "tumor":
        mask = (
            meta["Tissuetype"].astype(str).str.casefold() == "tumor"
        ) & (
            meta["FirstCategory"].astype(str) == "Sufficient Purity"
        )
    elif group_name.upper().startswith("G"):
        if "BCR_Gleason_Grade" not in meta.columns:
            msg = "Metadata column 'BCR_Gleason_Grade' was not found."
            raise ValueError(msg)
        grade_match = re.fullmatch(r"gg?(\d+)", group_name, flags=re.IGNORECASE)
        if grade_match is None:
            msg = f"Unsupported grade-style group label: {group_name}"
            raise ValueError(msg)
        grade_value = grade_match.group(1)
        normalized_grade = meta["BCR_Gleason_Grade"].apply(_normalize_grade_value)
        mask = (
            meta["Tissuetype"].astype(str).str.casefold() == "tumor"
        ) & (
            meta["FirstCategory"].astype(str) == "Sufficient Purity"
        ) & (
            normalized_grade == grade_value
        )
    else:
        msg = f"Unsupported group for summary plotting: {group_name}"
        raise ValueError(msg)

    return meta.loc[mask, sample_id_column].astype(str).tolist()


def venn_diagram(state: PhosphoDiffSummaryState) -> PhosphoDiffSummaryState:
    output_dir = state.cfg.output_dir
    site_sets = _ensure_site_sets(state)

    venn_sets = {f"{name} vs normal": sites for name, sites in site_sets.items()}
    fig, ax = plt.subplots(figsize=(12, 10))
    venn(
        venn_sets,
        ax=ax,
        figsize=(12, 10),
        fontsize=12,
        legend_loc="upper right",
    )
    ax.set_title("UP-regulated Phosphosites in Different Groups vs Normal")
    fig.tight_layout()
    fig.savefig(output_dir / "phospho_diff_summary_venn.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    results: list[dict[str, object]] = []
    set_names = list(site_sets.keys())
    for r in range(1, len(set_names) + 1):
        for combo in combinations(set_names, r):
            intersection = set.intersection(*(site_sets[name] for name in combo))
            other_names = [name for name in set_names if name not in combo]
            exclusive = intersection.copy()
            for other in other_names:
                exclusive -= site_sets[other]
            results.append(
                {
                    "combination": "&".join(combo),
                    "n_sets": len(combo),
                    "count": len(exclusive),
                    "elements": sorted(exclusive),
                }
            )

    overlap_table = pd.DataFrame(results).sort_values(["n_sets", "combination"]).reset_index(drop=True)
    overlap_table["elements"] = overlap_table["elements"].apply(lambda values: ";".join(values))
    overlap_table.to_csv(output_dir / "phospho_diff_summary_overlap.tsv", sep="\t", index=False)

    state.site_sets = site_sets
    state.overlap_table = overlap_table
    return state


def tumor_overlap_table(state: PhosphoDiffSummaryState) -> PhosphoDiffSummaryState:
    site_sets = _ensure_site_sets(state)
    grade_groups = ["GG2", "GG3", "GG4", "GG5"]
    tumor_sites = site_sets.get("tumor", set())

    column_values: dict[str, list[str]] = {}
    for group_name in grade_groups:
        group_sites = site_sets.get(group_name, set())
        pair_label = f"{group_name}_vs_normal overlap tumor_vs_normal"
        column_values[pair_label] = sorted(group_sites & tumor_sites)

        other_groups = [name for name in grade_groups if name != group_name]
        for other_group in other_groups:
            triple_label = (
                f"{group_name}_vs_normal overlap "
                f"{other_group}_vs_normal overlap tumor_vs_normal"
            )
            column_values[triple_label] = sorted(group_sites & site_sets.get(other_group, set()) & tumor_sites)

    max_len = max((len(values) for values in column_values.values()), default=0)
    padded_columns = {
        column_name: values + [""] * (max_len - len(values))
        for column_name, values in column_values.items()
    }
    pd.DataFrame(padded_columns).T.to_csv(
        state.cfg.output_dir / "phospho_diff_summary_tumor_overlap_table.tsv",
        sep="\t",
        index=True,
    )
    return state


def heatmap(state: PhosphoDiffSummaryState) -> PhosphoDiffSummaryState:
    if state.meta is None or state.data is None or state.sample_id_column is None or state.phosphosite_info is None:
        msg = "Summary inputs have not been loaded."
        raise ValueError(msg)
    if state.overlap_table is None:
        msg = "Venn overlap table is not available."
        raise ValueError(msg)

    meta = state.meta.copy()
    data = state.data.copy()
    phosphosite_info = state.phosphosite_info.copy()
    group_samples = {
        group_name: [
            sample for sample in _group_samples(meta, state.sample_id_column, group_name)
            if sample in data.columns
        ]
        for group_name in state.cfg.heatmap_groups
    }

    rows: list[dict[str, object]] = []
    for _, overlap_row in state.overlap_table.iterrows():
        combination = str(overlap_row["combination"])
        elements = [site for site in str(overlap_row["elements"]).split(";") if site]
        for site in elements:
            if site not in data.index:
                continue
            df_sub = data.loc[site]
            row: dict[str, object] = {
                "Phosphosite.Index": site,
                "combination": combination,
            }
            if site in phosphosite_info.index:
                row.update(phosphosite_info.loc[site].to_dict())
            for group_name in state.cfg.heatmap_groups:
                samples = group_samples[group_name]
                row[group_name] = df_sub[samples].median() if samples else pd.NA
            rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(state.cfg.output_dir / "phospho_diff_summary_heatmap.tsv", sep="\t", index=False)

    if result.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No overlapping phosphosites to plot", ha="center", va="center")
        ax.axis("off")
    else:
        plot_columns = ["normal", "GG1", "GG2", "GG3", "GG4", "GG5", "tumor"]
        plot_frame = result[plot_columns].copy()
        plot_frame.columns = ["Normal", "GG1", "GG2", "GG3", "GG4", "GG5", "Tumor"]
        label_series = result["gene_site"].fillna(result["Phosphosite.Index"])
        if "SequenceWindow" in result.columns:
            label_series = label_series + " | " + result["SequenceWindow"].fillna("")
        plot_frame.index = result["combination"] + " | " + label_series

        fig_height = max(4, 0.22 * len(plot_frame))
        fig, ax = plt.subplots(figsize=(8, fig_height))
        sns.heatmap(
            plot_frame,
            cmap="vlag",
            center=0,
            vmin=state.cfg.heatmap_vmin,
            vmax=state.cfg.heatmap_vmax,
            ax=ax,
            yticklabels=True,
        )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("Median Phosphosite Levels Across Normal, Grade Groups, and Tumor")
        fig.tight_layout()

    fig.savefig(state.cfg.output_dir / "phospho_diff_summary_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    state.heatmap_table = result
    return state


def grade_summary(state: PhosphoDiffSummaryState) -> PhosphoDiffSummaryState:
    return state


STEP_FUNCTIONS = {
    "venn_diagram": venn_diagram,
    "tumor_overlap_table": tumor_overlap_table,
    "heatmap": heatmap,
    "grade_summary": grade_summary,
}


def run_phospho_diff_summary(config_path: Path) -> Path:
    config = _read_config(config_path)
    cfg = _build_summary_config(config)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    state = _load_inputs(cfg)
    for step_name in _get_step_names(config):
        print(step_name)
        if step_name not in STEP_FUNCTIONS:
            msg = f"Unsupported phospho_diff_summary step: {step_name}"
            raise ValueError(msg)
        state = STEP_FUNCTIONS[step_name](state)

    output_path = cfg.output_dir / "phospho_diff_summary_heatmap.png"
    print(f"phospho_diff_summary output written to {output_path}")
    return output_path
