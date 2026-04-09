from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omicsone_streamlit.plots.volcano import plot_volcano as omics_plot_volcano
from omicsone_streamlit.utils.diff import compare_two_groups as omics_compare_two_groups


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


def _get_required_value(config: configparser.ConfigParser, section: str, option: str) -> str:
    if not config.has_section(section) or not config.has_option(section, option):
        msg = f"Config file is missing [{section}] {option}."
        raise ValueError(msg)
    return _strip_quotes(config.get(section, option))


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


def _get_optional_bool(
    config: configparser.ConfigParser,
    section: str,
    option: str,
    default: bool,
) -> bool:
    if not config.has_section(section) or not config.has_option(section, option):
        return default
    return config.getboolean(section, option)


def _get_step_names(config: configparser.ConfigParser) -> list[str]:
    if not config.has_section("steps"):
        return []

    step_names: list[str] = []
    for option in sorted(config.options("steps")):
        step_names.append(_strip_quotes(config.get("steps", option)))
    return step_names


def _infer_sep(path: Path) -> str:
    return "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","


def _pick_sample_column(meta: pd.DataFrame, data_columns: list[str], preferred: str | None) -> str:
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(["SampleID", "common_ID"])

    best_column: str | None = None
    best_overlap = -1
    data_set = set(data_columns)
    for column in dict.fromkeys(candidates):
        if column not in meta.columns:
            continue
        overlap = int(meta[column].astype(str).isin(data_set).sum())
        if overlap > best_overlap:
            best_overlap = overlap
            best_column = column

    if best_column is None or best_overlap <= 0:
        msg = "Unable to match metadata sample IDs to data columns."
        raise ValueError(msg)

    return best_column


def _extract_gene_label(value: object) -> str:
    text = str(value)
    return text.split("_")[0].split("|")[-1].split(" ")[0]


def _normalize_grade_value(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def _get_optional_float_with_fallback(
    config: configparser.ConfigParser,
    primary_section: str,
    primary_option: str,
    fallback_section: str,
    fallback_option: str,
    default: float,
) -> float:
    if config.has_section(primary_section) and config.has_option(primary_section, primary_option):
        return float(_strip_quotes(config.get(primary_section, primary_option)))
    return _get_optional_float(config, fallback_section, fallback_option, default)


@dataclass
class PhosphoDiffConfig:
    input_dir: Path
    phospho_path: Path
    meta_dir: Path
    meta_path: Path
    output_dir: Path
    group1: str
    group2: str
    group_column: str = "Tissuetype"
    purity_column: str = "FirstCategory"
    purity_value: str = "Sufficient Purity"
    sample_id_column: str | None = None
    phosphosite_columns: tuple[str, str, str] = ("gene_site", "Index", "SequenceWindow")
    method: str = "Wilcoxon(Unpaired)"
    fdr_cutoff: float = 0.01
    log2fc_cutoff: float = 0.58
    max_miss_ratio_global: float = 0.5
    max_miss_ratio_group: float = 0.5
    min_sample_size: int = 4
    include_group1_purity_filter: bool = True

    @property
    def data_path(self) -> Path:
        return self.input_dir / self.phospho_path

    @property
    def metadata_path(self) -> Path:
        return self.meta_dir / self.meta_path

    @property
    def prefix(self) -> str:
        return f"{self.group1}_vs_{self.group2}"


@dataclass
class PhosphoDiffState:
    cfg: PhosphoDiffConfig
    meta: pd.DataFrame | None = None
    data: pd.DataFrame | None = None
    phosphosite_info: pd.DataFrame | None = None
    sample_id_column: str | None = None
    group1_samples: list[str] = field(default_factory=list)
    group2_samples: list[str] = field(default_factory=list)
    diff: pd.DataFrame | None = None


def _load_pipeline_inputs(cfg: PhosphoDiffConfig) -> PhosphoDiffState:
    meta = pd.read_csv(cfg.metadata_path, sep=_infer_sep(cfg.metadata_path))
    data = pd.read_csv(cfg.data_path, sep=_infer_sep(cfg.data_path))

    missing_columns = [column for column in cfg.phosphosite_columns if column not in data.columns]
    if missing_columns:
        msg = f"Phosphosite columns were not found in {cfg.data_path}: {missing_columns}"
        raise ValueError(msg)

    phosphosite_info = data.loc[:, list(cfg.phosphosite_columns)].copy()
    phosphosite_info = phosphosite_info.fillna("").astype(str)
    phosphosite_index = phosphosite_info.apply(lambda row: "|".join(row.values.tolist()), axis=1)
    phosphosite_index.name = "Phosphosite.Index"
    phosphosite_info.index = phosphosite_index

    data = data.drop(columns=list(cfg.phosphosite_columns))
    data = data.apply(pd.to_numeric, errors="coerce")
    data.index = phosphosite_index

    sample_id_column = _pick_sample_column(meta, data.columns.tolist(), cfg.sample_id_column)
    return PhosphoDiffState(
        cfg=cfg,
        meta=meta,
        data=data,
        phosphosite_info=phosphosite_info,
        sample_id_column=sample_id_column,
    )


def get_sample_info(state: PhosphoDiffState) -> PhosphoDiffState:
    if state.meta is None or state.sample_id_column is None:
        msg = "Metadata has not been loaded."
        raise ValueError(msg)

    group_column = state.cfg.group_column
    if group_column not in state.meta.columns:
        msg = f"Metadata column '{group_column}' was not found."
        raise ValueError(msg)

    meta = state.meta.copy()
    group1_mask = meta[group_column].astype(str).str.casefold() == state.cfg.group1.casefold()
    group2_mask = meta[group_column].astype(str).str.casefold() == state.cfg.group2.casefold()

    grade_match = re.fullmatch(r"gg?(\d+)", state.cfg.group1, flags=re.IGNORECASE)
    if grade_match:
        if "BCR_Gleason_Grade" not in meta.columns:
            msg = "Metadata column 'BCR_Gleason_Grade' was not found."
            raise ValueError(msg)
        target_grade = grade_match.group(1)
        grade_values = meta["BCR_Gleason_Grade"].apply(_normalize_grade_value)
        group1_mask = (
            meta["Tissuetype"].astype(str).str.casefold() == "tumor"
        ) & (
            grade_values == target_grade
        )

    if state.cfg.include_group1_purity_filter and state.cfg.group1.casefold() == "tumor":
        purity_column = state.cfg.purity_column
        if purity_column not in meta.columns:
            msg = f"Metadata column '{purity_column}' was not found."
            raise ValueError(msg)
        group1_mask &= meta[purity_column].astype(str) == state.cfg.purity_value
    elif grade_match:
        purity_column = state.cfg.purity_column
        if purity_column not in meta.columns:
            msg = f"Metadata column '{purity_column}' was not found."
            raise ValueError(msg)
        group1_mask &= meta[purity_column].astype(str) == state.cfg.purity_value

    group1_samples = meta.loc[group1_mask, state.sample_id_column].astype(str).tolist()
    group2_samples = meta.loc[group2_mask, state.sample_id_column].astype(str).tolist()

    if state.data is None:
        msg = "Data matrix has not been loaded."
        raise ValueError(msg)

    data_columns = set(state.data.columns)
    group1_samples = [sample for sample in group1_samples if sample in data_columns]
    group2_samples = [sample for sample in group2_samples if sample in data_columns]

    if not group1_samples:
        msg = f"No data columns matched group1='{state.cfg.group1}'."
        raise ValueError(msg)
    if not group2_samples:
        msg = f"No data columns matched group2='{state.cfg.group2}'."
        raise ValueError(msg)

    state.group1_samples = group1_samples
    state.group2_samples = group2_samples

    sample_info = pd.DataFrame(
        {
            "sample": group1_samples + group2_samples,
            "group": [state.cfg.group1] * len(group1_samples) + [state.cfg.group2] * len(group2_samples),
        }
    )
    sample_info.to_csv(
        state.cfg.output_dir / f"{state.cfg.prefix}_sample_info.tsv",
        sep="\t",
        index=False,
    )
    return state


def compare_two_groups(state: PhosphoDiffState) -> PhosphoDiffState:
    if state.data is None:
        msg = "Data matrix has not been loaded."
        raise ValueError(msg)
    if state.phosphosite_info is None:
        msg = "Phosphosite index has not been prepared."
        raise ValueError(msg)
    if not state.group1_samples or not state.group2_samples:
        msg = "Group sample lists have not been prepared."
        raise ValueError(msg)

    diff = omics_compare_two_groups(
        state.data,
        state.group1_samples,
        state.group2_samples,
        method=state.cfg.method,
        fdr_cutoff=state.cfg.fdr_cutoff,
        log2fc_cutoff=state.cfg.log2fc_cutoff,
        max_miss_ratio_global=state.cfg.max_miss_ratio_global,
        max_miss_ratio_group=state.cfg.max_miss_ratio_group,
        min_sample_size=state.cfg.min_sample_size,
    ).copy()

    diff.insert(0, "Phosphosite.Index", diff.index.astype(str))
    diff = diff.join(state.phosphosite_info, how="left")
    diff["Gene"] = diff["gene_site"].map(_extract_gene_label)
    if "Significance" in diff.columns:
        diff["Significance"] = diff["Significance"].fillna("NS")
    state.diff = diff
    return state


def plot_volcano(state: PhosphoDiffState) -> PhosphoDiffState:
    if state.diff is None:
        msg = "Differential results are not available."
        raise ValueError(msg)

    plot_frame = state.diff.copy()
    xmin = plot_frame["Log2FC(median)"].min(skipna=True)
    xmax = plot_frame["Log2FC(median)"].max(skipna=True)
    xlimit = max(abs(np.floor(xmin)), abs(np.ceil(xmax)), state.cfg.log2fc_cutoff)

    fig = omics_plot_volcano(
        plot_frame,
        log2fc_threshold=state.cfg.log2fc_cutoff,
        xlim=(-xlimit, xlimit),
    )
    fig.savefig(
        state.cfg.output_dir / f"{state.cfg.prefix}_volcano.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    return state


def save_results(state: PhosphoDiffState) -> PhosphoDiffState:
    if state.diff is None:
        msg = "Differential results are not available."
        raise ValueError(msg)

    diff_path = state.cfg.output_dir / f"{state.cfg.prefix}_diff.tsv"
    state.diff.to_csv(diff_path, sep="\t", index=False)

    gene_dir = state.cfg.output_dir / f"genes_{state.cfg.prefix}"
    gene_dir.mkdir(parents=True, exist_ok=True)
    for label in ["S-U", "S-D", "U", "D"]:
        genes = state.diff.loc[state.diff["Significance"] == label, "Gene"].dropna().astype(str).drop_duplicates().tolist()
        (gene_dir / f"{label}_genes.txt").write_text("\n".join(genes), encoding="utf-8")

    summary_path = state.cfg.output_dir / f"{state.cfg.prefix}_summary.txt"
    summary_lines = [
        f"group1={state.cfg.group1}",
        f"group2={state.cfg.group2}",
        f"sample_id_column={state.sample_id_column}",
        f"n_group1={len(state.group1_samples)}",
        f"n_group2={len(state.group2_samples)}",
        f"fdr_cutoff={state.cfg.fdr_cutoff}",
        f"log2fc_cutoff={state.cfg.log2fc_cutoff}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return state


STEP_FUNCTIONS = {
    "get_sample_info": get_sample_info,
    "compare_two_groups": compare_two_groups,
    "plot_volcano": plot_volcano,
    "save_results": save_results,
}


def _build_phospho_diff_config(config: configparser.ConfigParser) -> PhosphoDiffConfig:
    return PhosphoDiffConfig(
        input_dir=_get_required_path(config, "input", "input_dir"),
        phospho_path=_get_required_path(config, "input", "phospho_path"),
        meta_dir=_get_required_path(config, "input", "meta_dir"),
        meta_path=_get_required_path(config, "input", "meta_path"),
        output_dir=_get_required_path(config, "output", "output_dir"),
        group1=_get_required_value(config, "settings", "group1"),
        group2=_get_required_value(config, "settings", "group2"),
        group_column=_get_optional_value(config, "input", "group_column", "Tissuetype"),
        purity_column=_get_optional_value(config, "input", "purity_column", "FirstCategory"),
        purity_value=_get_optional_value(config, "input", "purity_value", "Sufficient Purity"),
        sample_id_column=(
            _get_optional_value(config, "input", "sample_id_column", "")
            or None
        ),
        method=_get_optional_value(config, "analysis", "method", "Wilcoxon(Unpaired)"),
        fdr_cutoff=_get_optional_float_with_fallback(
            config,
            "settings",
            "FDR",
            "analysis",
            "fdr_cutoff",
            0.01,
        ),
        log2fc_cutoff=_get_optional_float_with_fallback(
            config,
            "settings",
            "log2FC_threshold",
            "analysis",
            "log2fc_cutoff",
            0.58,
        ),
        max_miss_ratio_global=_get_optional_float(
            config,
            "analysis",
            "max_miss_ratio_global",
            0.5,
        ),
        max_miss_ratio_group=_get_optional_float(
            config,
            "analysis",
            "max_miss_ratio_group",
            0.5,
        ),
        min_sample_size=int(_get_optional_float(config, "analysis", "min_sample_size", 4)),
        include_group1_purity_filter=_get_optional_bool(
            config,
            "input",
            "include_group1_purity_filter",
            True,
        ),
    )


def run_phospho_diff(config_path: Path) -> Path:
    config = _read_config(config_path)
    cfg = _build_phospho_diff_config(config)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.method != "Wilcoxon(Unpaired)":
        msg = f"Unsupported analysis method: {cfg.method}"
        raise ValueError(msg)

    state = _load_pipeline_inputs(cfg)
    for step_name in _get_step_names(config):
        print(step_name)
        if step_name not in STEP_FUNCTIONS:
            msg = f"Unsupported phospho_diff step: {step_name}"
            raise ValueError(msg)
        state = STEP_FUNCTIONS[step_name](state)

    if state.diff is None:
        msg = "The phospho_diff workflow completed without producing differential results."
        raise ValueError(msg)

    output_path = cfg.output_dir / f"{cfg.prefix}_diff.tsv"
    print(f"phospho_diff output written to {output_path}")
    return output_path
