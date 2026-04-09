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
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection


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


def _get_required_value_with_fallback(
    config: configparser.ConfigParser,
    primary_section: str,
    primary_option: str,
    fallback_section: str,
    fallback_option: str,
) -> str:
    if config.has_section(primary_section) and config.has_option(primary_section, primary_option):
        return _strip_quotes(config.get(primary_section, primary_option))
    return _get_required_value(config, fallback_section, fallback_option)


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
    if "|" in text:
        return text.split("|")[-1].split(" ")[0]
    return text.split(" ")[0]


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


def _build_group_mask(
    meta: pd.DataFrame,
    group_name: str,
    purity_column: str,
    purity_value: str,
) -> pd.Series:
    group_name_cf = group_name.casefold()
    if group_name_cf == "normal":
        return meta["Tissuetype"].astype(str).str.casefold() == "normal"

    if group_name_cf == "tumor":
        return (
            meta["Tissuetype"].astype(str).str.casefold() == "tumor"
        ) & (
            meta[purity_column].astype(str) == purity_value
        )

    grade_match = re.fullmatch(r"gg?(\d+)", group_name, flags=re.IGNORECASE)
    if grade_match is not None:
        if "BCR_Gleason_Grade" not in meta.columns:
            msg = "Metadata column 'BCR_Gleason_Grade' was not found."
            raise ValueError(msg)
        grade_values = meta["BCR_Gleason_Grade"].apply(_normalize_grade_value)
        return (
            meta["Tissuetype"].astype(str).str.casefold() == "tumor"
        ) & (
            meta[purity_column].astype(str) == purity_value
        ) & (
            grade_values == grade_match.group(1)
        )

    return meta["Tissuetype"].astype(str).str.casefold() == group_name_cf


@dataclass
class DiffConfig:
    input_dir: Path
    global_path: Path
    meta_dir: Path
    meta_path: Path
    output_dir: Path
    group1: str
    group2: str
    group_column: str = "Tissuetype"
    purity_column: str = "FirstCategory"
    purity_value: str = "Sufficient Purity"
    sample_id_column: str | None = None
    feature_column: str = "geneSymbol"
    method: str = "Wilcoxon(Unpaired)"
    fdr_cutoff: float = 0.01
    fold_change_cutoff: float = 1.5
    max_miss_ratio_global: float = 0.5
    max_miss_ratio_group: float = 0.5
    min_sample_size: int = 4
    include_group1_purity_filter: bool = True
    moderated_df_prior: float = 4.0

    @property
    def data_path(self) -> Path:
        return self.input_dir / self.global_path

    @property
    def metadata_path(self) -> Path:
        return self.meta_dir / self.meta_path

    @property
    def log2fc_cutoff(self) -> float:
        return float(np.log2(self.fold_change_cutoff))

    @property
    def prefix(self) -> str:
        return f"{self.group1}_vs_{self.group2}"


@dataclass
class DiffState:
    cfg: DiffConfig
    meta: pd.DataFrame | None = None
    data: pd.DataFrame | None = None
    sample_id_column: str | None = None
    group1_samples: list[str] = field(default_factory=list)
    group2_samples: list[str] = field(default_factory=list)
    diff: pd.DataFrame | None = None


def _load_pipeline_inputs(cfg: DiffConfig) -> DiffState:
    meta = pd.read_csv(cfg.metadata_path, sep=_infer_sep(cfg.metadata_path))
    data = pd.read_csv(cfg.data_path, sep=_infer_sep(cfg.data_path))

    if cfg.feature_column not in data.columns:
        msg = f"Feature column '{cfg.feature_column}' was not found in {cfg.data_path}."
        raise ValueError(msg)

    data = data.set_index(cfg.feature_column)
    data = data.apply(pd.to_numeric, errors="coerce")

    sample_id_column = _pick_sample_column(meta, data.columns.tolist(), cfg.sample_id_column)
    return DiffState(cfg=cfg, meta=meta, data=data, sample_id_column=sample_id_column)


def get_sample_info(state: DiffState) -> DiffState:
    if state.meta is None or state.sample_id_column is None:
        msg = "Metadata has not been loaded."
        raise ValueError(msg)

    group_column = state.cfg.group_column
    if group_column not in state.meta.columns:
        msg = f"Metadata column '{group_column}' was not found."
        raise ValueError(msg)
    purity_column = state.cfg.purity_column
    if purity_column not in state.meta.columns:
        msg = f"Metadata column '{purity_column}' was not found."
        raise ValueError(msg)

    meta = state.meta.copy()
    group1_mask = _build_group_mask(
        meta,
        state.cfg.group1,
        purity_column,
        state.cfg.purity_value,
    )
    group2_mask = _build_group_mask(
        meta,
        state.cfg.group2,
        purity_column,
        state.cfg.purity_value,
    )

    if not state.cfg.include_group1_purity_filter and state.cfg.group1.casefold() == "tumor":
        group1_mask = meta[group_column].astype(str).str.casefold() == "tumor"

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


def compare_two_groups(state: DiffState) -> DiffState:
    if state.data is None:
        msg = "Data matrix has not been loaded."
        raise ValueError(msg)
    if not state.group1_samples or not state.group2_samples:
        msg = "Group sample lists have not been prepared."
        raise ValueError(msg)

    if state.cfg.method == "Wilcoxon(Unpaired)":
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
    elif state.cfg.method == "ModeratedT":
        diff = _compare_two_groups_moderated_t(state)
    else:
        msg = f"Unsupported analysis method: {state.cfg.method}"
        raise ValueError(msg)

    diff["Gene"] = [_extract_gene_label(feature_name) for feature_name in diff.index]
    if "Significance" in diff.columns:
        diff["Significance"] = diff["Significance"].fillna("NS")
    diff = _append_pairwise_metadata(state, diff)
    sort_columns = ["AbsLog2FC(median)", "FDR", "Gene"]
    ascending = [False, True, True]
    present_sort_columns = [column for column in sort_columns if column in diff.columns]
    present_ascending = [ascending[idx] for idx, column in enumerate(sort_columns) if column in diff.columns]
    diff = diff.sort_values(present_sort_columns, ascending=present_ascending, na_position="last")
    state.diff = diff
    return state


def plot_volcano(state: DiffState) -> DiffState:
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


def save_results(state: DiffState) -> DiffState:
    if state.diff is None:
        msg = "Differential results are not available."
        raise ValueError(msg)

    diff_path = state.cfg.output_dir / f"{state.cfg.prefix}_diff.tsv"
    state.diff.to_csv(diff_path, sep="\t", index=False)

    gene_dir = state.cfg.output_dir / f"genes_{state.cfg.prefix}"
    gene_dir.mkdir(parents=True, exist_ok=True)
    for label in ["S-U", "S-D", "U", "D"]:
        genes = state.diff.loc[state.diff["Significance"] == label, "Gene"].dropna().astype(str).tolist()
        (gene_dir / f"{label}_genes.txt").write_text("\n".join(genes), encoding="utf-8")

    summary_path = state.cfg.output_dir / f"{state.cfg.prefix}_summary.txt"
    summary_lines = [
        f"group1={state.cfg.group1}",
        f"group2={state.cfg.group2}",
        f"method={state.cfg.method}",
        f"sample_id_column={state.sample_id_column}",
        f"n_group1={len(state.group1_samples)}",
        f"n_group2={len(state.group2_samples)}",
        f"fdr_cutoff={state.cfg.fdr_cutoff}",
        f"fold_change_cutoff={state.cfg.fold_change_cutoff}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return state


STEP_FUNCTIONS = {
    "get_sample_info": get_sample_info,
    "compare_two_groups": compare_two_groups,
    "plot_volcano": plot_volcano,
    "save_results": save_results,
}


def _build_diff_config(config: configparser.ConfigParser) -> DiffConfig:
    log2fc_cutoff = _get_optional_float_with_fallback(
        config,
        "settings",
        "log2FC_threshold",
        "analysis",
        "log2fc_cutoff",
        float(np.log2(1.5)),
    )
    return DiffConfig(
        input_dir=_get_required_path(config, "input", "input_dir"),
        global_path=_get_required_path(config, "input", "global_path"),
        meta_dir=_get_required_path(config, "input", "meta_dir"),
        meta_path=_get_required_path(config, "input", "meta_path"),
        output_dir=_get_required_path(config, "output", "output_dir"),
        group1=_get_required_value_with_fallback(
            config,
            "settings",
            "group1",
            "input",
            "group1",
        ),
        group2=_get_required_value_with_fallback(
            config,
            "settings",
            "group2",
            "input",
            "group2",
        ),
        group_column=_get_optional_value(config, "input", "group_column", "Tissuetype"),
        purity_column=_get_optional_value(config, "input", "purity_column", "FirstCategory"),
        purity_value=_get_optional_value(config, "input", "purity_value", "Sufficient Purity"),
        sample_id_column=(
            _get_optional_value(config, "input", "sample_id_column", "")
            or None
        ),
        feature_column=_get_optional_value(config, "input", "feature_column", "geneSymbol"),
        method=_get_optional_value(config, "analysis", "method", "Wilcoxon(Unpaired)"),
        fdr_cutoff=_get_optional_float_with_fallback(
            config,
            "settings",
            "FDR",
            "analysis",
            "fdr_cutoff",
            0.01,
        ),
        fold_change_cutoff=float(np.power(2.0, log2fc_cutoff)),
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
        moderated_df_prior=_get_optional_float(
            config,
            "analysis",
            "moderated_df_prior",
            4.0,
        ),
    )


def run_global_diff(config_path: Path) -> Path:
    config = _read_config(config_path)
    cfg = _build_diff_config(config)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.method not in {"Wilcoxon(Unpaired)", "ModeratedT"}:
        msg = f"Unsupported analysis method: {cfg.method}"
        raise ValueError(msg)

    state = _load_pipeline_inputs(cfg)
    for step_name in _get_step_names(config):
        print(step_name)
        if step_name not in STEP_FUNCTIONS:
            msg = f"Unsupported global_diff step: {step_name}"
            raise ValueError(msg)
        state = STEP_FUNCTIONS[step_name](state)

    if state.diff is None:
        msg = "The global_diff workflow completed without producing differential results."
        raise ValueError(msg)

    output_path = cfg.output_dir / f"{cfg.prefix}_diff.tsv"
    print(f"global_diff output written to {output_path}")
    return output_path


def _append_pairwise_metadata(state: DiffState, diff: pd.DataFrame) -> pd.DataFrame:
    data = state.data
    if data is None:
        msg = "Data matrix has not been loaded."
        raise ValueError(msg)

    group1_values = data[state.group1_samples]
    group2_values = data[state.group2_samples]
    summary = pd.DataFrame(index=data.index)
    summary["group1_label"] = state.cfg.group1
    summary["group2_label"] = state.cfg.group2
    summary["n_group1_total"] = len(state.group1_samples)
    summary["n_group2_total"] = len(state.group2_samples)
    summary["n_group1_nonmissing"] = group1_values.notna().sum(axis=1)
    summary["n_group2_nonmissing"] = group2_values.notna().sum(axis=1)
    summary["group1_mean"] = group1_values.mean(axis=1, skipna=True)
    summary["group2_mean"] = group2_values.mean(axis=1, skipna=True)
    summary["group1_median"] = group1_values.median(axis=1, skipna=True)
    summary["group2_median"] = group2_values.median(axis=1, skipna=True)
    summary["AbsLog2FC(mean)"] = (summary["group1_mean"] - summary["group2_mean"]).abs()
    summary["AbsLog2FC(median)"] = (summary["group1_median"] - summary["group2_median"]).abs()
    result = diff.join(summary, how="left")
    return result


def _compare_two_groups_moderated_t(state: DiffState) -> pd.DataFrame:
    data = state.data
    if data is None:
        msg = "Data matrix has not been loaded."
        raise ValueError(msg)

    group1 = data[state.group1_samples]
    group2 = data[state.group2_samples]

    n1 = group1.notna().sum(axis=1)
    n2 = group2.notna().sum(axis=1)
    total_n = n1 + n2
    global_missing_ratio = 1.0 - (total_n / (len(state.group1_samples) + len(state.group2_samples)))
    group1_missing_ratio = 1.0 - (n1 / len(state.group1_samples))
    group2_missing_ratio = 1.0 - (n2 / len(state.group2_samples))

    valid = (
        (n1 >= state.cfg.min_sample_size)
        & (n2 >= state.cfg.min_sample_size)
        & (global_missing_ratio <= state.cfg.max_miss_ratio_global)
        & (group1_missing_ratio <= state.cfg.max_miss_ratio_group)
        & (group2_missing_ratio <= state.cfg.max_miss_ratio_group)
    )

    mean1 = group1.mean(axis=1, skipna=True)
    mean2 = group2.mean(axis=1, skipna=True)
    median1 = group1.median(axis=1, skipna=True)
    median2 = group2.median(axis=1, skipna=True)
    var1 = group1.var(axis=1, skipna=True, ddof=1)
    var2 = group2.var(axis=1, skipna=True, ddof=1)

    pooled_df = (n1 + n2 - 2).astype(float)
    pooled_var = (((n1 - 1) * var1) + ((n2 - 1) * var2)) / pooled_df.replace(0, np.nan)
    prior_var = float(np.nanmedian(pooled_var[valid & np.isfinite(pooled_var)]))
    if not np.isfinite(prior_var) or prior_var <= 0:
        prior_var = 1.0

    df_prior = float(state.cfg.moderated_df_prior)
    post_var = ((df_prior * prior_var) + (pooled_df * pooled_var)) / (df_prior + pooled_df)
    se = np.sqrt(post_var * ((1.0 / n1) + (1.0 / n2)))
    stat = (mean1 - mean2) / se.replace(0, np.nan)
    df_total = pooled_df + df_prior
    p_value = pd.Series(
        2.0 * stats.t.sf(np.abs(stat), df=df_total),
        index=data.index,
        dtype=float,
    )

    stat = stat.where(valid)
    p_value = p_value.where(valid)
    fdr = pd.Series(np.nan, index=data.index, dtype=float)
    valid_p = p_value.dropna()
    if not valid_p.empty:
        _, fdr_values = fdrcorrection(valid_p.to_numpy(), alpha=state.cfg.fdr_cutoff)
        fdr.loc[valid_p.index] = fdr_values

    result = pd.DataFrame(index=data.index)
    result["Feature"] = result.index.astype(str)
    result["Log2FC(median)"] = median1 - median2
    result["Log2FC(mean)"] = mean1 - mean2
    result["ModeratedT(Stats)"] = stat
    result["ModeratedT(P-value)"] = p_value
    result["FDR"] = fdr
    result["-Log10(FDR)"] = -np.log10(result["FDR"].clip(lower=np.finfo(float).tiny))
    result.loc[result["FDR"].isna(), "-Log10(FDR)"] = np.nan

    result["Significance"] = "NS"
    up_mask = (result["FDR"] <= state.cfg.fdr_cutoff) & (result["Log2FC(median)"] >= state.cfg.log2fc_cutoff)
    down_mask = (result["FDR"] <= state.cfg.fdr_cutoff) & (result["Log2FC(median)"] <= -state.cfg.log2fc_cutoff)
    suggest_up_mask = result["Log2FC(median)"] >= state.cfg.log2fc_cutoff
    suggest_down_mask = result["Log2FC(median)"] <= -state.cfg.log2fc_cutoff
    result.loc[suggest_up_mask, "Significance"] = "U"
    result.loc[suggest_down_mask, "Significance"] = "D"
    result.loc[up_mask, "Significance"] = "S-U"
    result.loc[down_mask, "Significance"] = "S-D"
    return result
