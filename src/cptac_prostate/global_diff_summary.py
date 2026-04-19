from __future__ import annotations

import configparser
from collections import Counter
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


def _get_step_names(config: configparser.ConfigParser) -> list[str]:
    if not config.has_section("steps"):
        return []

    step_names: list[str] = []
    for option in sorted(config.options("steps")):
        step_names.append(_strip_quotes(config.get("steps", option)))
    return step_names


def _read_gene_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    return {line.strip() for line in text.splitlines() if line.strip()}


@dataclass
class DiffSummaryConfig:
    input_dir: Path
    global_path: Path
    meta_dir: Path
    meta_path: Path
    output_dir: Path
    sample_id_column: str | None = None
    feature_column: str = "geneSymbol"
    venn_groups: list[str] = field(default_factory=lambda: ["GG2", "GG3", "GG4", "GG5", "tumor"])
    heatmap_groups: list[str] = field(
        default_factory=lambda: ["normal", "GG1", "GG2", "GG3", "GG4", "GG5", "tumor"]
    )
    global_data_type: str = "DDA"
    global_quant_method: str = "log2_ratio"
    heatmap_transform: str = "raw"
    heatmap_vmin: float = -0.58
    heatmap_vmax: float = 0.58
    up_gene_label: str = "S-U"

    @property
    def data_path(self) -> Path:
        return self.input_dir / self.global_path

    @property
    def metadata_path(self) -> Path:
        return self.meta_dir / self.meta_path


@dataclass
class DiffSummaryState:
    cfg: DiffSummaryConfig
    meta: pd.DataFrame | None = None
    data: pd.DataFrame | None = None
    sample_id_column: str | None = None
    gene_sets: dict[str, set[str]] = field(default_factory=dict)
    overlap_table: pd.DataFrame | None = None
    heatmap_table: pd.DataFrame | None = None


def _build_summary_config(config: configparser.ConfigParser) -> DiffSummaryConfig:
    quant_method = _get_optional_value(config, "settings", "global_quant_method", "log2_ratio")
    data_type = _get_optional_value(config, "settings", "global_data_type", "DDA")
    heatmap_transform = _get_optional_value(config, "settings", "heatmap_transform", "raw")
    return DiffSummaryConfig(
        input_dir=_get_required_path(config, "input", "input_dir"),
        global_path=_get_required_path(config, "input", "global_path"),
        meta_dir=_get_required_path(config, "input", "meta_dir"),
        meta_path=_get_required_path(config, "input", "meta_path"),
        output_dir=_get_required_path(config, "output", "output_dir"),
        sample_id_column=(
            _get_optional_value(config, "input", "sample_id_column", "")
            or None
        ),
        feature_column=_get_optional_value(config, "input", "feature_column", "geneSymbol"),
        global_data_type=data_type,
        global_quant_method=quant_method,
        heatmap_transform=heatmap_transform,
    )


def _load_inputs(cfg: DiffSummaryConfig) -> DiffSummaryState:
    meta = pd.read_csv(cfg.metadata_path, sep=_infer_sep(cfg.metadata_path))
    data = pd.read_csv(cfg.data_path, sep=_infer_sep(cfg.data_path))
    if cfg.feature_column not in data.columns:
        msg = f"Feature column '{cfg.feature_column}' was not found in {cfg.data_path}."
        raise ValueError(msg)

    data = data.set_index(cfg.feature_column)
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data[~data.index.duplicated(keep="first")]

    sample_id_column = _pick_sample_column(meta, data.columns.tolist(), cfg.sample_id_column)
    return DiffSummaryState(cfg=cfg, meta=meta, data=data, sample_id_column=sample_id_column)


def _load_gene_sets(output_dir: Path, group_names: list[str], up_gene_label: str) -> dict[str, set[str]]:
    gene_sets: dict[str, set[str]] = {}
    for group_name in group_names:
        gene_dir = output_dir / f"genes_{group_name}_vs_normal"
        if group_name.casefold() == "tumor":
            gene_dir = output_dir / "genes_tumor_vs_normal"
        gene_sets[group_name] = _read_gene_list(gene_dir / f"{up_gene_label}_genes.txt")
    return gene_sets


def _ensure_gene_sets(state: DiffSummaryState) -> dict[str, set[str]]:
    if not state.gene_sets:
        state.gene_sets = _load_gene_sets(
            state.cfg.output_dir,
            state.cfg.venn_groups,
            state.cfg.up_gene_label,
        )
    return state.gene_sets


def _ensure_heatmap_table(state: DiffSummaryState) -> pd.DataFrame:
    if state.heatmap_table is None:
        path = state.cfg.output_dir / "global_diff_summary_heatmap.tsv"
        if not path.exists():
            msg = "Heatmap table is not available."
            raise ValueError(msg)
        state.heatmap_table = pd.read_csv(path, sep="\t")
    return state.heatmap_table


def _compute_trunk_tumor_program(
    grade_sets: dict[str, set[str]],
) -> list[str]:
    """Identify the tumor-upregulated genes shared across GG2-GG5 as a trunk program."""
    return sorted(set.intersection(*(grade_sets[group] for group in ["GG2", "GG3", "GG4", "GG5"])))


def _compute_grade_progression_metrics(state: DiffSummaryState) -> dict[str, object]:
    gene_sets = _ensure_gene_sets(state)
    heatmap_table = _ensure_heatmap_table(state)
    if state.meta is None or state.data is None or state.sample_id_column is None:
        msg = "Summary inputs have not been loaded."
        raise ValueError(msg)

    grade_sets = {
        group_name: gene_sets.get(group_name, set()) & gene_sets.get("tumor", set())
        for group_name in ["GG2", "GG3", "GG4", "GG5"]
    }
    trunk_genes = _compute_trunk_tumor_program(grade_sets)
    gained_g5 = sorted(grade_sets["GG5"] - grade_sets["GG2"])
    lost_g5 = sorted(grade_sets["GG2"] - grade_sets["GG5"])
    unique_genes = {
        group_name: sorted(
            grade_sets[group_name]
            - set.union(*(grade_sets[other] for other in grade_sets if other != group_name))
        )
        for group_name in grade_sets
    }

    pairwise_jaccard: list[dict[str, object]] = []
    for left, right in combinations(["GG2", "GG3", "GG4", "GG5"], 2):
        intersection = grade_sets[left] & grade_sets[right]
        union = grade_sets[left] | grade_sets[right]
        pairwise_jaccard.append(
            {
                "pair": f"{left}-{right}",
                "intersection": len(intersection),
                "union": len(union),
                "jaccard": (len(intersection) / len(union)) if union else 0.0,
            }
        )
    pairwise_jaccard = sorted(pairwise_jaccard, key=lambda item: item["jaccard"], reverse=True)

    sample_sizes = {
        group_name: len(
            [
                sample
                for sample in _group_samples(state.meta, state.sample_id_column, group_name)
                if sample in state.data.columns
            ]
        )
        for group_name in ["GG1", "GG2", "GG3", "GG4", "GG5", "normal", "tumor"]
    }

    union_genes = sorted(set.union(*grade_sets.values()))
    if not union_genes or heatmap_table.empty or "gene" not in heatmap_table.columns:
        trend_frame = pd.DataFrame(columns=["gene", "normal", "GG1", "GG2", "GG3", "GG4", "GG5", "tumor"])
    else:
        trend_frame = (
            heatmap_table[heatmap_table["gene"].isin(union_genes)]
            .groupby("gene")[["normal", "GG1", "GG2", "GG3", "GG4", "GG5", "tumor"]]
            .median()
            .reset_index()
        )

    trend_records: list[dict[str, object]] = []
    for _, row in trend_frame.iterrows():
        g2 = float(row["GG2"])
        g3 = float(row["GG3"])
        g4 = float(row["GG4"])
        g5 = float(row["GG5"])
        trend_records.append(
            {
                "gene": str(row["gene"]),
                "GG2": g2,
                "GG3": g3,
                "GG4": g4,
                "GG5": g5,
                "delta_G5_G2": g5 - g2,
                "monotonic_up": g2 <= g3 <= g4 <= g5,
                "monotonic_down": g2 >= g3 >= g4 >= g5,
            }
        )
    trend_df = pd.DataFrame(
        trend_records,
        columns=[
            "gene",
            "GG2",
            "GG3",
            "GG4",
            "GG5",
            "delta_G5_G2",
            "monotonic_up",
            "monotonic_down",
        ],
    )
    top_increasing = trend_df.sort_values("delta_G5_G2", ascending=False).head(8)
    top_decreasing = trend_df.sort_values("delta_G5_G2", ascending=True).head(8)
    monotonic_up = sorted(trend_df.loc[trend_df["monotonic_up"], "gene"].tolist())
    monotonic_down = sorted(trend_df.loc[trend_df["monotonic_down"], "gene"].tolist())

    pattern_counter: dict[str, list[str]] = {}
    for gene in union_genes:
        pattern = "".join("1" if gene in grade_sets[group] else "0" for group in ["GG2", "GG3", "GG4", "GG5"])
        pattern_counter.setdefault(pattern, []).append(gene)

    gene_count_counter = Counter()
    for genes in grade_sets.values():
        gene_count_counter.update(genes)
    count_bins = {
        count: sorted([gene for gene, gene_count in gene_count_counter.items() if gene_count == count])
        for count in [4, 3, 2, 1]
    }

    return {
        "grade_sets": grade_sets,
        "trunk_genes": trunk_genes,
        "gained_g5": gained_g5,
        "lost_g5": lost_g5,
        "unique_genes": unique_genes,
        "pairwise_jaccard": pairwise_jaccard,
        "sample_sizes": sample_sizes,
        "trend_frame": trend_frame,
        "top_increasing": top_increasing,
        "top_decreasing": top_decreasing,
        "monotonic_up": monotonic_up,
        "monotonic_down": monotonic_down,
        "pattern_counter": pattern_counter,
        "count_bins": count_bins,
    }


def grade_summary(state: DiffSummaryState) -> DiffSummaryState:
    _compute_grade_progression_metrics(state)
    return state


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


def venn_diagram(state: DiffSummaryState) -> DiffSummaryState:
    output_dir = state.cfg.output_dir
    gene_sets = _ensure_gene_sets(state)

    venn_sets = {f"{name} vs normal": genes for name, genes in gene_sets.items()}
    fig, ax = plt.subplots(figsize=(12, 10))
    if any(venn_sets.values()):
        venn(
            venn_sets,
            ax=ax,
            figsize=(12, 10),
            fontsize=12,
            legend_loc="upper right",
        )
    else:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No up-regulated proteins were detected for the configured groups.",
            ha="center",
            va="center",
            fontsize=14,
        )
    ax.set_title("UP-regulated Proteins in Different Groups vs Normal")
    fig.tight_layout()
    fig.savefig(output_dir / "global_diff_summary_venn.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    results: list[dict[str, object]] = []
    set_names = list(gene_sets.keys())
    for r in range(1, len(set_names) + 1):
        for combo in combinations(set_names, r):
            intersection = set.intersection(*(gene_sets[name] for name in combo))
            other_names = [name for name in set_names if name not in combo]
            exclusive = intersection.copy()
            for other in other_names:
                exclusive -= gene_sets[other]
            results.append(
                {
                    "combination": "&".join(combo),
                    "n_sets": len(combo),
                    "count": len(exclusive),
                    "elements": sorted(exclusive),
                }
            )

    overlap_table = pd.DataFrame(results)
    if overlap_table.empty:
        overlap_table = pd.DataFrame(columns=["combination", "n_sets", "count", "elements"])
    else:
        overlap_table = overlap_table.sort_values(["n_sets", "combination"]).reset_index(drop=True)
        overlap_table["elements"] = overlap_table["elements"].apply(lambda values: ";".join(values))
    overlap_table.to_csv(output_dir / "global_diff_summary_overlap.tsv", sep="\t", index=False)

    state.gene_sets = gene_sets
    state.overlap_table = overlap_table
    return state


def tumor_overlap_table(state: DiffSummaryState) -> DiffSummaryState:
    gene_sets = _ensure_gene_sets(state)
    grade_groups = ["GG2", "GG3", "GG4", "GG5"]
    tumor_genes = gene_sets.get("tumor", set())

    column_values: dict[str, list[str]] = {}
    for group_name in grade_groups:
        group_genes = gene_sets.get(group_name, set())
        pair_label = f"{group_name}_vs_normal overlap tumor_vs_normal"
        column_values[pair_label] = sorted(group_genes & tumor_genes)

        other_groups = [name for name in grade_groups if name != group_name]
        for other_group in other_groups:
            triple_label = (
                f"{group_name}_vs_normal overlap "
                f"{other_group}_vs_normal overlap tumor_vs_normal"
            )
            column_values[triple_label] = sorted(group_genes & gene_sets.get(other_group, set()) & tumor_genes)

    max_len = max((len(values) for values in column_values.values()), default=0)
    padded_columns = {
        column_name: values + [""] * (max_len - len(values))
        for column_name, values in column_values.items()
    }
    pd.DataFrame(padded_columns).T.to_csv(
        state.cfg.output_dir / "global_diff_summary_tumor_overlap_table.tsv",
        sep="\t",
        index=True,
    )
    return state


def heatmap(state: DiffSummaryState) -> DiffSummaryState:
    if state.meta is None or state.data is None or state.sample_id_column is None:
        msg = "Summary inputs have not been loaded."
        raise ValueError(msg)
    if state.overlap_table is None:
        msg = "Venn overlap table is not available."
        raise ValueError(msg)

    meta = state.meta.copy()
    data = state.data.copy()
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
        elements = [gene for gene in str(overlap_row["elements"]).split(";") if gene]
        for gene in elements:
            if gene not in data.index:
                continue
            df_sub = data.loc[gene]
            row: dict[str, object] = {"gene": gene, "combination": combination}
            for group_name in state.cfg.heatmap_groups:
                samples = group_samples[group_name]
                row[group_name] = df_sub[samples].median() if samples else pd.NA
            rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(state.cfg.output_dir / "global_diff_summary_heatmap.tsv", sep="\t", index=False)

    if result.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No overlapping genes to plot", ha="center", va="center")
        ax.axis("off")
    else:
        plot_columns = ["normal", "GG1", "GG2", "GG3", "GG4", "GG5", "tumor"]
        plot_frame = result[plot_columns].copy()
        plot_frame.columns = ["Normal", "GG1", "GG2", "GG3", "GG4", "GG5", "Tumor"]
        row_labels = result["combination"] + " | " + result["gene"]
        plot_frame.index = row_labels

        transform = state.cfg.heatmap_transform.casefold()
        display_frame = plot_frame.copy()
        title_suffix = state.cfg.global_quant_method.replace("_", " ")
        if transform == "zscore":
            row_means = display_frame.mean(axis=1)
            row_stds = display_frame.std(axis=1, ddof=0).replace(0, pd.NA)
            display_frame = display_frame.sub(row_means, axis=0).div(row_stds, axis=0).fillna(0.0)
            title_suffix = f"{title_suffix}, z score"

        fig_height = max(4, 0.22 * len(plot_frame))
        fig, ax = plt.subplots(figsize=(8, fig_height))
        plot_values = pd.to_numeric(display_frame.stack(), errors="coerce").dropna()
        quant_method = state.cfg.global_quant_method.casefold()
        if transform == "zscore":
            sns.heatmap(
                display_frame,
                cmap="vlag",
                center=0,
                vmin=-2.5,
                vmax=2.5,
                ax=ax,
                yticklabels=True,
            )
        elif quant_method == "log2_abundance":
            vmin = float(plot_values.quantile(0.05)) if not plot_values.empty else None
            vmax = float(plot_values.quantile(0.95)) if not plot_values.empty else None
            sns.heatmap(
                display_frame,
                cmap="mako",
                vmin=vmin,
                vmax=vmax,
                ax=ax,
                yticklabels=True,
            )
        else:
            sns.heatmap(
                display_frame,
                cmap="vlag",
                center=0,
                vmin=state.cfg.heatmap_vmin,
                vmax=state.cfg.heatmap_vmax,
                ax=ax,
                yticklabels=True,
            )
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(f"Median Protein Levels Across Normal, Grade Groups, and Tumor ({title_suffix})")
        fig.tight_layout()

    fig.savefig(state.cfg.output_dir / "global_diff_summary_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    state.heatmap_table = result
    return state


STEP_FUNCTIONS = {
    "venn_diagram": venn_diagram,
    "tumor_overlap_table": tumor_overlap_table,
    "heatmap": heatmap,
    "grade_summary": grade_summary,
}


def run_global_diff_summary(config_path: Path) -> Path:
    config = _read_config(config_path)
    cfg = _build_summary_config(config)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    state = _load_inputs(cfg)
    for step_name in _get_step_names(config):
        print(step_name)
        if step_name not in STEP_FUNCTIONS:
            msg = f"Unsupported global_diff_summary step: {step_name}"
            raise ValueError(msg)
        state = STEP_FUNCTIONS[step_name](state)

    output_path = cfg.output_dir / "global_diff_summary_heatmap.png"
    print(f"global_diff_summary output written to {output_path}")
    return output_path
