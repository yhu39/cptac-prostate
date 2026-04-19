from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from cptac_prostate.prerank_gsea import _load_config as _load_prerank_config
from cptac_prostate.purity_validation import (
    _evaluate_feature_matrix,
    _fmt,
    _fmt_g,
    _normalize_pathway_term,
    _prepare_matrix_and_metadata,
    _read_gmt,
    _read_table_auto,
    _resolve_base_diff_inputs,
    _run_prerank_gsea,
    _safe_fdrcorrect,
    _safe_spearman,
    _split_gene_field,
    _strip_quotes,
)

MAX_NEG_LOG10_FDR = 5.0
PLOT_AXIS_PADDING = 1.35


def _read_config(config_path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.read(config_path)
    return parser


def _get_required_path(config: configparser.ConfigParser, section: str, option: str) -> Path:
    if not config.has_section(section) or not config.has_option(section, option):
        msg = f"Config file is missing [{section}] {option}."
        raise ValueError(msg)
    return Path(_strip_quotes(config.get(section, option)))


def _get_optional_path(config: configparser.ConfigParser, section: str, option: str) -> Path | None:
    if not config.has_section(section) or not config.has_option(section, option):
        return None
    value = _strip_quotes(config.get(section, option))
    return Path(value) if value else None


def _get_optional_value(config: configparser.ConfigParser, section: str, option: str, default: str) -> str:
    if not config.has_section(section) or not config.has_option(section, option):
        return default
    return _strip_quotes(config.get(section, option))


def _get_optional_float(config: configparser.ConfigParser, section: str, option: str, default: float) -> float:
    if not config.has_section(section) or not config.has_option(section, option):
        return default
    return float(_strip_quotes(config.get(section, option)))


def _get_optional_int(config: configparser.ConfigParser, section: str, option: str, default: int) -> int:
    if not config.has_section(section) or not config.has_option(section, option):
        return default
    return int(_strip_quotes(config.get(section, option)))


@dataclass
class PathwayPurityConfig:
    output_dir: Path
    gsea_config_path: Path
    gsea_results_path: Path
    matrix_path: Path
    metadata_path: Path
    hallmark_gmt_path: Path
    base_diff_config: Path | None = None
    feature_column: str = "geneSymbol"
    sample_id_column: str | None = None
    group_column: str = "Tissuetype"
    tumor_label: str = "tumor"
    normal_label: str = "normal"
    purity_column: str = "Purity"
    batch_column: str | None = None
    tumor_filter_column: str | None = None
    tumor_filter_value: str | None = None
    high_purity_quantile: float = 0.5
    min_group_size: int = 3
    gsea_permutation_num: int = 1000
    gsea_min_size: int = 15
    gsea_max_size: int = 500
    gsea_seed: int = 123
    substantial_effect_change_pct: float = 30.0
    pathway_significance_fdr: float = 0.05
    top_plot_n: int = 10
    pathway_score_method: str = "mean_zscore"
    protein_missing_value_method: str = "keep"


def _load_config(config_path: Path) -> PathwayPurityConfig:
    config = _read_config(config_path)
    base_diff_config = _get_optional_path(config, "input", "base_diff_config")
    resolved = _resolve_base_diff_inputs(base_diff_config) if base_diff_config else {}
    prerank_cfg = _load_prerank_config(_get_required_path(config, "input", "gsea_config_path"))

    matrix_path = _get_optional_path(config, "input", "matrix_path") or resolved.get("matrix_path")
    metadata_path = _get_optional_path(config, "input", "metadata_path") or resolved.get("metadata_path")
    if matrix_path is None or metadata_path is None:
        msg = "Provide [input] matrix_path and metadata_path, or [input] base_diff_config."
        raise ValueError(msg)

    hallmark_gmt_path = _get_optional_path(config, "input", "hallmark_gmt_path")
    if hallmark_gmt_path is None:
        if Path(prerank_cfg.gene_sets).exists():
            hallmark_gmt_path = Path(prerank_cfg.gene_sets)
        else:
            msg = "Provide [input] hallmark_gmt_path for reproducible reruns of the original Hallmark GSEA."
            raise ValueError(msg)

    return PathwayPurityConfig(
        output_dir=_get_required_path(config, "output", "output_dir"),
        gsea_config_path=_get_required_path(config, "input", "gsea_config_path"),
        gsea_results_path=_get_required_path(config, "input", "gsea_results_path"),
        matrix_path=matrix_path,
        metadata_path=metadata_path,
        hallmark_gmt_path=hallmark_gmt_path,
        base_diff_config=base_diff_config,
        feature_column=_get_optional_value(config, "settings", "feature_column", resolved.get("feature_column", "geneSymbol")),
        sample_id_column=_get_optional_value(config, "settings", "sample_id_column", "") or resolved.get("sample_id_column"),
        group_column=_get_optional_value(config, "settings", "group_column", resolved.get("group_column", "Tissuetype")),
        tumor_label=_get_optional_value(config, "settings", "tumor_label", resolved.get("tumor_label", "tumor")),
        normal_label=_get_optional_value(config, "settings", "normal_label", resolved.get("normal_label", "normal")),
        purity_column=_get_optional_value(config, "settings", "purity_column", resolved.get("purity_column", "Purity")),
        batch_column=_get_optional_value(config, "settings", "batch_column", "") or None,
        tumor_filter_column=_get_optional_value(config, "settings", "tumor_filter_column", "") or None,
        tumor_filter_value=_get_optional_value(config, "settings", "tumor_filter_value", "") or None,
        high_purity_quantile=_get_optional_float(config, "settings", "high_purity_quantile", 0.5),
        min_group_size=_get_optional_int(config, "settings", "min_group_size", 3),
        gsea_permutation_num=_get_optional_int(config, "settings", "gsea_permutation_num", prerank_cfg.permutation_num),
        gsea_min_size=_get_optional_int(config, "settings", "gsea_min_size", prerank_cfg.min_size),
        gsea_max_size=_get_optional_int(config, "settings", "gsea_max_size", prerank_cfg.max_size),
        gsea_seed=_get_optional_int(config, "settings", "gsea_seed", prerank_cfg.seed),
        substantial_effect_change_pct=_get_optional_float(config, "settings", "substantial_effect_change_pct", 30.0),
        pathway_significance_fdr=_get_optional_float(config, "settings", "pathway_significance_fdr", 0.05),
        top_plot_n=_get_optional_int(config, "settings", "top_plot_n", 10),
        pathway_score_method=_get_optional_value(config, "settings", "pathway_score_method", "mean_zscore").casefold(),
        protein_missing_value_method=_get_optional_value(config, "settings", "protein_missing_value_method", "keep").casefold(),
    )


def _build_original_pathway_table(cfg: PathwayPurityConfig, gene_sets: dict[str, list[str]]) -> pd.DataFrame:
    table = _read_table_auto(cfg.gsea_results_path).copy()
    if "Term" not in table.columns or "NES" not in table.columns or "FDR q-val" not in table.columns:
        msg = "Original GSEA results must contain Term, NES, and FDR q-val columns."
        raise ValueError(msg)
    table["Term"] = table["Term"].astype(str).str.strip()
    table["normalized_term"] = table["Term"].map(_normalize_pathway_term)
    table["NES"] = pd.to_numeric(table["NES"], errors="coerce")
    table["FDR q-val"] = pd.to_numeric(table["FDR q-val"], errors="coerce")
    table["Lead_genes"] = table["Lead_genes"] if "Lead_genes" in table.columns else ""
    gene_set_lookup = {_normalize_pathway_term(term): genes for term, genes in gene_sets.items()}
    table["gene_set_genes"] = table["normalized_term"].map(lambda value: ";".join(gene_set_lookup.get(value, [])))
    table["original_direction"] = np.where(table["NES"] > 0, "positive_NES_up_in_tumor", "negative_NES_up_in_normal")
    table["original_significant"] = table["FDR q-val"] < cfg.pathway_significance_fdr
    return table


def _zscore_by_row(matrix: pd.DataFrame) -> pd.DataFrame:
    means = matrix.mean(axis=1)
    stds = matrix.std(axis=1, ddof=0).replace(0.0, np.nan)
    return matrix.sub(means, axis=0).div(stds, axis=0)


def _apply_missing_value_strategy(
    matrix: pd.DataFrame,
    cfg: PathwayPurityConfig,
    summary: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    processed = matrix.copy()
    method = cfg.protein_missing_value_method
    summary["protein_missing_value_method"] = method
    summary["n_proteins_before_missing_value_handling"] = int(processed.shape[0])
    summary["n_samples_before_missing_value_handling"] = int(processed.shape[1])
    summary["n_proteins_with_any_na_before_missing_value_handling"] = int(processed.isna().any(axis=1).sum())
    summary["n_total_na_before_missing_value_handling"] = int(processed.isna().sum().sum())

    if method == "keep":
        pass
    elif method == "drop_any_na":
        processed = processed.loc[~processed.isna().any(axis=1)].copy()
    elif method == "protein_median":
        row_medians = processed.median(axis=1, skipna=True)
        processed = processed.T.fillna(row_medians).T
    elif method == "zero":
        processed = processed.fillna(0.0)
    else:
        msg = f"Unsupported protein_missing_value_method: {method}"
        raise ValueError(msg)

    summary["n_proteins_after_missing_value_handling"] = int(processed.shape[0])
    summary["n_total_na_after_missing_value_handling"] = int(processed.isna().sum().sum())
    summary["n_proteins_removed_by_missing_value_handling"] = (
        summary["n_proteins_before_missing_value_handling"] - summary["n_proteins_after_missing_value_handling"]
    )
    summary["n_retained_proteins"] = int(processed.shape[0])
    summary["n_retained_samples"] = int(processed.shape[1])
    return processed, summary


def _build_pathway_score_table(
    matrix: pd.DataFrame,
    meta: pd.DataFrame,
    sample_id_column: str,
    original_pathways: pd.DataFrame,
    cfg: PathwayPurityConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pathway_gene_sets: dict[str, list[str]] = {}
    rows: list[pd.Series] = []
    definitions: list[dict[str, Any]] = []
    for row in original_pathways.itertuples():
        genes = [gene for gene in _split_gene_field(row.gene_set_genes) if gene in matrix.index]
        lead_genes = [gene for gene in _split_gene_field(row.Lead_genes) if gene in matrix.index]
        if not genes:
            continue
        pathway_gene_sets[row.Term] = genes
        definitions.append(
            {
                "pathway": row.Term,
                "pathway_score_method": cfg.pathway_score_method,
                "n_genes_in_set": len(_split_gene_field(row.gene_set_genes)),
                "n_genes_scored": len(genes),
                "n_leading_edge_scored": len(lead_genes),
                "scored_genes": ";".join(genes),
                "scored_leading_edge_genes": ";".join(lead_genes),
            }
        )
    if not pathway_gene_sets:
        msg = "No significant pathways had scorable genes in the protein matrix."
        raise ValueError(msg)

    if cfg.pathway_score_method == "mean_zscore":
        z_matrix = _zscore_by_row(matrix)
        for pathway, genes in pathway_gene_sets.items():
            score = z_matrix.loc[genes].mean(axis=0, skipna=True)
            score.name = pathway
            rows.append(score)
        score_matrix = pd.DataFrame(rows).T
    elif cfg.pathway_score_method == "ssgsea":
        import gseapy

        ssgsea_result = gseapy.ssgsea(
            data=matrix,
            gene_sets=pathway_gene_sets,
            sample_norm_method="rank",
            outdir=None,
            min_size=max(1, cfg.gsea_min_size),
            max_size=cfg.gsea_max_size,
            permutation_num=0,
            no_plot=True,
            threads=1,
            seed=cfg.gsea_seed,
            verbose=False,
        )
        res2d = ssgsea_result.res2d.copy()
        score_column = "NES" if "NES" in res2d.columns else "ES"
        score_matrix = res2d.pivot(index="Name", columns="Term", values=score_column).apply(pd.to_numeric, errors="coerce")
    else:
        msg = f"Unsupported pathway_score_method: {cfg.pathway_score_method}"
        raise ValueError(msg)

    score_table = meta.copy()
    score_table.insert(0, sample_id_column, score_table.index)
    for column in score_matrix.columns:
        score_table[column] = pd.to_numeric(score_matrix[column].reindex(score_table.index), errors="coerce")
    return score_table, pd.DataFrame(definitions)


def _evaluate_pathway_scores(
    score_table: pd.DataFrame,
    original_pathways: pd.DataFrame,
    cfg: PathwayPurityConfig,
    high_purity_threshold: float,
) -> pd.DataFrame:
    pathway_columns = [row.Term for row in original_pathways.itertuples() if row.Term in score_table.columns]
    score_matrix = score_table.set_index(score_table.columns[0])[pathway_columns].T
    diff_unadjusted, diff_adjusted, comparison = _evaluate_feature_matrix(score_matrix, score_table.set_index(score_table.columns[0]), cfg, high_purity_threshold)
    tumor_mask = score_table["is_tumor"].eq(1) & score_table["purity_numeric"].notna()
    tumor_samples = score_table.loc[tumor_mask, score_table.columns[0]].astype(str).tolist()
    purity = score_table.set_index(score_table.columns[0]).loc[tumor_samples, "purity_numeric"]
    rows: list[dict[str, Any]] = []
    original_lookup = original_pathways.set_index("Term")
    comp_lookup = comparison.set_index("protein")
    for pathway in pathway_columns:
        values = score_table.set_index(score_table.columns[0])[pathway]
        rho, p_value, n_corr = _safe_spearman(values.reindex(tumor_samples), purity)
        comp_row = comp_lookup.loc[pathway]
        original_row = original_lookup.loc[pathway]
        rows.append(
            {
                "pathway": pathway,
                "original_term": pathway,
                "original_nes": original_row["NES"],
                "original_fdr": original_row["FDR q-val"],
                "original_direction": original_row["original_direction"],
                "score_purity_spearman_rho": rho,
                "score_purity_spearman_p_value": p_value,
                "n_tumor_with_purity": n_corr,
                "score_unadjusted_effect_size": comp_row["unadjusted_effect_size"],
                "score_unadjusted_fdr": comp_row["unadjusted_fdr"],
                "score_adjusted_effect_size": comp_row["adjusted_effect_size"],
                "score_adjusted_fdr": comp_row["adjusted_fdr"],
                "score_percent_attenuation": comp_row["percent_attenuation"],
                "score_significance_transition": comp_row["significance_transition"],
            }
        )
    result = pd.DataFrame(rows)
    result["score_purity_spearman_fdr"] = _safe_fdrcorrect(result["score_purity_spearman_p_value"])
    return result


def _match_adjusted_gsea(
    original_pathways: pd.DataFrame,
    adjusted_results: pd.DataFrame,
    pathway_significance_fdr: float,
) -> pd.DataFrame:
    adjusted = adjusted_results.copy()
    adjusted["normalized_term"] = adjusted["Term"].astype(str).str.strip().map(_normalize_pathway_term)
    adjusted_lookup = adjusted.set_index("normalized_term")
    rows: list[dict[str, Any]] = []
    for _, row in original_pathways.iterrows():
        adjusted_row = adjusted_lookup.loc[row["normalized_term"]] if row["normalized_term"] in adjusted_lookup.index else None
        adjusted_nes = np.nan
        adjusted_fdr = np.nan
        adjusted_term = np.nan
        if adjusted_row is not None:
            if isinstance(adjusted_row, pd.DataFrame):
                adjusted_row = adjusted_row.iloc[0]
            adjusted_term = adjusted_row["Term"]
            adjusted_nes = pd.to_numeric(adjusted_row["NES"], errors="coerce")
            adjusted_fdr = pd.to_numeric(adjusted_row["FDR q-val"], errors="coerce")
        rows.append(
            {
                "pathway": row["Term"],
                "original_nes": row["NES"],
                "original_fdr": row["FDR q-val"],
                "original_direction": row["original_direction"],
                "adjusted_term": adjusted_term,
                "adjusted_nes": adjusted_nes,
                "adjusted_fdr": adjusted_fdr,
                "retained_significance_after_adjustment": bool(row["original_significant"]) and pd.notna(adjusted_fdr) and adjusted_fdr < pathway_significance_fdr,
                "lost_significance_after_adjustment": bool(row["original_significant"]) and pd.notna(adjusted_fdr) and adjusted_fdr >= pathway_significance_fdr,
                "nes_direction_changed": pd.notna(adjusted_nes) and np.sign(adjusted_nes) != np.sign(row["NES"]),
                "absolute_nes_attenuation": abs(row["NES"]) - abs(adjusted_nes) if pd.notna(adjusted_nes) else np.nan,
                "missing_in_adjusted_gsea": pd.isna(adjusted_fdr),
            }
        )
    return pd.DataFrame(rows)


def _plot_pathway_score_scatter(
    score_table: pd.DataFrame,
    pathway_results: pd.DataFrame,
    output_dir: Path,
    top_n: int,
    direction_filter: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = pathway_results.sort_values(["original_fdr", "score_purity_spearman_fdr"], na_position="last")
    directions = [direction_filter] if direction_filter else ["positive_NES_up_in_tumor", "negative_NES_up_in_normal"]
    for direction in directions:
        subset = ranked.loc[ranked["original_direction"] == direction].head(top_n)
        for row in subset.itertuples():
            frame = score_table.loc[score_table["is_tumor"].eq(1) & score_table["purity_numeric"].notna(), ["purity_numeric", row.pathway]].dropna()
            if frame.empty:
                continue
            frame = frame.copy()
            frame["purity_numeric"] = pd.to_numeric(frame["purity_numeric"], errors="coerce")
            frame[row.pathway] = pd.to_numeric(frame[row.pathway], errors="coerce")
            frame = frame.dropna()
            if frame.empty:
                continue
            plt.figure(figsize=(5.0, 4.0))
            plt.scatter(frame["purity_numeric"], frame[row.pathway], color="#2563eb", s=24)
            if len(frame) >= 2:
                slope, intercept = np.polyfit(frame["purity_numeric"], frame[row.pathway], 1)
                x = np.linspace(frame["purity_numeric"].min(), frame["purity_numeric"].max(), 100)
                plt.plot(x, slope * x + intercept, color="#dc2626", linewidth=1.5)
            plt.xlabel("Tumor purity")
            plt.ylabel("Pathway score")
            plt.title(f"{row.pathway}\nrho={_fmt(row.score_purity_spearman_rho)}, FDR={_fmt_g(row.score_purity_spearman_fdr)}")
            plt.tight_layout()
            safe_name = "".join(character if character.isalnum() or character in "._-" else "_" for character in row.pathway)
            plt.savefig(output_dir / f"{safe_name}.png", dpi=220)
            plt.close()


def _plot_pathway_significance_change(
    gsea_comparison: pd.DataFrame,
    pathway_significance_fdr: float,
    output_path: Path,
    direction_filter: str | None = None,
) -> None:
    if gsea_comparison.empty:
        return
    frame = gsea_comparison.copy()
    frame["original_fdr_plot"] = pd.to_numeric(frame["original_fdr"], errors="coerce").clip(lower=1e-300)
    frame["adjusted_fdr_plot"] = pd.to_numeric(frame["adjusted_fdr"], errors="coerce").fillna(1.0).clip(lower=1e-300)
    frame["minus_log10_original_fdr_raw"] = -np.log10(frame["original_fdr_plot"])
    frame["minus_log10_adjusted_fdr_raw"] = -np.log10(frame["adjusted_fdr_plot"])
    frame["minus_log10_original_fdr"] = frame["minus_log10_original_fdr_raw"].clip(upper=MAX_NEG_LOG10_FDR)
    frame["minus_log10_adjusted_fdr"] = frame["minus_log10_adjusted_fdr_raw"].clip(upper=MAX_NEG_LOG10_FDR)
    frame["direction_label"] = np.where(frame["original_nes"] >= 0, "positive_NES", "negative_NES")
    if direction_filter == "positive_NES_up_in_tumor":
        frame = frame.loc[frame["direction_label"] == "positive_NES"].copy()
    elif direction_filter == "negative_NES_up_in_normal":
        frame = frame.loc[frame["direction_label"] == "negative_NES"].copy()
    if frame.empty:
        return
    frame["status_label"] = np.select(
        [
            frame["retained_significance_after_adjustment"].eq(True),
            frame["lost_significance_after_adjustment"].eq(True),
        ],
        [
            "retained_after_adjustment",
            "lost_after_adjustment",
        ],
        default="other",
    )

    colors = {
        "retained_after_adjustment": "#15803d",
        "lost_after_adjustment": "#dc2626",
        "other": "#64748b",
    }
    threshold = -np.log10(pathway_significance_fdr)
    frame["sort_group"] = np.select(
        [
            frame["status_label"].eq("lost_after_adjustment"),
            frame["status_label"].eq("retained_after_adjustment"),
        ],
        [0, 1],
        default=2,
    )
    frame["max_significance"] = frame[["minus_log10_original_fdr", "minus_log10_adjusted_fdr"]].max(axis=1)
    frame = frame.sort_values(
        ["sort_group", "max_significance", "minus_log10_original_fdr"],
        ascending=[True, False, False],
        na_position="last",
    ).reset_index(drop=True)
    frame["y"] = np.arange(len(frame))

    fig_height = max(6.0, 0.34 * len(frame) + 1.8)
    plt.figure(figsize=(9.6, fig_height))
    for row in frame.itertuples():
        plt.plot(
            [row.minus_log10_adjusted_fdr, row.minus_log10_original_fdr],
            [row.y, row.y],
            color="#cbd5e1",
            linewidth=1.4,
            zorder=1,
        )

    plt.scatter(
        frame["minus_log10_original_fdr"],
        frame["y"],
        color="#93c5fd",
        marker="o",
        s=34,
        edgecolors="none",
        zorder=3,
    )
    plt.scatter(
        frame["minus_log10_adjusted_fdr"],
        frame["y"],
        color="#fca5a5",
        marker="o",
        s=34,
        edgecolors="none",
        zorder=4,
    )

    lost_count = int(frame["status_label"].eq("lost_after_adjustment").sum())
    retained_count = int(frame["status_label"].eq("retained_after_adjustment").sum())
    if lost_count > 0 and retained_count > 0:
        plt.axhline(lost_count - 0.5, color="#64748b", linestyle=":", linewidth=0.9, zorder=0)

    plt.axvline(threshold, color="#94a3b8", linestyle="--", linewidth=1.2, zorder=0)
    plt.yticks(frame["y"], frame["pathway"])
    plt.gca().invert_yaxis()
    plt.xlim(0.0, MAX_NEG_LOG10_FDR + PLOT_AXIS_PADDING)
    plt.xlabel(f"-log10(GSEA FDR), capped at {int(MAX_NEG_LOG10_FDR)}")
    plt.ylabel("")
    if direction_filter == "positive_NES_up_in_tumor":
        title_suffix = "Positive NES"
    elif direction_filter == "negative_NES_up_in_normal":
        title_suffix = "Negative NES"
    else:
        title_suffix = "All Original Significant Pathways"
    plt.title(f"Pathway Significance Before vs After Purity Adjustment\n{title_suffix}")

    xmin, xmax = plt.xlim()
    label_x = 1.02
    if lost_count > 0:
        plt.text(
            label_x,
            max(0.2, lost_count * 0.35),
            "Lost after\nadjustment",
            transform=plt.gca().get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=8,
            color=colors["lost_after_adjustment"],
        )
    if retained_count > 0:
        retained_center = lost_count + max(0.2, retained_count * 0.45)
        plt.text(
            label_x,
            retained_center,
            "Retained after\nadjustment",
            transform=plt.gca().get_yaxis_transform(),
            va="center",
            ha="left",
            fontsize=8,
            color=colors["retained_after_adjustment"],
        )

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#93c5fd", markeredgecolor="#93c5fd", markersize=6, label="Original GSEA significance"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#fca5a5", markeredgecolor="#fca5a5", markersize=6, label="Purity-adjusted GSEA significance"),
        Line2D([0], [0], color="#cbd5e1", linewidth=1.4, label="Same pathway before vs after adjustment"),
        Line2D([0], [0], color="#94a3b8", linestyle="--", linewidth=1.2, label=f"Significance threshold: FDR = {pathway_significance_fdr}"),
    ]
    xticks = list(range(0, int(MAX_NEG_LOG10_FDR) + 1))
    xlabels = [">5" if tick == int(MAX_NEG_LOG10_FDR) else str(tick) for tick in xticks]
    plt.xticks(xticks, xlabels)
    plt.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=7)
    plt.tight_layout(rect=(0.0, 0.0, 0.76, 1.0))
    plt.savefig(output_path, dpi=220)
    plt.close()


def _plot_pathway_significance_scatter(
    gsea_comparison: pd.DataFrame,
    pathway_significance_fdr: float,
    output_path: Path,
    direction_filter: str | None = None,
) -> None:
    if gsea_comparison.empty:
        return
    frame = gsea_comparison.copy()
    frame["original_fdr_plot"] = pd.to_numeric(frame["original_fdr"], errors="coerce").clip(lower=1e-300)
    frame["adjusted_fdr_plot"] = pd.to_numeric(frame["adjusted_fdr"], errors="coerce").fillna(1.0).clip(lower=1e-300)
    frame["minus_log10_original_fdr_raw"] = -np.log10(frame["original_fdr_plot"])
    frame["minus_log10_adjusted_fdr_raw"] = -np.log10(frame["adjusted_fdr_plot"])
    frame["minus_log10_original_fdr"] = frame["minus_log10_original_fdr_raw"].clip(upper=MAX_NEG_LOG10_FDR)
    frame["minus_log10_adjusted_fdr"] = frame["minus_log10_adjusted_fdr_raw"].clip(upper=MAX_NEG_LOG10_FDR)
    frame["direction_label"] = np.where(frame["original_nes"] >= 0, "positive_NES", "negative_NES")
    if direction_filter == "positive_NES_up_in_tumor":
        frame = frame.loc[frame["direction_label"] == "positive_NES"].copy()
    elif direction_filter == "negative_NES_up_in_normal":
        frame = frame.loc[frame["direction_label"] == "negative_NES"].copy()
    if frame.empty:
        return

    frame["status_label"] = np.select(
        [
            frame["retained_significance_after_adjustment"].eq(True),
            frame["lost_significance_after_adjustment"].eq(True),
        ],
        [
            "retained_after_adjustment",
            "lost_after_adjustment",
        ],
        default="other",
    )
    colors = {
        "retained_after_adjustment": "#15803d",
        "lost_after_adjustment": "#dc2626",
        "other": "#64748b",
    }
    threshold = -np.log10(pathway_significance_fdr)

    plt.figure(figsize=(7.2, 5.8))
    for status_label, color in colors.items():
        subset = frame.loc[frame["status_label"] == status_label]
        if subset.empty:
            continue
        plt.scatter(
            subset["minus_log10_original_fdr"],
            subset["minus_log10_adjusted_fdr"],
            color=color,
            s=42,
            alpha=0.9,
            label=status_label.replace("_", " "),
        )

    def _pick_label_position(
        x: float,
        y: float,
        occupied: list[tuple[float, float]],
        x_span: float,
        y_span: float,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
    ) -> tuple[float, float]:
        def _clamp(candidate_x: float, candidate_y: float) -> tuple[float, float]:
            margin_x = 0.03 * max(x_span, 1.0)
            margin_y = 0.04 * max(y_span, 1.0)
            clamped_x = min(max(candidate_x, x_min + margin_x), x_max - margin_x)
            clamped_y = min(max(candidate_y, y_min + margin_y), y_max - margin_y)
            return clamped_x, clamped_y

        candidates = [
            (x + 0.06 * x_span, y + 0.04 * y_span),
            (x + 0.06 * x_span, y - 0.04 * y_span),
            (x - 0.08 * x_span, y + 0.05 * y_span),
            (x - 0.08 * x_span, y - 0.05 * y_span),
            (x + 0.10 * x_span, y),
            (x, y + 0.07 * y_span),
            (x, y - 0.07 * y_span),
            (x - 0.12 * x_span, y),
        ]
        candidates = [_clamp(cx, cy) for cx, cy in candidates]
        if not occupied:
            return candidates[0]
        best = candidates[0]
        best_score = float("-inf")
        for candidate in candidates:
            distance_to_labels = min(
                abs(candidate[0] - ox) / max(x_span, 1e-6) + abs(candidate[1] - oy) / max(y_span, 1e-6)
                for ox, oy in occupied
            )
            distance_to_point = abs(candidate[0] - x) / max(x_span, 1e-6) + abs(candidate[1] - y) / max(y_span, 1e-6)
            score = distance_to_labels - 0.15 * distance_to_point
            if score > best_score:
                best = candidate
                best_score = score
        return best

    label_candidates = frame.sort_values(
        ["minus_log10_original_fdr", "minus_log10_adjusted_fdr"],
        ascending=[False, False],
        na_position="last",
    )
    x_min, x_max = 0.0, MAX_NEG_LOG10_FDR + PLOT_AXIS_PADDING
    y_min = min(frame["minus_log10_adjusted_fdr"].min(), 0.0)
    y_max = max(frame["minus_log10_adjusted_fdr"].max(), threshold, 1.0, MAX_NEG_LOG10_FDR) + PLOT_AXIS_PADDING
    x_span = max(x_max - x_min, 1.0)
    y_span = max(y_max - y_min, 1.0)
    occupied_positions: list[tuple[float, float]] = []
    for row in label_candidates.itertuples():
        label_x, label_y = _pick_label_position(
            row.minus_log10_original_fdr,
            row.minus_log10_adjusted_fdr,
            occupied_positions,
            x_span,
            y_span,
            x_min,
            x_max,
            y_min,
            y_max,
        )
        occupied_positions.append((label_x, label_y))
        plt.annotate(
            row.pathway,
            xy=(row.minus_log10_original_fdr, row.minus_log10_adjusted_fdr),
            xytext=(label_x, label_y),
            textcoords="data",
            fontsize=7,
            arrowprops={"arrowstyle": "->", "color": "#94a3b8", "lw": 0.8, "shrinkA": 0, "shrinkB": 3},
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.7},
        )

    plt.axvline(threshold, color="#94a3b8", linestyle="--", linewidth=1.0)
    plt.axhline(threshold, color="#94a3b8", linestyle="--", linewidth=1.0)
    if direction_filter == "positive_NES_up_in_tumor":
        title_suffix = "Positive NES"
    elif direction_filter == "negative_NES_up_in_normal":
        title_suffix = "Negative NES"
    else:
        title_suffix = "All Original Significant Pathways"
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel(f"-log10(original GSEA FDR), capped at {int(MAX_NEG_LOG10_FDR)}")
    plt.ylabel(f"-log10(purity-adjusted GSEA FDR), capped at {int(MAX_NEG_LOG10_FDR)}")
    plt.title(f"Pathway Significance Scatter\n{title_suffix}")
    ticks = list(range(0, int(MAX_NEG_LOG10_FDR) + 1))
    tick_labels = [">5" if tick == int(MAX_NEG_LOG10_FDR) else str(tick) for tick in ticks]
    plt.xticks(ticks, tick_labels)
    plt.yticks(ticks, tick_labels)
    plt.legend(frameon=False, fontsize=7, loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def _write_report(
    cfg: PathwayPurityConfig,
    summary: dict[str, Any],
    original_pathways: pd.DataFrame,
    pathway_results: pd.DataFrame,
    gsea_comparison: pd.DataFrame,
) -> Path:
    report_path = cfg.output_dir / "pathway_purity_summary.md"
    positive = gsea_comparison.loc[(gsea_comparison["original_fdr"] < cfg.pathway_significance_fdr) & (gsea_comparison["original_nes"] > 0)]
    negative = gsea_comparison.loc[(gsea_comparison["original_fdr"] < cfg.pathway_significance_fdr) & (gsea_comparison["original_nes"] < 0)]
    lines = [
        "# Pathway Purity Validation Summary",
        "",
        "## Inputs",
        f"- Original GSEA config: `{cfg.gsea_config_path}`",
        f"- Original GSEA results: `{cfg.gsea_results_path}`",
        f"- Matrix: `{cfg.matrix_path}`",
        f"- Metadata: `{cfg.metadata_path}`",
        f"- Hallmark GMT: `{cfg.hallmark_gmt_path}`",
        f"- Pathway score method: `{cfg.pathway_score_method}`",
        f"- Protein missing-value method: `{cfg.protein_missing_value_method}`",
        f"- Retained tumors: {summary['n_retained_tumor']}",
        f"- Retained normals: {summary['n_retained_normal']}",
        f"- Tumor filter: `{summary['tumor_filter_column'] or 'none'}` = `{summary['tumor_filter_value'] or 'none'}`",
        f"- Proteins before missing-value handling: {summary['n_proteins_before_missing_value_handling']}",
        f"- Proteins removed by missing-value handling: {summary['n_proteins_removed_by_missing_value_handling']}",
        f"- Proteins after missing-value handling: {summary['n_proteins_after_missing_value_handling']}",
        "",
        "## Original significant pathways",
        f"- Positive NES significant pathways (FDR < {cfg.pathway_significance_fdr}): {len(positive)}",
        f"- Negative NES significant pathways (FDR < {cfg.pathway_significance_fdr}): {len(negative)}",
        "",
        "## How significance after purity adjustment is evaluated",
        "1. Compute a sample-level pathway score and quantify its association with tumor purity in tumor samples only.",
        "2. Fit protein-level models for tumor vs NAT without purity adjustment (`protein ~ group`).",
        "3. Fit protein-level models for tumor vs NAT with purity adjustment (`protein ~ group + purity`, plus batch if configured).",
        "4. Build a purity-adjusted ranked protein list from the adjusted group effect statistics and rerun preranked GSEA.",
        f"5. Call a pathway retained after purity adjustment if it remains significant at `adjusted FDR < {cfg.pathway_significance_fdr}` in the rerun GSEA.",
        "6. Interpret a pathway as purity-associated if the pathway score tracks purity, but interpret it as not fully purity-explained when the adjusted GSEA still remains significant.",
        "",
        "## Pathways that lost significance after purity adjustment",
    ]
    lost = gsea_comparison.loc[gsea_comparison["lost_significance_after_adjustment"]].sort_values(["original_fdr", "adjusted_fdr"], na_position="last")
    if lost.empty:
        lines.append("- No originally significant pathways lost significance after purity adjustment.")
    else:
        for row in lost.head(20).itertuples():
            lines.append(
                "- "
                + f"`{row.pathway}`: original NES={_fmt(row.original_nes)}, original FDR={_fmt_g(row.original_fdr)}, "
                + f"adjusted NES={_fmt(row.adjusted_nes)}, adjusted FDR={_fmt_g(row.adjusted_fdr)}"
            )
    lines.extend(["", "## Pathway-score purity associations"])
    ranked = pathway_results.sort_values(["score_purity_spearman_fdr", "original_fdr"], na_position="last")
    for row in ranked.head(20).itertuples():
        lines.append(
            "- "
            + f"`{row.pathway}`: rho={_fmt(row.score_purity_spearman_rho)}, score purity FDR={_fmt_g(row.score_purity_spearman_fdr)}, "
            + f"original NES={_fmt(row.original_nes)}, adjusted score FDR={_fmt_g(row.score_adjusted_fdr)}"
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (cfg.output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _write_config_snapshot(cfg: PathwayPurityConfig) -> None:
    lines = [
        "[task]",
        'name = "pathway_purity_validation"',
        "",
        "[input]",
        f'gsea_config_path = "{cfg.gsea_config_path}"',
        f'gsea_results_path = "{cfg.gsea_results_path}"',
        f'matrix_path = "{cfg.matrix_path}"',
        f'metadata_path = "{cfg.metadata_path}"',
        f'hallmark_gmt_path = "{cfg.hallmark_gmt_path}"',
        f'base_diff_config = "{cfg.base_diff_config}"' if cfg.base_diff_config else "base_diff_config = ",
        "",
        "[output]",
        f'output_dir = "{cfg.output_dir}"',
        "",
        "[settings]",
        f'feature_column = "{cfg.feature_column}"',
        f'sample_id_column = "{cfg.sample_id_column}"' if cfg.sample_id_column else "sample_id_column = ",
        f'group_column = "{cfg.group_column}"',
        f'tumor_label = "{cfg.tumor_label}"',
        f'normal_label = "{cfg.normal_label}"',
        f'purity_column = "{cfg.purity_column}"',
        f'batch_column = "{cfg.batch_column}"' if cfg.batch_column else "batch_column = ",
        f'tumor_filter_column = "{cfg.tumor_filter_column}"' if cfg.tumor_filter_column else "tumor_filter_column = ",
        f'tumor_filter_value = "{cfg.tumor_filter_value}"' if cfg.tumor_filter_value else "tumor_filter_value = ",
        f"high_purity_quantile = {cfg.high_purity_quantile}",
        f"min_group_size = {cfg.min_group_size}",
        f"gsea_permutation_num = {cfg.gsea_permutation_num}",
        f"gsea_min_size = {cfg.gsea_min_size}",
        f"gsea_max_size = {cfg.gsea_max_size}",
        f"gsea_seed = {cfg.gsea_seed}",
        f"substantial_effect_change_pct = {cfg.substantial_effect_change_pct}",
        f"pathway_significance_fdr = {cfg.pathway_significance_fdr}",
        f"top_plot_n = {cfg.top_plot_n}",
        f'pathway_score_method = "{cfg.pathway_score_method}"',
        f'protein_missing_value_method = "{cfg.protein_missing_value_method}"',
    ]
    (cfg.output_dir / "config.ini").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pathway_purity_validation(config_path: Path) -> Path:
    cfg = _load_config(config_path)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pathway_purity] loading original GSEA results: {cfg.gsea_results_path}")
    print(f"[pathway_purity] loading matrix: {cfg.matrix_path}")
    print(f"[pathway_purity] loading metadata: {cfg.metadata_path}")
    matrix, meta, summary = _prepare_matrix_and_metadata(cfg)
    matrix, summary = _apply_missing_value_strategy(matrix, cfg, summary)

    tumor_purity = meta.loc[meta["is_tumor"].eq(1), "purity_numeric"].dropna()
    if tumor_purity.empty:
        msg = "No tumor purity values were available after alignment."
        raise ValueError(msg)
    high_purity_threshold = float(tumor_purity.quantile(cfg.high_purity_quantile))

    gene_sets = _read_gmt(cfg.hallmark_gmt_path)
    original_pathways = _build_original_pathway_table(cfg, gene_sets)
    original_sig = original_pathways.loc[original_pathways["original_significant"]].copy()
    if original_sig.empty:
        msg = f"No original significant pathways passed FDR < {cfg.pathway_significance_fdr}."
        raise ValueError(msg)

    print(f"[pathway_purity] original significant pathways: {len(original_sig)}")
    score_table, pathway_definitions = _build_pathway_score_table(matrix, meta, summary["sample_id_column"], original_sig, cfg)

    print("[pathway_purity] fitting protein-level models for reranked GSEA")
    diff_unadjusted, diff_adjusted, _ = _evaluate_feature_matrix(matrix, meta, cfg, high_purity_threshold)

    print(f"[pathway_purity] rerunning Hallmark preranked GSEA from {cfg.hallmark_gmt_path}")
    hallmark_unadjusted = _run_prerank_gsea(
        diff_unadjusted.set_index("protein")["t_value"],
        cfg.hallmark_gmt_path,
        cfg.output_dir / "gsea_hallmark_unadjusted",
        cfg.gsea_permutation_num,
        cfg.gsea_min_size,
        cfg.gsea_max_size,
        cfg.gsea_seed,
    )
    hallmark_adjusted = _run_prerank_gsea(
        diff_adjusted.set_index("protein")["t_value"],
        cfg.hallmark_gmt_path,
        cfg.output_dir / "gsea_hallmark_adjusted",
        cfg.gsea_permutation_num,
        cfg.gsea_min_size,
        cfg.gsea_max_size,
        cfg.gsea_seed,
    )

    print("[pathway_purity] evaluating sample-level pathway scores")
    pathway_results = _evaluate_pathway_scores(score_table, original_sig, cfg, high_purity_threshold)
    gsea_comparison = _match_adjusted_gsea(original_sig, hallmark_adjusted, cfg.pathway_significance_fdr)
    pathway_results = pathway_results.merge(
        gsea_comparison[
            [
                "pathway",
                "adjusted_term",
                "adjusted_nes",
                "adjusted_fdr",
                "retained_significance_after_adjustment",
                "lost_significance_after_adjustment",
                "nes_direction_changed",
                "absolute_nes_attenuation",
                "missing_in_adjusted_gsea",
            ]
        ],
        on="pathway",
        how="left",
    )
    pathway_results["purity_sensitive_interpretation"] = np.select(
        [
            (pathway_results["original_nes"] > 0) & (pathway_results["score_purity_spearman_rho"] > 0) & pathway_results["lost_significance_after_adjustment"],
            (pathway_results["original_nes"] < 0) & (pathway_results["score_purity_spearman_rho"] < 0) & pathway_results["lost_significance_after_adjustment"],
            pathway_results["retained_significance_after_adjustment"],
        ],
        [
            "positive_NES_pathway_is_purity_sensitive",
            "negative_NES_pathway_is_purity_sensitive",
            "pathway_retained_after_purity_adjustment",
        ],
        default="mixed_or_uncertain",
    )

    plots_dir = cfg.output_dir / "plots"
    _plot_pathway_score_scatter(score_table, pathway_results, plots_dir / "pathway_score_vs_purity", cfg.top_plot_n)
    _plot_pathway_score_scatter(
        score_table,
        pathway_results,
        plots_dir / "pathway_score_vs_purity_positive_NES",
        cfg.top_plot_n,
        direction_filter="positive_NES_up_in_tumor",
    )
    _plot_pathway_score_scatter(
        score_table,
        pathway_results,
        plots_dir / "pathway_score_vs_purity_negative_NES",
        cfg.top_plot_n,
        direction_filter="negative_NES_up_in_normal",
    )
    _plot_pathway_significance_change(
        gsea_comparison,
        cfg.pathway_significance_fdr,
        plots_dir / "pathway_significance_before_vs_after_purity_adjustment.png",
    )
    _plot_pathway_significance_change(
        gsea_comparison,
        cfg.pathway_significance_fdr,
        plots_dir / "pathway_significance_before_vs_after_purity_adjustment_positive_NES.png",
        direction_filter="positive_NES_up_in_tumor",
    )
    _plot_pathway_significance_change(
        gsea_comparison,
        cfg.pathway_significance_fdr,
        plots_dir / "pathway_significance_before_vs_after_purity_adjustment_negative_NES.png",
        direction_filter="negative_NES_up_in_normal",
    )
    _plot_pathway_significance_scatter(
        gsea_comparison,
        cfg.pathway_significance_fdr,
        plots_dir / "pathway_significance_scatter_positive_NES.png",
        direction_filter="positive_NES_up_in_tumor",
    )
    _plot_pathway_significance_scatter(
        gsea_comparison,
        cfg.pathway_significance_fdr,
        plots_dir / "pathway_significance_scatter_negative_NES.png",
        direction_filter="negative_NES_up_in_normal",
    )

    print("[pathway_purity] writing outputs")
    cleaned_metadata = meta.copy()
    cleaned_metadata.insert(0, summary["sample_id_column"], cleaned_metadata.index)
    cleaned_metadata["high_purity_threshold"] = high_purity_threshold
    cleaned_metadata.to_csv(cfg.output_dir / "cleaned_metadata.csv", index=False)
    pathway_definitions.to_csv(cfg.output_dir / "pathway_definitions.csv", index=False)
    score_table.to_csv(cfg.output_dir / "pathway_sample_scores.csv", index=False)
    original_sig.to_csv(cfg.output_dir / "original_significant_pathways.csv", index=False)
    original_sig.loc[original_sig["NES"] > 0].to_csv(cfg.output_dir / "original_significant_positive_NES_pathways.csv", index=False)
    original_sig.loc[original_sig["NES"] < 0].to_csv(cfg.output_dir / "original_significant_negative_NES_pathways.csv", index=False)
    pathway_results.to_csv(cfg.output_dir / "pathway_purity_evaluation.csv", index=False)
    gsea_comparison.to_csv(cfg.output_dir / "pathway_gsea_comparison_original_vs_adjusted.csv", index=False)
    pd.DataFrame([summary | {"high_purity_threshold": high_purity_threshold, "n_original_significant_pathways": len(original_sig)}]).to_csv(
        cfg.output_dir / "analysis_summary.csv",
        index=False,
    )
    _write_config_snapshot(cfg)
    report_path = _write_report(cfg, summary, original_sig, pathway_results, gsea_comparison)
    print(f"[pathway_purity] finished; report written to {report_path}")
    return report_path
