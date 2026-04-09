from __future__ import annotations

import configparser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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


def _median_abs_deviation(frame: pd.DataFrame) -> pd.Series:
    median = frame.median(axis=1, skipna=True)
    return frame.sub(median, axis=0).abs().median(axis=1, skipna=True)


def _build_pairwise_counts(frame: pd.DataFrame) -> pd.DataFrame:
    columns = list(frame.columns)
    counts = pd.DataFrame(index=columns, columns=columns, dtype=int)
    for left in columns:
        for right in columns:
            counts.loc[left, right] = int(frame[[left, right]].dropna().shape[0])
    return counts


def _compute_linear_gcv_percent(log2_values: pd.DataFrame) -> pd.Series:
    # For log2-ratio data, SD on the log2 scale is stable around zero.
    # Convert SD(log2) into geometric CV on the linear ratio scale.
    log_sd = log2_values.std(axis=1, skipna=True, ddof=1)
    ln_sd = np.log(2.0) * log_sd
    return np.sqrt(np.expm1(np.square(ln_sd))) * 100.0


def _save_distribution_plot(protein_metrics: pd.DataFrame, output_dir: Path) -> Path:
    plot_path = output_dir / "global_sample_cv_metric_distributions.png"
    plot_frame = protein_metrics.loc[
        :,
        [
            "log2_sd",
            "log2_mad",
            "log2_iqr",
            "log2_range",
            "geometric_cv_percent",
        ],
    ].melt(var_name="metric", value_name="value")
    plot_frame = plot_frame.replace([np.inf, -np.inf], np.nan).dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.boxplot(data=plot_frame, x="metric", y="value", ax=axes[0], color="#9ecae1")
    axes[0].set_title("Protein-Level Replicate Variability")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Value")
    axes[0].tick_params(axis="x", rotation=25)

    subset = plot_frame.loc[plot_frame["metric"].isin(["log2_sd", "geometric_cv_percent"])].copy()
    sns.histplot(
        data=subset,
        x="value",
        hue="metric",
        bins=50,
        element="step",
        stat="density",
        common_norm=False,
        ax=axes[1],
    )
    axes[1].set_title("Variability Metric Density")
    axes[1].set_xlabel("Value")
    axes[1].set_ylabel("Density")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _save_heatmap(correlation: pd.DataFrame, output_dir: Path) -> Path:
    plot_path = output_dir / "global_sample_cv_replicate_correlation_heatmap.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        correlation,
        cmap="vlag",
        vmin=0.8,
        vmax=1.0,
        annot=True,
        fmt=".3f",
        square=True,
        cbar_kws={"label": "Pearson r"},
        ax=ax,
    )
    ax.set_title("Replicate Pearson Correlation")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _save_threshold_plot(protein_metrics: pd.DataFrame, output_dir: Path) -> Path:
    plot_path = output_dir / "global_sample_cv_threshold_selection.png"
    metric = pd.to_numeric(protein_metrics["log2_sd"], errors="coerce").dropna()
    threshold_frame = pd.DataFrame(
        {
            "quantile": [0.5, 0.75, 0.9, 0.95],
        }
    )
    threshold_frame["log2_sd"] = threshold_frame["quantile"].map(metric.quantile)
    threshold_frame["recommended_fc"] = np.power(2.0, 2.0 * threshold_frame["log2_sd"])
    threshold_frame["label"] = threshold_frame["quantile"].map(
        {
            0.5: "Median protein",
            0.75: "75th percentile",
            0.9: "90th percentile",
            0.95: "95th percentile",
        }
    )

    fc_grid = np.arange(1.1, 3.05, 0.05)
    coverage_frame = pd.DataFrame({"fold_change": fc_grid})
    coverage_frame["log2_fc"] = np.log2(coverage_frame["fold_change"])
    coverage_frame["fraction_exceeding_2sd"] = coverage_frame["log2_fc"].map(
        lambda value: (value > 2.0 * metric).mean()
    )

    chosen_thresholds = pd.DataFrame({"fold_change": [1.3, 1.5, 1.75, 2.0, 2.2]})
    chosen_thresholds["log2_fc"] = np.log2(chosen_thresholds["fold_change"])
    chosen_thresholds["fraction_exceeding_2sd"] = chosen_thresholds["log2_fc"].map(
        lambda value: (value > 2.0 * metric).mean()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(
        coverage_frame["fold_change"],
        coverage_frame["fraction_exceeding_2sd"] * 100.0,
        color="#2c7fb8",
        linewidth=2,
    )
    for _, row in chosen_thresholds.iterrows():
        axes[0].scatter(row["fold_change"], row["fraction_exceeding_2sd"] * 100.0, color="#d95f0e", s=40)
        axes[0].annotate(
            f"{row['fold_change']:.2f}x\n{row['fraction_exceeding_2sd'] * 100.0:.1f}%",
            (row["fold_change"], row["fraction_exceeding_2sd"] * 100.0),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
        )
    axes[0].set_title("How Many Proteins Exceed Replicate Error")
    axes[0].set_xlabel("Chosen fold-change cutoff")
    axes[0].set_ylabel("Proteins with |log2FC| > 2 x log2SD (%)")
    axes[0].set_ylim(0, 100)
    axes[0].grid(alpha=0.2)

    axes[1].bar(
        threshold_frame["label"],
        threshold_frame["recommended_fc"],
        color=["#bdd7e7", "#6baed6", "#3182bd", "#08519c"],
    )
    for _, row in threshold_frame.iterrows():
        axes[1].annotate(
            f"log2SD={row['log2_sd']:.3f}\nFC={row['recommended_fc']:.2f}x",
            (row["label"], row["recommended_fc"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
        )
    axes[1].set_title("Recommended FC by Replicate-Error Quantile")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Fold change needed to exceed ~2 x log2SD")
    axes[1].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def _save_text_summary(
    *,
    replicate_columns: list[str],
    quant_type: str,
    protein_metrics: pd.DataFrame,
    replicate_summary: pd.DataFrame,
    pearson_corr: pd.DataFrame,
    output_dir: Path,
) -> Path:
    summary_path = output_dir / "global_sample_cv_summary.txt"
    nonnull_metrics = protein_metrics.replace([np.inf, -np.inf], np.nan)

    log2_sd_median = nonnull_metrics["log2_sd"].median(skipna=True)
    log2_sd_p90 = nonnull_metrics["log2_sd"].quantile(0.9)
    gcv_median = nonnull_metrics["geometric_cv_percent"].median(skipna=True)
    gcv_p90 = nonnull_metrics["geometric_cv_percent"].quantile(0.9)
    pairwise_values = pearson_corr.where(~np.eye(len(pearson_corr), dtype=bool)).stack()
    corr_median = pairwise_values.median()
    corr_min = pairwise_values.min()

    worst_replicate = replicate_summary.sort_values(
        by=["median_abs_delta_to_consensus", "rmse_to_consensus"],
        ascending=[False, False],
    ).iloc[0]

    lines = [
        "Global replicate QC summary",
        f"Replicate prefix: {replicate_columns[0].split('_')[0] if replicate_columns else 'NA'}",
        f"Replicate columns ({len(replicate_columns)}): {', '.join(replicate_columns)}",
        f"Quant type: {quant_type}",
        "",
        "Why this does not rely on ordinary CV on log2 ratios:",
        "CV is unstable when the mean is near zero, which is common for log2-ratio data.",
        "Primary variability metrics here are log2 SD, log2 MAD, log2 IQR, and pairwise replicate correlation.",
        "A geometric CV% is also reported after back-transforming from log2 ratio to linear ratio.",
        "",
        f"Median protein log2 SD: {log2_sd_median:.4f}",
        f"90th percentile protein log2 SD: {log2_sd_p90:.4f}",
        f"Median protein geometric CV%: {gcv_median:.2f}",
        f"90th percentile protein geometric CV%: {gcv_p90:.2f}",
        f"Median pairwise Pearson correlation: {corr_median:.4f}",
        f"Minimum pairwise Pearson correlation: {corr_min:.4f}",
        "",
        "Replicate furthest from consensus:",
        (
            f"{worst_replicate['replicate']} | "
            f"median_abs_delta_to_consensus={worst_replicate['median_abs_delta_to_consensus']:.4f}, "
            f"rmse_to_consensus={worst_replicate['rmse_to_consensus']:.4f}, "
            f"pearson_to_consensus={worst_replicate['pearson_to_consensus']:.4f}"
        ),
    ]
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def run_global_sample_cv(config_path: Path) -> Path:
    config = _read_config(config_path)
    input_dir = _get_required_path(config, "input", "input_dir")
    global_path = _get_required_path(config, "input", "global_path")
    output_dir = _get_required_path(config, "output", "output_dir")
    replicate_prefix = _get_required_value(config, "settings", "global_replicates_prefix")
    quant_type = _get_optional_value(config, "settings", "global_data_quant", "raw")
    feature_column = _get_optional_value(
        config,
        "settings",
        "feature_column",
        "Protein.Group.Accessions",
    )

    data_path = input_dir / global_path
    output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(data_path, sep="\t")
    if feature_column not in data.columns:
        feature_column = str(data.columns[0])

    feature_values = data[feature_column].astype(str)
    data = data.loc[~feature_values.str.contains("Cont", na=False)].copy()

    replicate_columns = [column for column in data.columns if str(column).startswith(replicate_prefix)]
    if len(replicate_columns) < 2:
        msg = f"Need at least 2 replicate columns with prefix '{replicate_prefix}'."
        raise ValueError(msg)

    replicate_data = data.loc[:, replicate_columns].apply(pd.to_numeric, errors="coerce")

    protein_metrics = pd.DataFrame({feature_column: data[feature_column]})
    protein_metrics["n_replicates_observed"] = replicate_data.notna().sum(axis=1)
    protein_metrics["replicate_mean"] = replicate_data.mean(axis=1, skipna=True)
    protein_metrics["replicate_median"] = replicate_data.median(axis=1, skipna=True)
    protein_metrics["log2_sd"] = replicate_data.std(axis=1, skipna=True, ddof=1)
    protein_metrics["log2_mad"] = _median_abs_deviation(replicate_data)
    protein_metrics["log2_iqr"] = (
        replicate_data.quantile(0.75, axis=1) - replicate_data.quantile(0.25, axis=1)
    )
    protein_metrics["log2_range"] = replicate_data.max(axis=1, skipna=True) - replicate_data.min(
        axis=1,
        skipna=True,
    )

    if quant_type.lower() == "log2_ratio":
        protein_metrics["geometric_cv_percent"] = _compute_linear_gcv_percent(replicate_data)
    else:
        mean_values = replicate_data.mean(axis=1, skipna=True)
        std_values = replicate_data.std(axis=1, skipna=True, ddof=1)
        protein_metrics["geometric_cv_percent"] = np.where(
            mean_values.abs() > 0,
            std_values / mean_values.abs() * 100.0,
            np.nan,
        )

    consensus = replicate_data.median(axis=1, skipna=True)
    delta_to_consensus = replicate_data.sub(consensus, axis=0)
    replicate_summary = pd.DataFrame(index=replicate_columns)
    replicate_summary["replicate"] = replicate_summary.index
    replicate_summary["non_missing_proteins"] = replicate_data.notna().sum(axis=0)
    replicate_summary["mean_shift_to_consensus"] = delta_to_consensus.mean(axis=0, skipna=True)
    replicate_summary["median_abs_delta_to_consensus"] = delta_to_consensus.abs().median(axis=0, skipna=True)
    replicate_summary["rmse_to_consensus"] = np.sqrt(np.square(delta_to_consensus).mean(axis=0, skipna=True))
    replicate_summary["pearson_to_consensus"] = replicate_data.corrwith(consensus, axis=0, method="pearson")
    replicate_summary["spearman_to_consensus"] = replicate_data.corrwith(consensus, axis=0, method="spearman")
    replicate_summary = replicate_summary.reset_index(drop=True)

    pearson_corr = replicate_data.corr(method="pearson")
    spearman_corr = replicate_data.corr(method="spearman")
    pairwise_counts = _build_pairwise_counts(replicate_data)

    protein_metrics_path = output_dir / "global_sample_cv_protein_metrics.tsv"
    replicate_summary_path = output_dir / "global_sample_cv_replicate_summary.tsv"
    pearson_corr_path = output_dir / "global_sample_cv_replicate_pearson.tsv"
    spearman_corr_path = output_dir / "global_sample_cv_replicate_spearman.tsv"
    pairwise_counts_path = output_dir / "global_sample_cv_pairwise_nonmissing_counts.tsv"

    protein_metrics.to_csv(protein_metrics_path, sep="\t", index=False)
    replicate_summary.to_csv(replicate_summary_path, sep="\t", index=False)
    pearson_corr.to_csv(pearson_corr_path, sep="\t")
    spearman_corr.to_csv(spearman_corr_path, sep="\t")
    pairwise_counts.to_csv(pairwise_counts_path, sep="\t")

    summary_path = _save_text_summary(
        replicate_columns=replicate_columns,
        quant_type=quant_type,
        protein_metrics=protein_metrics,
        replicate_summary=replicate_summary,
        pearson_corr=pearson_corr,
        output_dir=output_dir,
    )
    distribution_plot_path = _save_distribution_plot(protein_metrics, output_dir)
    heatmap_plot_path = _save_heatmap(pearson_corr, output_dir)
    threshold_plot_path = _save_threshold_plot(protein_metrics, output_dir)

    print(f"Replicate columns found: {', '.join(replicate_columns)}")
    print(f"Protein metrics written to {protein_metrics_path}")
    print(f"Replicate summary written to {replicate_summary_path}")
    print(f"Pearson correlation written to {pearson_corr_path}")
    print(f"Spearman correlation written to {spearman_corr_path}")
    print(f"Pairwise non-missing counts written to {pairwise_counts_path}")
    print(f"Summary written to {summary_path}")
    print(f"Distribution plot written to {distribution_plot_path}")
    print(f"Correlation heatmap written to {heatmap_plot_path}")
    print(f"Threshold selection plot written to {threshold_plot_path}")
    return summary_path
