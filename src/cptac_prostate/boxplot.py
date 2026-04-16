from __future__ import annotations

import configparser
from dataclasses import dataclass
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

from cptac_prostate.global_diff import _build_group_mask, _infer_sep, _pick_sample_column
from cptac_prostate.global_diff import _normalize_grade_value


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


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _significance_stars(pvalue: float | None) -> str:
    if pvalue is None or pd.isna(pvalue):
        return "ns"
    if pvalue < 0.001:
        return "***"
    if pvalue < 0.01:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "ns"


@dataclass
class BoxplotConfig:
    input_dir: Path
    global_path: Path
    meta_dir: Path
    meta_path: Path
    output_dir: Path
    gene_names: list[str]
    group_order: list[str] | None
    plot_name: str
    sample_id_column: str | None = None
    feature_column: str = "geneSymbol"
    group_column: str = "Tissuetype"
    purity_column: str = "FirstCategory"
    purity_value: str = "Sufficient Purity"
    reference_group: str = "normal"
    figure_ncols: int = 4
    cluster_method: str | None = None

    @property
    def data_path(self) -> Path:
        return self.input_dir / self.global_path

    @property
    def metadata_path(self) -> Path:
        return self.meta_dir / self.meta_path


def _build_boxplot_config(config: configparser.ConfigParser) -> BoxplotConfig:
    gene_names = _parse_csv_list(_get_optional_value(config, "settings", "gene_names", ""))
    if not gene_names:
        single_gene = _get_optional_value(config, "settings", "gene", "")
        if single_gene:
            gene_names = [single_gene]
    if not gene_names:
        msg = "Config file must define [settings] gene_names or [settings] gene."
        raise ValueError(msg)

    group_order_value = _get_optional_value(config, "settings", "group_order", "")
    group_order = _parse_csv_list(group_order_value) if group_order_value else None
    cluster_method = _get_optional_value(config, "settings", "cluster_method", "") or None
    return BoxplotConfig(
        input_dir=_get_required_path(config, "input", "input_dir"),
        global_path=_get_required_path(config, "input", "global_path"),
        meta_dir=_get_required_path(config, "input", "meta_dir"),
        meta_path=_get_required_path(config, "input", "meta_path"),
        output_dir=_get_required_path(config, "output", "output_dir"),
        gene_names=gene_names,
        group_order=group_order,
        plot_name=_get_optional_value(
            config,
            "settings",
            "plot_name",
            f"{gene_names[0]}_boxplot" if len(gene_names) == 1 else "boxplot",
        ),
        sample_id_column=(
            _get_optional_value(config, "input", "sample_id_column", "")
            or None
        ),
        feature_column=_get_optional_value(config, "input", "feature_column", "geneSymbol"),
        group_column=_get_optional_value(config, "input", "group_column", "Tissuetype"),
        purity_column=_get_optional_value(config, "input", "purity_column", "FirstCategory"),
        purity_value=_get_optional_value(config, "input", "purity_value", "Sufficient Purity"),
        reference_group=_get_optional_value(config, "settings", "reference_group", "normal"),
        figure_ncols=int(float(_get_optional_value(config, "settings", "figure_ncols", "4"))),
        cluster_method=cluster_method,
    )


def _normalize_cluster_label(cluster_method: str, value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    if cluster_method == "BCR_Gleason_Grade":
        normalized = _normalize_grade_value(value)
        if normalized is None:
            return None
        if str(normalized).isdigit():
            return f"GG{normalized}"
        return None
    return text


def _group_sort_key(label: str) -> tuple[int, int | str]:
    if label == "normal":
        return (0, 0)
    if label == "tumor":
        return (2, 0)
    if label.startswith("GG") and label.removeprefix("GG").isdigit():
        return (1, int(label.removeprefix("GG")))
    return (1, label)


def _build_group_assignments(
    cfg: BoxplotConfig,
    meta: pd.DataFrame,
    sample_id_column: str,
    data_columns: set[str],
) -> tuple[dict[str, list[str]], list[str], pd.DataFrame]:
    if cfg.cluster_method:
        if cfg.cluster_method not in meta.columns:
            msg = f"Metadata column '{cfg.cluster_method}' was not found."
            raise ValueError(msg)
        if cfg.purity_column not in meta.columns:
            msg = f"Metadata column '{cfg.purity_column}' was not found."
            raise ValueError(msg)

        group_to_samples: dict[str, list[str]] = {"normal": []}
        normal_mask = meta["Tissuetype"].astype(str).str.casefold() == "normal"
        normal_samples = meta.loc[normal_mask, sample_id_column].astype(str).tolist()
        group_to_samples["normal"] = [sample for sample in normal_samples if sample in data_columns]

        tumor_mask = (
            meta["Tissuetype"].astype(str).str.casefold() == "tumor"
        ) & (
            meta[cfg.purity_column].astype(str) == cfg.purity_value
        )
        tumor_meta = meta.loc[tumor_mask, [sample_id_column, cfg.cluster_method]].copy()
        tumor_meta["cluster_group"] = tumor_meta[cfg.cluster_method].apply(
            lambda value: _normalize_cluster_label(cfg.cluster_method or "", value)
        )
        tumor_meta = tumor_meta.loc[tumor_meta["cluster_group"].notna()].copy()

        for cluster_group, subframe in tumor_meta.groupby("cluster_group"):
            samples = subframe[sample_id_column].astype(str).tolist()
            matched_samples = [sample for sample in samples if sample in data_columns]
            if matched_samples:
                group_to_samples[str(cluster_group)] = matched_samples

        if cfg.group_order is not None:
            group_order = [group for group in cfg.group_order if group in group_to_samples]
        else:
            cluster_groups = [group for group in group_to_samples if group != "normal"]
            cluster_groups = sorted(cluster_groups, key=_group_sort_key)
            group_order = ["normal", *cluster_groups]
        count_rows = [
            {"group": group_name, "n_samples": len(group_to_samples[group_name])}
            for group_name in group_order
        ]
        return group_to_samples, group_order, pd.DataFrame(count_rows)

    default_group_order = cfg.group_order or ["normal", "GG1", "GG2", "GG3", "GG4", "GG5", "tumor"]
    group_to_samples: dict[str, list[str]] = {}
    for group_name in default_group_order:
        mask = _build_group_mask(meta, group_name, cfg.purity_column, cfg.purity_value)
        samples = meta.loc[mask, sample_id_column].astype(str).tolist()
        group_to_samples[group_name] = [sample for sample in samples if sample in data_columns]
    count_rows = [
        {"group": group_name, "n_samples": len(group_to_samples[group_name])}
        for group_name in default_group_order
    ]
    return group_to_samples, default_group_order, pd.DataFrame(count_rows)


def _build_long_frame(cfg: BoxplotConfig) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    meta = pd.read_csv(cfg.metadata_path, sep=_infer_sep(cfg.metadata_path))
    data = pd.read_csv(cfg.data_path, sep=_infer_sep(cfg.data_path))
    if cfg.feature_column not in data.columns:
        msg = f"Feature column '{cfg.feature_column}' was not found in {cfg.data_path}."
        raise ValueError(msg)

    sample_id_column = _pick_sample_column(meta, data.columns.tolist(), cfg.sample_id_column)
    data = data.set_index(cfg.feature_column)
    data = data.apply(pd.to_numeric, errors="coerce")

    group_to_samples, group_order, group_counts = _build_group_assignments(
        cfg,
        meta,
        sample_id_column,
        set(data.columns),
    )

    rows: list[dict[str, object]] = []
    missing_genes: list[str] = []
    for gene_name in cfg.gene_names:
        if gene_name not in data.index:
            missing_genes.append(gene_name)
            continue
        gene_values = data.loc[gene_name]
        for group_name in group_order:
            for sample in group_to_samples[group_name]:
                value = gene_values.get(sample, np.nan)
                if pd.notna(value):
                    rows.append(
                        {
                            "gene": gene_name,
                            "group": group_name,
                            "sample": sample,
                            "value": float(value),
                        }
                    )

    if not rows:
        msg = "No plottable gene values were found for the requested genes."
        raise ValueError(msg)

    long_frame = pd.DataFrame(rows)
    if missing_genes:
        missing_path = cfg.output_dir / f"{cfg.plot_name}_missing_genes.txt"
        missing_path.write_text("\n".join(missing_genes) + "\n", encoding="utf-8")
    return long_frame, group_order, group_counts


def _build_stats_table(cfg: BoxplotConfig, long_frame: pd.DataFrame, group_order: list[str]) -> pd.DataFrame:
    stats_rows: list[dict[str, object]] = []
    reference_group = cfg.reference_group

    for gene_name in cfg.gene_names:
        gene_frame = long_frame.loc[long_frame["gene"] == gene_name].copy()
        ref_values = gene_frame.loc[gene_frame["group"] == reference_group, "value"].dropna().tolist()
        for group_name in group_order:
            group_values = gene_frame.loc[gene_frame["group"] == group_name, "value"].dropna().tolist()
            pvalue = np.nan
            if group_name != reference_group and ref_values and group_values:
                pvalue = float(mannwhitneyu(group_values, ref_values, alternative="two-sided").pvalue)
            stats_rows.append(
                {
                    "gene": gene_name,
                    "group": group_name,
                    "n": len(group_values),
                    "median": float(np.median(group_values)) if group_values else np.nan,
                    "reference_group": reference_group,
                    "reference_n": len(ref_values),
                    "pvalue_vs_normal": pvalue,
                    "significance": _significance_stars(pvalue) if group_name != reference_group else "",
                }
            )

    return pd.DataFrame(stats_rows)


def _annotate_significance(
    ax: plt.Axes,
    gene_frame: pd.DataFrame,
    stats_table: pd.DataFrame,
    cfg: BoxplotConfig,
    group_order: list[str],
) -> None:
    y_values = gene_frame["value"].dropna()
    if y_values.empty:
        return

    ymin = float(y_values.min())
    ymax = float(y_values.max())
    yrange = ymax - ymin
    if yrange <= 0:
        yrange = max(abs(ymax), 1.0) * 0.2
    base = ymax + yrange * 0.10

    positions = {group: idx for idx, group in enumerate(group_order)}
    gene_stats = stats_table.loc[stats_table["gene"] == gene_frame["gene"].iloc[0]]
    for _, row in gene_stats.iterrows():
        group_name = str(row["group"])
        if group_name == cfg.reference_group:
            continue
        if row["significance"] == "":
            continue
        xpos = positions[group_name]
        ax.text(xpos, base, row["significance"], ha="center", va="bottom", fontsize=11, color="#8c2d04")

    ax.set_ylim(ymin - yrange * 0.05, base + yrange * 0.10)


def _plot_boxplots(
    cfg: BoxplotConfig,
    long_frame: pd.DataFrame,
    stats_table: pd.DataFrame,
    group_order: list[str],
) -> Path:
    sns.set_theme(style="whitegrid")
    n_panels = len(cfg.gene_names)
    ncols = max(1, min(cfg.figure_ncols, n_panels))
    nrows = ceil(n_panels / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.8 * ncols, 4.2 * nrows), squeeze=False)
    flat_axes = axes.flatten()

    base_palette = {
        "normal": "#6baed6",
        "GG1": "#74c476",
        "GG2": "#9e9ac8",
        "GG3": "#fd8d3c",
        "GG4": "#fc9272",
        "GG5": "#de2d26",
        "tumor": "#756bb1",
    }
    dynamic_colors = sns.color_palette("Set2", n_colors=max(len(group_order), 3)).as_hex()
    palette = {
        group_name: base_palette.get(group_name, dynamic_colors[idx % len(dynamic_colors)])
        for idx, group_name in enumerate(group_order)
    }

    for idx, gene_name in enumerate(cfg.gene_names):
        ax = flat_axes[idx]
        gene_frame = long_frame.loc[long_frame["gene"] == gene_name].copy()
        sns.boxplot(
            data=gene_frame,
            x="group",
            y="value",
            order=group_order,
            palette=palette,
            fliersize=0,
            linewidth=1.1,
            ax=ax,
        )
        sns.stripplot(
            data=gene_frame,
            x="group",
            y="value",
            order=group_order,
            color="#1b1b1b",
            alpha=0.45,
            size=3.2,
            jitter=0.22,
            ax=ax,
        )
        _annotate_significance(ax, gene_frame, stats_table, cfg, group_order)
        ax.set_title(gene_name, fontsize=12, weight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("log2 ratio")
        ax.tick_params(axis="x", rotation=35)

    for idx in range(len(cfg.gene_names), len(flat_axes)):
        flat_axes[idx].axis("off")

    fig.tight_layout()
    output_path = cfg.output_dir / f"{cfg.plot_name}.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def run_boxplot(config_path: Path) -> Path:
    config = _read_config(config_path)
    cfg = _build_boxplot_config(config)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    long_frame, group_order, group_counts = _build_long_frame(cfg)
    group_counts.to_csv(cfg.output_dir / f"{cfg.plot_name}_group_counts.tsv", sep="\t", index=False)
    stats_table = _build_stats_table(cfg, long_frame, group_order)
    stats_table.to_csv(cfg.output_dir / f"{cfg.plot_name}_stats.tsv", sep="\t", index=False)
    long_frame.to_csv(cfg.output_dir / f"{cfg.plot_name}_values.tsv", sep="\t", index=False)
    output_path = _plot_boxplots(cfg, long_frame, stats_table, group_order)
    print(group_counts.to_string(index=False))
    print(f"boxplot output written to {output_path}")
    return output_path
