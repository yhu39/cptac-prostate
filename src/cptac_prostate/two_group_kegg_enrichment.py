from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill
import time

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
        raise ValueError(f"Config file is missing [{section}] {option}.")
    return Path(_strip_quotes(config.get(section, option)))


def _get_required_value(config: configparser.ConfigParser, section: str, option: str) -> str:
    if not config.has_section(section) or not config.has_option(section, option):
        raise ValueError(f"Config file is missing [{section}] {option}.")
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


def _get_optional_int(
    config: configparser.ConfigParser,
    section: str,
    option: str,
    default: int,
) -> int:
    if not config.has_section(section) or not config.has_option(section, option):
        return default
    return int(_strip_quotes(config.get(section, option)))


def _parse_csv_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_overlap_hits(value: str) -> int:
    if not value or "/" not in str(value):
        return 0
    numerator, _ = str(value).split("/", 1)
    return int(float(numerator))


def _read_gene_list(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _normalize_organism(value: str) -> str:
    mapping = {
        "human": "human",
        "homo sapiens": "human",
        "hs": "human",
        "hsapiens": "human",
        "mouse": "mouse",
        "mus musculus": "mouse",
        "mm": "mouse",
    }
    key = value.strip().casefold()
    return mapping.get(key, value)


@dataclass
class TwoGroupKeggConfig:
    genes_dir: Path
    output_dir: Path
    plot_prefix: str = "tumor_vs_normal_kegg_enrichment"
    summary_filename: str = "tumor_vs_normal_kegg_enrichment_summary.md"
    gene_sets: str = "KEGG_2021_Human"
    organism: str = "Human"
    groups: tuple[str, ...] = ("S-U", "S-D")
    adj_p_cutoff: float = 0.05
    min_overlap_hits: int = 2
    top_n_terms: int = 12
    figure_width_scale: float = 1.0


def _load_config(config_path: Path) -> TwoGroupKeggConfig:
    config = _read_config(config_path)
    return TwoGroupKeggConfig(
        genes_dir=_get_required_path(config, "input", "genes_dir"),
        output_dir=_get_required_path(config, "output", "output_dir"),
        plot_prefix=_get_optional_value(config, "settings", "plot_prefix", "tumor_vs_normal_kegg_enrichment"),
        summary_filename=_get_optional_value(
            config, "settings", "summary_filename", "tumor_vs_normal_kegg_enrichment_summary.md"
        ),
        gene_sets=_get_optional_value(config, "settings", "gene_sets", "KEGG_2021_Human"),
        organism=_get_optional_value(config, "settings", "organism", "Human"),
        groups=_parse_csv_list(_get_optional_value(config, "settings", "groups", "S-U,S-D")),
        adj_p_cutoff=_get_optional_float(config, "settings", "adj_p_cutoff", 0.05),
        min_overlap_hits=_get_optional_int(config, "settings", "min_overlap_hits", 2),
        top_n_terms=_get_optional_int(config, "settings", "top_n_terms", 12),
        figure_width_scale=_get_optional_float(config, "settings", "figure_width_scale", 1.0),
    )


def _run_single_enrichment(gene_list: list[str], cfg: TwoGroupKeggConfig) -> pd.DataFrame:
    try:
        import gseapy
    except ImportError as exc:
        raise ImportError("gseapy is required for KEGG enrichment.") from exc

    if len(gene_list) < 2:
        return pd.DataFrame()

    enrich_kwargs: dict[str, object] = {
        "gene_list": gene_list,
        "gene_sets": cfg.gene_sets,
        "organism": _normalize_organism(cfg.organism),
        "outdir": None,
        "cutoff": 1.0,
    }

    last_error: Exception | None = None
    for attempt in range(5):
        try:
            result = gseapy.enrichr(**enrich_kwargs).results.copy()
            break
        except Exception as exc:
            last_error = exc
            if "429" not in str(exc):
                raise
            time.sleep(2**attempt)
    else:
        assert last_error is not None
        raise last_error

    if result.empty:
        return result

    result["overlap_hits"] = result["Overlap"].astype(str).map(_parse_overlap_hits)
    result["neg_log10_fdr"] = -np.log10(result["Adjusted P-value"].clip(lower=np.finfo(float).tiny))
    result["is_significant"] = result["Adjusted P-value"] <= cfg.adj_p_cutoff
    return result


def _run_enrichment(cfg: TwoGroupKeggConfig) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    gene_sets: dict[str, list[str]] = {}
    frames: list[pd.DataFrame] = []
    for group_name in cfg.groups:
        genes = _read_gene_list(cfg.genes_dir / f"{group_name}_genes.txt")
        gene_sets[group_name] = genes
        result = _run_single_enrichment(genes, cfg)
        if result.empty:
            result = pd.DataFrame(
                columns=[
                    "Gene_set",
                    "Term",
                    "Overlap",
                    "P-value",
                    "Adjusted P-value",
                    "Old P-value",
                    "Old Adjusted P-value",
                    "Odds Ratio",
                    "Combined Score",
                    "Genes",
                    "overlap_hits",
                    "neg_log10_fdr",
                    "is_significant",
                ]
            )
        result["group_name"] = group_name
        result["input_gene_count"] = len(genes)
        result["input_genes"] = ";".join(genes)
        frames.append(result)
    return pd.concat(frames, ignore_index=True), gene_sets


def _build_plot_frame(all_results: pd.DataFrame, cfg: TwoGroupKeggConfig) -> pd.DataFrame:
    if all_results.empty:
        return pd.DataFrame()

    sig = all_results.loc[
        (all_results["is_significant"]) & (all_results["overlap_hits"] >= cfg.min_overlap_hits)
    ].copy()
    if sig.empty:
        return sig

    chosen_terms: list[str] = []
    for group_name in cfg.groups:
        subset = sig.loc[sig["group_name"] == group_name].sort_values(
            ["Adjusted P-value", "neg_log10_fdr"], ascending=[True, False]
        )
        chosen_terms.extend(subset["Term"].head(cfg.top_n_terms).tolist())
    chosen_terms = list(dict.fromkeys(chosen_terms))
    plot_frame = sig.loc[sig["Term"].isin(chosen_terms)].copy()
    if plot_frame.empty:
        return plot_frame

    plot_frame["term_label"] = plot_frame["Term"].map(lambda value: fill(str(value), width=36))
    term_order = (
        plot_frame.groupby("term_label", as_index=False)["neg_log10_fdr"]
        .max()
        .sort_values("neg_log10_fdr", ascending=False)["term_label"]
        .tolist()
    )
    plot_frame["term_label"] = pd.Categorical(plot_frame["term_label"], categories=term_order, ordered=True)
    plot_frame["group_name"] = pd.Categorical(
        plot_frame["group_name"], categories=list(cfg.groups), ordered=True
    )
    return plot_frame.sort_values(["term_label", "group_name"]).reset_index(drop=True)


def _save_tables(
    cfg: TwoGroupKeggConfig,
    all_results: pd.DataFrame,
    plot_frame: pd.DataFrame,
    gene_sets: dict[str, list[str]],
) -> dict[str, Path]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "all_terms": cfg.output_dir / f"{cfg.plot_prefix}_all_terms.tsv",
        "significant_terms": cfg.output_dir / f"{cfg.plot_prefix}_significant_terms.tsv",
        "input_sets": cfg.output_dir / f"{cfg.plot_prefix}_input_gene_sets.tsv",
        "png": cfg.output_dir / f"{cfg.plot_prefix}.png",
        "pdf": cfg.output_dir / f"{cfg.plot_prefix}_editable_text.pdf",
        "summary": cfg.output_dir / cfg.summary_filename,
        "config": cfg.output_dir / "config_tumor_vs_normal_kegg_enrichment.ini",
    }
    all_results.to_csv(paths["all_terms"], sep="\t", index=False)
    plot_frame.to_csv(paths["significant_terms"], sep="\t", index=False)
    pd.DataFrame(
        {
            "group_name": list(gene_sets.keys()),
            "gene_count": [len(genes) for genes in gene_sets.values()],
            "genes": [";".join(genes) for genes in gene_sets.values()],
        }
    ).to_csv(paths["input_sets"], sep="\t", index=False)
    return paths


def _plot_enrichment(
    cfg: TwoGroupKeggConfig,
    plot_frame: pd.DataFrame,
    gene_sets: dict[str, list[str]],
    paths: dict[str, Path],
) -> None:
    sns.set_theme(style="white")
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    if plot_frame.empty:
        fig, ax = plt.subplots(figsize=(8.0, 3.0))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            (
                f"No significant KEGG terms passed FDR <= {cfg.adj_p_cutoff:.3g} "
                f"and overlap hits >= {cfg.min_overlap_hits}."
            ),
            ha="center",
            va="center",
            fontsize=12,
        )
        fig.savefig(paths["png"], dpi=300, bbox_inches="tight")
        fig.savefig(paths["pdf"], dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    term_order = plot_frame["term_label"].cat.categories.tolist()
    group_order = list(cfg.groups)
    y_positions = {term: idx for idx, term in enumerate(term_order)}
    x_positions = {group: idx for idx, group in enumerate(group_order)}
    group_labels = {"S-U": "S-U", "S-D": "S-D"}

    base_width = 0.9 * len(group_order) + 2.4
    fig, ax = plt.subplots(
        figsize=(max(4.5, base_width * cfg.figure_width_scale), 0.52 * len(term_order) + 2.8)
    )
    ax.set_facecolor("white")
    max_hits = max(int(plot_frame["overlap_hits"].max()), 1)
    scatter = ax.scatter(
        plot_frame["group_name"].map(x_positions),
        plot_frame["term_label"].map(y_positions),
        s=plot_frame["overlap_hits"] / max_hits * 720 + 80,
        c=plot_frame["neg_log10_fdr"],
        cmap="Reds",
        edgecolors="#404040",
        linewidths=0.7,
    )
    ax.set_xticks(range(len(group_order)))
    ax.set_xticklabels([group_labels.get(group, group) for group in group_order], fontsize=25)
    ax.set_yticks(range(len(term_order)))
    ax.set_yticklabels(term_order, fontsize=23)
    ax.invert_yaxis()
    ax.set_xlabel("Tumor-vs-normal direction", fontsize=24)
    ax.set_ylabel("KEGG pathway", fontsize=24)
    ax.set_title(
        "KEGG enrichment for tumor-vs-normal S-U and S-D proteins",
        loc="left",
        fontsize=30,
        fontweight="bold",
        pad=10,
    )
    ax.set_xticks(np.arange(-0.5, len(group_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(term_order), 1), minor=True)
    ax.grid(which="minor", color="#e6e6e6", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("-log10 adjusted P-value", fontsize=22)
    colorbar.ax.tick_params(labelsize=20)

    size_values = sorted({value for value in plot_frame["overlap_hits"].tolist() if value > 0})
    if len(size_values) > 3:
        size_values = [size_values[0], size_values[len(size_values) // 2], size_values[-1]]
    size_handles = [
        plt.scatter([], [], s=value / max_hits * 720 + 80, facecolor="white", edgecolor="#404040")
        for value in size_values
    ]
    size_labels = [str(value) for value in size_values]
    legend = ax.legend(
        size_handles,
        size_labels,
        title="Overlap hits",
        loc="upper left",
        bbox_to_anchor=(1.18, 0.35),
        frameon=False,
        borderaxespad=0.0,
    )
    legend.get_title().set_fontsize(20)
    for text in legend.get_texts():
        text.set_fontsize(20)

    fig.text(
        0.02,
        0.02,
        (
            f"S-U genes: {len(gene_sets.get('S-U', []))}; "
            f"S-D genes: {len(gene_sets.get('S-D', []))}. "
            "Dot size represents overlap hits."
        ),
        fontsize=19.6,
    )
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight")
    fig.savefig(paths["pdf"], dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_summary(
    cfg: TwoGroupKeggConfig,
    plot_frame: pd.DataFrame,
    gene_sets: dict[str, list[str]],
    paths: dict[str, Path],
) -> None:
    lines = [
        "# Tumor vs Normal KEGG Enrichment Summary",
        "",
        f"- Gene-set library: `{cfg.gene_sets}`",
        f"- S-U gene count: {len(gene_sets.get('S-U', []))}",
        f"- S-D gene count: {len(gene_sets.get('S-D', []))}",
        f"- Plot PNG: `{paths['png'].name}`",
        f"- Editable-text PDF: `{paths['pdf'].name}`",
        f"- All terms table: `{paths['all_terms'].name}`",
        f"- Significant terms table: `{paths['significant_terms'].name}`",
        "",
    ]
    if plot_frame.empty:
        lines.append(
            f"- No significant KEGG terms passed FDR <= {cfg.adj_p_cutoff:.3g} and overlap hits >= {cfg.min_overlap_hits}."
        )
    else:
        for group_name in cfg.groups:
            subset = plot_frame.loc[plot_frame["group_name"] == group_name].copy()
            lines.append(f"## {group_name}")
            if subset.empty:
                lines.append("- No significant KEGG terms passed the filter.")
            else:
                for _, row in subset.sort_values("Adjusted P-value", ascending=True).head(8).iterrows():
                    lines.append(
                        f"- {row['Term']} | FDR={row['Adjusted P-value']:.3g} | overlap={int(row['overlap_hits'])}"
                    )
            lines.append("")
    paths["summary"].write_text("\n".join(lines), encoding="utf-8")


def _save_config_snapshot(cfg: TwoGroupKeggConfig, path: Path) -> None:
    parser = configparser.ConfigParser()
    parser["task"] = {"name": "two_group_kegg_enrichment"}
    parser["input"] = {"genes_dir": str(cfg.genes_dir)}
    parser["output"] = {"output_dir": str(cfg.output_dir)}
    parser["settings"] = {
        "plot_prefix": cfg.plot_prefix,
        "summary_filename": cfg.summary_filename,
        "gene_sets": cfg.gene_sets,
        "organism": cfg.organism,
        "groups": ",".join(cfg.groups),
        "adj_p_cutoff": str(cfg.adj_p_cutoff),
        "min_overlap_hits": str(cfg.min_overlap_hits),
        "top_n_terms": str(cfg.top_n_terms),
        "figure_width_scale": str(cfg.figure_width_scale),
    }
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        parser.write(handle)


def run_two_group_kegg_enrichment(config_path: Path) -> Path:
    cfg = _load_config(config_path)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    all_results, gene_sets = _run_enrichment(cfg)
    plot_frame = _build_plot_frame(all_results, cfg)
    paths = _save_tables(cfg, all_results, plot_frame, gene_sets)
    _plot_enrichment(cfg, plot_frame, gene_sets, paths)
    _write_summary(cfg, plot_frame, gene_sets, paths)
    _save_config_snapshot(cfg, paths["config"])
    print(f"two_group_kegg_enrichment output written to {paths['png']}")
    return paths["png"]
