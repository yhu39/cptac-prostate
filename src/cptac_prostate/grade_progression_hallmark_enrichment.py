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
    parser.read(config_path, encoding="utf-8-sig")
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


def _parse_csv_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _read_gene_list(path: Path) -> set[str]:
    if not path.exists():
        return set()
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    return {line.strip() for line in text.splitlines() if line.strip()}


@dataclass(frozen=True)
class GradeProgressionHallmarkConfig:
    run_dir: Path
    output_dir: Path
    plot_prefix: str = "grade_progression_hallmark_enrichment"
    summary_filename: str = "grade_progression_hallmark_enrichment_summary.md"
    grades: tuple[str, ...] = ("GG2", "GG3", "GG4", "GG5")
    up_gene_label: str = "S-U"
    gene_scope: str = "tumor_overlap_su"
    enrichment_method: str = "gseapy"
    gene_sets: str = "MSigDB_Hallmark_2020"
    organism: str = "human"
    adj_p_cutoff: float = 0.05
    top_n_terms: int = 12
    min_overlap_hits: int = 2


def _build_config(config: configparser.ConfigParser) -> GradeProgressionHallmarkConfig:
    grades = _parse_csv_list(_get_optional_value(config, "settings", "grades", "GG2,GG3,GG4,GG5"))
    return GradeProgressionHallmarkConfig(
        run_dir=_get_required_path(config, "input", "run_dir"),
        output_dir=_get_required_path(config, "output", "output_dir"),
        plot_prefix=_get_optional_value(
            config,
            "settings",
            "plot_prefix",
            "grade_progression_hallmark_enrichment",
        ),
        summary_filename=_get_optional_value(
            config,
            "settings",
            "summary_filename",
            "grade_progression_hallmark_enrichment_summary.md",
        ),
        grades=grades or ("GG2", "GG3", "GG4", "GG5"),
        up_gene_label=_get_optional_value(config, "settings", "up_gene_label", "S-U"),
        gene_scope=_get_optional_value(config, "settings", "gene_scope", "tumor_overlap_su"),
        enrichment_method=_get_optional_value(config, "settings", "enrichment_method", "gseapy"),
        gene_sets=_get_optional_value(config, "settings", "gene_sets", "MSigDB_Hallmark_2020"),
        organism=_get_optional_value(config, "settings", "organism", "human"),
        adj_p_cutoff=float(_get_optional_value(config, "settings", "adj_p_cutoff", "0.05")),
        top_n_terms=int(float(_get_optional_value(config, "settings", "top_n_terms", "12"))),
        min_overlap_hits=int(float(_get_optional_value(config, "settings", "min_overlap_hits", "2"))),
    )


def _grade_gene_sets(cfg: GradeProgressionHallmarkConfig) -> dict[str, list[str]]:
    tumor_genes = _read_gene_list(cfg.run_dir / "genes_tumor_vs_normal" / f"{cfg.up_gene_label}_genes.txt")
    grade_gene_sets: dict[str, list[str]] = {}
    for grade in cfg.grades:
        grade_genes = _read_gene_list(cfg.run_dir / f"genes_{grade}_vs_normal" / f"{cfg.up_gene_label}_genes.txt")
        if cfg.gene_scope == "tumor_overlap_su":
            selected = sorted(grade_genes & tumor_genes)
        elif cfg.gene_scope == "grade_su":
            selected = sorted(grade_genes)
        else:
            msg = f"Unsupported gene_scope: {cfg.gene_scope}"
            raise ValueError(msg)
        grade_gene_sets[grade] = selected
    return grade_gene_sets


def _parse_overlap_hits(value: str) -> int:
    if not value or "/" not in value:
        return 0
    numerator, _ = value.split("/", 1)
    return int(float(numerator))


def _run_single_enrichment(
    gene_list: list[str],
    cfg: GradeProgressionHallmarkConfig,
) -> pd.DataFrame:
    try:
        import gseapy
    except ImportError as exc:
        msg = "gseapy is required for grade progression hallmark enrichment."
        raise ImportError(msg) from exc

    if cfg.enrichment_method.casefold() != "gseapy":
        msg = f"Unsupported enrichment_method: {cfg.enrichment_method}"
        raise ValueError(msg)
    if len(gene_list) < 2:
        return pd.DataFrame()

    gene_sets_input: str | Path = cfg.gene_sets
    gene_sets_path = Path(cfg.gene_sets)
    if gene_sets_path.exists():
        gene_sets_input = gene_sets_path

    enrich_kwargs: dict[str, object] = {
        "gene_list": gene_list,
        "gene_sets": gene_sets_input,
        "outdir": None,
        "cutoff": 1.0,
    }
    if isinstance(gene_sets_input, str):
        enrich_kwargs["organism"] = cfg.organism

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


def _run_enrichment(cfg: GradeProgressionHallmarkConfig) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    grade_gene_sets = _grade_gene_sets(cfg)
    result_frames: list[pd.DataFrame] = []
    for grade, gene_list in grade_gene_sets.items():
        grade_result = _run_single_enrichment(gene_list, cfg)
        if grade_result.empty:
            grade_result = pd.DataFrame(
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
        grade_result["grade"] = grade
        grade_result["input_gene_count"] = len(gene_list)
        grade_result["input_genes"] = ";".join(gene_list)
        grade_result["gene_scope"] = cfg.gene_scope
        result_frames.append(grade_result)

    all_results = pd.concat(result_frames, ignore_index=True)
    return all_results, grade_gene_sets


def _build_plot_frame(
    all_results: pd.DataFrame,
    cfg: GradeProgressionHallmarkConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if all_results.empty:
        return pd.DataFrame(), pd.DataFrame()

    sig_results = all_results.loc[
        (all_results["is_significant"])
        & (all_results["overlap_hits"] >= cfg.min_overlap_hits)
    ].copy()
    if sig_results.empty:
        return sig_results, pd.DataFrame()

    top_terms = (
        sig_results.groupby("Term", as_index=False)["neg_log10_fdr"]
        .max()
        .sort_values("neg_log10_fdr", ascending=False)
        .head(cfg.top_n_terms)["Term"]
        .tolist()
    )
    filtered = sig_results.loc[sig_results["Term"].isin(top_terms)].copy()
    filtered["term_label"] = filtered["Term"].map(lambda value: fill(str(value), width=28))

    term_order = (
        filtered.groupby("term_label", as_index=False)["neg_log10_fdr"]
        .max()
        .sort_values("neg_log10_fdr", ascending=False)["term_label"]
        .tolist()
    )
    grade_order = list(cfg.grades)
    filtered["term_label"] = pd.Categorical(filtered["term_label"], categories=term_order, ordered=True)
    filtered["grade"] = pd.Categorical(filtered["grade"], categories=grade_order, ordered=True)
    filtered = filtered.sort_values(["term_label", "grade"]).reset_index(drop=True)

    matrix = (
        filtered.pivot(index="term_label", columns="grade", values="neg_log10_fdr")
        .reindex(index=term_order, columns=grade_order)
        .fillna(0.0)
        .reset_index()
    )
    return filtered, matrix


def _save_tables(
    cfg: GradeProgressionHallmarkConfig,
    all_results: pd.DataFrame,
    plot_frame: pd.DataFrame,
    matrix: pd.DataFrame,
    grade_gene_sets: dict[str, list[str]],
) -> dict[str, Path]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "all_terms": cfg.output_dir / f"{cfg.plot_prefix}_all_terms.tsv",
        "significant_terms": cfg.output_dir / f"{cfg.plot_prefix}_significant_terms.tsv",
        "matrix": cfg.output_dir / f"{cfg.plot_prefix}_matrix.tsv",
        "input_sets": cfg.output_dir / f"{cfg.plot_prefix}_input_gene_sets.tsv",
        "png": cfg.output_dir / f"{cfg.plot_prefix}.png",
        "pdf_editable": cfg.output_dir / f"{cfg.plot_prefix}_editable_text.pdf",
        "summary": cfg.output_dir / cfg.summary_filename,
    }

    all_results.to_csv(paths["all_terms"], sep="\t", index=False)
    plot_frame.to_csv(paths["significant_terms"], sep="\t", index=False)
    matrix.to_csv(paths["matrix"], sep="\t", index=False)
    pd.DataFrame(
        {
            "grade": list(grade_gene_sets.keys()),
            "gene_count": [len(genes) for genes in grade_gene_sets.values()],
            "genes": [";".join(genes) for genes in grade_gene_sets.values()],
        }
    ).to_csv(paths["input_sets"], sep="\t", index=False)
    return paths


def _plot_enrichment(
    cfg: GradeProgressionHallmarkConfig,
    plot_frame: pd.DataFrame,
    grade_gene_sets: dict[str, list[str]],
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
                f"No significant Hallmark terms passed FDR <= {cfg.adj_p_cutoff:.3g} "
                f"and overlap hits >= {cfg.min_overlap_hits}."
            ),
            ha="center",
            va="center",
            fontsize=12,
        )
        fig.savefig(paths["png"], dpi=300, bbox_inches="tight")
        fig.savefig(paths["pdf_editable"], dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    term_order = plot_frame["term_label"].cat.categories.tolist()
    grade_order = list(cfg.grades)
    y_positions = {term: idx for idx, term in enumerate(term_order)}
    x_positions = {grade: idx for idx, grade in enumerate(grade_order)}

    fig, ax = plt.subplots(figsize=(1.8 * len(grade_order) + 5.2, 0.55 * len(term_order) + 2.6))
    ax.set_facecolor("white")

    max_hits = max(int(plot_frame["overlap_hits"].max()), 1)
    scatter = ax.scatter(
        plot_frame["grade"].map(x_positions),
        plot_frame["term_label"].map(y_positions),
        s=plot_frame["overlap_hits"] / max_hits * 720 + 80,
        c=plot_frame["neg_log10_fdr"],
        cmap="Reds",
        edgecolors="#404040",
        linewidths=0.7,
    )

    ax.set_xticks(range(len(grade_order)))
    ax.set_xticklabels(grade_order, fontsize=11)
    ax.set_yticks(range(len(term_order)))
    ax.set_yticklabels(term_order, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Grade", fontsize=11)
    ax.set_ylabel("MSigDB Hallmark term", fontsize=11)
    ax.set_title(
        "Significant MSigDB Hallmark enrichment across GG2-GG5",
        loc="left",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )

    ax.set_xticks(np.arange(-0.5, len(grade_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(term_order), 1), minor=True)
    ax.grid(which="minor", color="#e6e6e6", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("-log10 adjusted P-value", fontsize=10)

    legend_counts = sorted({min(max_hits, count) for count in [2, 3, 4, 5] if count <= max_hits or count == 2})
    if legend_counts:
        handles = [
            ax.scatter([], [], s=count / max_hits * 720 + 80, color="#d95f02", edgecolors="#404040", linewidths=0.7)
            for count in legend_counts
        ]
        ax.legend(
            handles,
            [str(count) for count in legend_counts],
            title="Overlap hits",
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.18, 1.02),
            fontsize=10,
            title_fontsize=10,
        )

    sample_text = ", ".join(f"{grade} genes={len(grade_gene_sets[grade])}" for grade in grade_order)
    fig.text(
        0.01,
        0.01,
        (
            f"Input scope: {cfg.gene_scope}. Library: {cfg.gene_sets}. "
            f"FDR cutoff <= {cfg.adj_p_cutoff:.3g}, min overlap hits >= {cfg.min_overlap_hits}. "
            f"{sample_text}."
        ),
        ha="left",
        va="bottom",
        fontsize=9,
    )

    fig.savefig(paths["png"], dpi=300, bbox_inches="tight")
    fig.savefig(paths["pdf_editable"], dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_summary(
    cfg: GradeProgressionHallmarkConfig,
    all_results: pd.DataFrame,
    plot_frame: pd.DataFrame,
    grade_gene_sets: dict[str, list[str]],
    paths: dict[str, Path],
) -> None:
    if plot_frame.empty:
        summary_lines = [
            "# Grade Progression Hallmark Enrichment Summary",
            "",
            "## Result",
            (
                f"No significant Hallmark terms passed adjusted P-value <= {cfg.adj_p_cutoff:.3g} "
                f"and overlap hits >= {cfg.min_overlap_hits}."
            ),
        ]
        paths["summary"].write_text("\n".join(summary_lines) + "\n", encoding="utf-8-sig")
        return

    top_by_grade: list[str] = []
    for grade in cfg.grades:
        grade_hits = (
            plot_frame.loc[plot_frame["grade"] == grade, ["Term", "Adjusted P-value", "Overlap"]]
            .sort_values("Adjusted P-value", ascending=True)
            .head(3)
        )
        if grade_hits.empty:
            top_by_grade.append(f"- {grade}: none")
            continue
        text = ", ".join(
            f"{row['Term']} (FDR={row['Adjusted P-value']:.3g}, overlap={row['Overlap']})"
            for _, row in grade_hits.iterrows()
        )
        top_by_grade.append(f"- {grade}: {text}")

    significant_term_count = plot_frame["Term"].nunique()
    all_sig_count = int((all_results["Adjusted P-value"] <= cfg.adj_p_cutoff).sum())
    summary_lines = [
        "# Grade Progression Hallmark Enrichment Summary",
        "",
        "## Generated files",
        f"- Figure PNG: `{paths['png'].name}`",
        f"- Editable-text PDF: `{paths['pdf_editable'].name}`",
        f"- All enrichment terms: `{paths['all_terms'].name}`",
        f"- Significant terms used for plotting: `{paths['significant_terms'].name}`",
        f"- Plot matrix: `{paths['matrix'].name}`",
        f"- Input gene sets: `{paths['input_sets'].name}`",
        "",
        "## Method",
        f"- Backend: `{cfg.enrichment_method}`",
        f"- Library: `{cfg.gene_sets}`",
        f"- Organism: `{cfg.organism}`",
        f"- Gene scope: `{cfg.gene_scope}`",
        f"- Significance threshold: adjusted P-value <= {cfg.adj_p_cutoff:.3g}",
        f"- Minimum overlap hits displayed: {cfg.min_overlap_hits}",
        "",
        "## Interpretation",
        (
            f"- Across all grades, {all_sig_count} grade-term pairs passed the significance cutoff, "
            f"representing {significant_term_count} unique Hallmark pathways in the plotted version."
        ),
        (
            "- This figure is a database-driven enrichment view, unlike the manual functional-module figure. "
            "Rows are MSigDB Hallmark terms and each dot shows that a term is significantly enriched in that grade."
        ),
        "- Dot color represents enrichment significance as `-log10(adjusted P-value)`.",
        "- Dot size represents how many input proteins from that grade hit the Hallmark term.",
        "",
        "## Top significant Hallmark terms by grade",
        *top_by_grade,
        "",
        "## Input gene counts",
        *[f"- {grade}: {len(grade_gene_sets[grade])} genes" for grade in cfg.grades],
    ]
    paths["summary"].write_text("\n".join(summary_lines) + "\n", encoding="utf-8-sig")


def run_grade_progression_hallmark_enrichment(config_path: Path) -> Path:
    config = _read_config(config_path)
    cfg = _build_config(config)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    all_results, grade_gene_sets = _run_enrichment(cfg)
    plot_frame, matrix = _build_plot_frame(all_results, cfg)
    paths = _save_tables(cfg, all_results, plot_frame, matrix, grade_gene_sets)
    _plot_enrichment(cfg, plot_frame, grade_gene_sets, paths)
    _write_summary(cfg, all_results, plot_frame, grade_gene_sets, paths)
    print(f"grade_progression_hallmark_enrichment output written to {paths['png']}")
    return paths["png"]
