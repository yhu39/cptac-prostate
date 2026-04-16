from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


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


def _get_optional_value(
    config: configparser.ConfigParser,
    section: str,
    option: str,
    default: str,
) -> str:
    if not config.has_section(section) or not config.has_option(section, option):
        return default
    return _strip_quotes(config.get(section, option))


def _get_optional_int(
    config: configparser.ConfigParser,
    section: str,
    option: str,
    default: int,
) -> int:
    if not config.has_section(section) or not config.has_option(section, option):
        return default
    return int(_strip_quotes(config.get(section, option)))


def _get_optional_float(
    config: configparser.ConfigParser,
    section: str,
    option: str,
    default: float,
) -> float:
    if not config.has_section(section) or not config.has_option(section, option):
        return default
    return float(_strip_quotes(config.get(section, option)))


def _read_table_auto(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {path}")


@dataclass
class PrerankGseaConfig:
    input_table: Path
    output_dir: Path
    gene_column: str = "Gene"
    rank_column: str | None = None
    rank_metric_name: str | None = None
    group_a_label: str = "group A"
    group_b_label: str = "group B"
    gene_sets: str = "MSigDB_Hallmark_2020"
    organism: str = "human"
    permutation_num: int = 1000
    min_size: int = 15
    max_size: int = 500
    seed: int = 123
    top_plot_n: int = 5


def _load_config(config_path: Path) -> PrerankGseaConfig:
    config = _read_config(config_path)
    return PrerankGseaConfig(
        input_table=_get_required_path(config, "input", "input_table"),
        output_dir=_get_required_path(config, "output", "output_dir"),
        gene_column=_get_optional_value(config, "settings", "gene_column", "Gene"),
        rank_column=_get_optional_value(config, "settings", "rank_column", "") or None,
        rank_metric_name=_get_optional_value(config, "settings", "rank_metric_name", "") or None,
        group_a_label=_get_optional_value(config, "settings", "group_a_label", "group A"),
        group_b_label=_get_optional_value(config, "settings", "group_b_label", "group B"),
        gene_sets=_get_optional_value(config, "settings", "gene_sets", "MSigDB_Hallmark_2020"),
        organism=_get_optional_value(config, "settings", "organism", "human"),
        permutation_num=_get_optional_int(config, "settings", "permutation_num", 1000),
        min_size=_get_optional_int(config, "settings", "min_size", 15),
        max_size=_get_optional_int(config, "settings", "max_size", 500),
        seed=_get_optional_int(config, "settings", "seed", 123),
        top_plot_n=_get_optional_int(config, "settings", "top_plot_n", 5),
    )


def _pick_rank_column(df: pd.DataFrame) -> tuple[str, str]:
    t_candidates = [
        column for column in df.columns
        if re.search(r"(^|[^A-Za-z])(moderated.?t|t.?stat|t.?value|tstat|statistic)($|[^A-Za-z])", str(column), re.I)
        and "p-value" not in str(column).casefold()
    ]
    if t_candidates:
        return t_candidates[0], f"Using first-choice ranking metric from column `{t_candidates[0]}`."

    p_candidates = [column for column in df.columns if "p-value" in str(column).casefold() or str(column).casefold() == "p"]
    fc_candidates = [column for column in df.columns if "log2fc" in str(column).casefold()]
    if p_candidates and fc_candidates:
        return "__signed_neg_log10_p__", (
            "No t statistic column was found. Using second-choice ranking metric: "
            "sign(Log2FC) * -log10(p-value)."
        )

    if fc_candidates:
        return fc_candidates[0], f"No t or p-value-based ranking available. Using third-choice metric `{fc_candidates[0]}`."

    raise ValueError(
        "Could not infer a ranking metric. Expected a t statistic, or both p-value and log2 fold change, or log2 fold change alone."
    )


def _prepare_ranked_table(cfg: PrerankGseaConfig) -> tuple[pd.DataFrame, str]:
    df = _read_table_auto(cfg.input_table)
    if cfg.gene_column not in df.columns:
        raise ValueError(f"Required gene column `{cfg.gene_column}` was not found in {cfg.input_table}.")

    rank_column = cfg.rank_column
    metric_note = cfg.rank_metric_name
    if rank_column is None:
        rank_column, metric_note = _pick_rank_column(df)
    elif rank_column not in df.columns:
        raise ValueError(f"Requested rank column `{rank_column}` was not found in {cfg.input_table}.")

    gene_series = df[cfg.gene_column].astype(str).str.strip()
    working = pd.DataFrame({"gene": gene_series})

    if rank_column == "__signed_neg_log10_p__":
        p_candidates = [column for column in df.columns if "p-value" in str(column).casefold() or str(column).casefold() == "p"]
        fc_candidates = [column for column in df.columns if "log2fc" in str(column).casefold()]
        p_col = p_candidates[0]
        fc_col = fc_candidates[0]
        pvals = pd.to_numeric(df[p_col], errors="coerce")
        fc = pd.to_numeric(df[fc_col], errors="coerce")
        working["ranking_metric"] = -np.log10(pvals.clip(lower=np.finfo(float).tiny)) * np.sign(fc)
        metric_note = metric_note or f"Ranking metric = sign({fc_col}) * -log10({p_col})."
    else:
        working["ranking_metric"] = pd.to_numeric(df[rank_column], errors="coerce")
        metric_note = metric_note or f"Ranking metric = `{rank_column}`."

    working = working.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=["gene", "ranking_metric"]).copy()
    working = working.loc[working["gene"] != ""].copy()
    working["abs_metric"] = working["ranking_metric"].abs()
    working = (
        working.sort_values(["gene", "abs_metric"], ascending=[True, False])
        .drop_duplicates(subset=["gene"], keep="first")
        .drop(columns=["abs_metric"])
        .sort_values("ranking_metric", ascending=False)
        .reset_index(drop=True)
    )
    if working.empty:
        raise ValueError("No valid ranked genes remained after removing missing, infinite, and duplicate values.")
    return working, metric_note


def _normalize_result_table(res2d: pd.DataFrame) -> pd.DataFrame:
    results = res2d.copy().reset_index(drop=True)
    rename_map = {
        "Name": "Name",
        "Term": "Term",
        "ES": "ES",
        "NES": "NES",
        "NOM p-val": "NOM p-val",
        "FDR q-val": "FDR q-val",
        "FWER p-val": "FWER p-val",
        "Tag %": "Tag %",
        "Gene %": "Gene %",
        "Lead_genes": "Lead_genes",
        "ledge_genes": "Lead_genes",
    }
    results = results.rename(columns={column: rename_map.get(column, column) for column in results.columns})
    for column in ["ES", "NES", "NOM p-val", "FDR q-val", "FWER p-val"]:
        if column in results.columns:
            results[column] = pd.to_numeric(results[column], errors="coerce")
    if "Lead_genes" not in results.columns:
        results["Lead_genes"] = ""
    return results


def _sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def _run_prerank(cfg: PrerankGseaConfig, ranked: pd.DataFrame):
    import gseapy

    return gseapy.prerank(
        rnk=ranked[["gene", "ranking_metric"]],
        gene_sets=cfg.gene_sets,
        permutation_num=cfg.permutation_num,
        min_size=cfg.min_size,
        max_size=cfg.max_size,
        seed=cfg.seed,
        outdir=None,
        format="png",
        verbose=True,
    )


def _save_tables(cfg: PrerankGseaConfig, ranked: pd.DataFrame, results: pd.DataFrame) -> dict[str, Path]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "ranked": cfg.output_dir / "gsea_ranked_gene_list.tsv",
        "full_csv": cfg.output_dir / "gsea_full_results.csv",
        "full_tsv": cfg.output_dir / "gsea_full_results.tsv",
        "sig_025": cfg.output_dir / "gsea_sig_fdr_0.25.csv",
        "sig_005": cfg.output_dir / "gsea_sig_fdr_0.05.csv",
        "top_up": cfg.output_dir / "gsea_top_up.csv",
        "top_down": cfg.output_dir / "gsea_top_down.csv",
        "summary": cfg.output_dir / "gsea_summary.md",
        "config": cfg.output_dir / "config_prerank_gsea.ini",
        "plots": cfg.output_dir / "gsea_plots",
    }
    paths["plots"].mkdir(parents=True, exist_ok=True)
    ranked.to_csv(paths["ranked"], sep="\t", index=False)
    results.to_csv(paths["full_csv"], index=False)
    results.to_csv(paths["full_tsv"], sep="\t", index=False)
    return paths


def _save_filtered_tables(results: pd.DataFrame, paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sig_025 = results.loc[results["FDR q-val"] < 0.25].copy()
    sig_005 = results.loc[results["FDR q-val"] < 0.05].copy()
    sig_025.to_csv(paths["sig_025"], index=False)
    sig_005.to_csv(paths["sig_005"], index=False)

    top_source = sig_025 if not sig_025.empty else results.copy()
    top_up = top_source.loc[top_source["NES"] > 0].sort_values(["NES", "FDR q-val"], ascending=[False, True]).head(10).copy()
    top_down = top_source.loc[top_source["NES"] < 0].sort_values(["NES", "FDR q-val"], ascending=[True, True]).head(10).copy()
    top_columns = ["Term", "ES", "NES", "NOM p-val", "FDR q-val", "Lead_genes"]
    top_up.loc[:, top_columns].to_csv(paths["top_up"], index=False)
    top_down.loc[:, top_columns].to_csv(paths["top_down"], index=False)
    return sig_025, sig_005, top_up, top_down


def _plot_top_terms(cfg: PrerankGseaConfig, pre_res, top_up: pd.DataFrame, top_down: pd.DataFrame, paths: dict[str, Path]) -> None:
    import matplotlib.pyplot as plt
    from gseapy.plot import gseaplot

    for direction, frame in [("up", top_up.head(cfg.top_plot_n)), ("down", top_down.head(cfg.top_plot_n))]:
        for idx, row in frame.iterrows():
            term = str(row["Term"])
            out_png = paths["plots"] / f"{direction}_{idx+1:02d}_{_sanitize_filename(term)}.png"
            gseaplot(
                rank_metric=pre_res.ranking,
                term=term,
                ofname=str(out_png),
                **pre_res.results[term],
            )
            plt.close("all")


def _write_summary(
    cfg: PrerankGseaConfig,
    metric_note: str,
    ranked: pd.DataFrame,
    results: pd.DataFrame,
    sig_025: pd.DataFrame,
    sig_005: pd.DataFrame,
    top_up: pd.DataFrame,
    top_down: pd.DataFrame,
    paths: dict[str, Path],
) -> None:
    def _theme(term: str) -> str:
        text = term.casefold()
        if "glycol" in text or "cholesterol" in text or "metabolism" in text or "oxidative" in text:
            return "metabolism"
        if "immune" in text or "interferon" in text or "inflammatory" in text or "complement" in text:
            return "immune/inflammation"
        if "myogenesis" in text or "emt" in text or "junction" in text or "coagulation" in text:
            return "ECM / adhesion / stromal-like"
        if "e2f" in text or "g2m" in text or "mitotic" in text or "mTOR".casefold() in text:
            return "cell cycle / proliferation"
        if "unfolded" in text or "protein" in text:
            return "proteostasis / ER stress"
        return "mixed"

    lines = [
        "# GSEA Prerank Summary",
        "",
        f"- Input table: `{cfg.input_table}`",
        f"- Gene column: `{cfg.gene_column}`",
        f"- Ranking metric: {metric_note}",
        f"- Gene-set source: `{cfg.gene_sets}` via `gseapy.prerank()`.",
        "- Hallmark source note: the script uses the `MSigDB_Hallmark_2020` library name through gseapy/Enrichr when a local GMT is not provided.",
        f"- Genes retained in final ranked list: {len(ranked)}",
        f"- Permutations: {cfg.permutation_num}",
        f"- min_size={cfg.min_size}, max_size={cfg.max_size}, seed={cfg.seed}",
        "",
        "## Significance summary",
        f"- Total pathways tested: {len(results)}",
        f"- FDR q-value < 0.25: {len(sig_025)}",
        f"- FDR q-value < 0.05: {len(sig_005)}",
        "",
        f"## Strongest positively enriched pathways (up in {cfg.group_a_label})",
    ]
    if top_up.empty:
        lines.append("- No positively enriched pathways passed the current filter.")
    else:
        for _, row in top_up.iterrows():
            lines.append(
                f"- {row['Term']} | NES={row['NES']:.3f} | FDR={row['FDR q-val']:.3g} | theme={_theme(str(row['Term']))}"
            )
    lines.extend(["", f"## Strongest negatively enriched pathways (up in {cfg.group_b_label})"])
    if top_down.empty:
        lines.append("- No negatively enriched pathways passed the current filter.")
    else:
        for _, row in top_down.iterrows():
            lines.append(
                f"- {row['Term']} | NES={row['NES']:.3f} | FDR={row['FDR q-val']:.3g} | theme={_theme(str(row['Term']))}"
            )
    if sig_025.empty:
        lines.extend(
            [
                "",
                "## Interpretation note",
                "- No pathways passed FDR < 0.25. The top up/down tables therefore represent nominal trends only and are not formally significant.",
            ]
        )
    paths["summary"].write_text("\n".join(lines), encoding="utf-8")


def _save_config_snapshot(cfg: PrerankGseaConfig, path: Path) -> None:
    parser = configparser.ConfigParser()
    parser["task"] = {"name": "prerank_gsea"}
    parser["input"] = {"input_table": str(cfg.input_table)}
    parser["output"] = {"output_dir": str(cfg.output_dir)}
    parser["settings"] = {
        "gene_column": cfg.gene_column,
        "rank_column": cfg.rank_column or "",
        "rank_metric_name": cfg.rank_metric_name or "",
        "group_a_label": cfg.group_a_label,
        "group_b_label": cfg.group_b_label,
        "gene_sets": cfg.gene_sets,
        "organism": cfg.organism,
        "permutation_num": str(cfg.permutation_num),
        "min_size": str(cfg.min_size),
        "max_size": str(cfg.max_size),
        "seed": str(cfg.seed),
        "top_plot_n": str(cfg.top_plot_n),
    }
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        parser.write(handle)


def run_prerank_gsea(config_path: Path) -> Path:
    cfg = _load_config(config_path)
    print(f"[gsea] loading input table: {cfg.input_table}")
    ranked, metric_note = _prepare_ranked_table(cfg)
    print(f"[gsea] ranking metric selected: {metric_note}")
    print(f"[gsea] ranked genes after cleanup/deduplication: {len(ranked)}")
    pre_res = _run_prerank(cfg, ranked)
    results = _normalize_result_table(pre_res.res2d)
    if "Term" not in results.columns:
        raise ValueError("GSEA results are missing the `Term` column.")
    print(f"[gsea] pathways returned: {len(results)}")
    paths = _save_tables(cfg, ranked, results)
    sig_025, sig_005, top_up, top_down = _save_filtered_tables(results, paths)
    print(f"[gsea] FDR<0.25 pathways: {len(sig_025)}")
    print(f"[gsea] FDR<0.05 pathways: {len(sig_005)}")
    _plot_top_terms(cfg, pre_res, top_up, top_down, paths)
    _write_summary(cfg, metric_note, ranked, results, sig_025, sig_005, top_up, top_down, paths)
    _save_config_snapshot(cfg, paths["config"])
    print(f"[gsea] outputs written to: {cfg.output_dir}")
    return paths["full_csv"]
