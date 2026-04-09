from __future__ import annotations

import configparser
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cptac_prostate.global_diff import (
    _get_optional_float,
    _get_optional_value,
    _get_required_path,
    _infer_sep,
    _pick_sample_column,
    _read_config,
    _strip_quotes,
    run_global_diff,
)


def _get_step_names(config: configparser.ConfigParser) -> list[str]:
    if not config.has_section("steps"):
        return []

    step_names: list[str] = []
    for option in sorted(config.options("steps")):
        step_names.append(_strip_quotes(config.get("steps", option)))
    return step_names


def _parse_contrast_value(value: str) -> tuple[str, str]:
    cleaned = value.replace("vs", ",").replace("VS", ",")
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    if len(parts) != 2:
        msg = f"Invalid pairwise contrast definition: {value}"
        raise ValueError(msg)
    return parts[0], parts[1]


def _read_contrasts(config: configparser.ConfigParser) -> list[tuple[str, str]]:
    if config.has_section("contrasts"):
        contrasts: list[tuple[str, str]] = []
        for option in sorted(config.options("contrasts")):
            contrasts.append(_parse_contrast_value(_strip_quotes(config.get("contrasts", option))))
        return contrasts

    groups_text = _get_optional_value(config, "settings", "pairwise_groups", "")
    groups = [group.strip() for group in groups_text.split(",") if group.strip()]
    if len(groups) < 2:
        msg = "Provide either a [contrasts] section or [settings] pairwise_groups with at least two groups."
        raise ValueError(msg)
    return list(combinations(groups, 2))


@dataclass
class PairwiseConfig:
    input_dir: Path
    global_path: Path
    meta_dir: Path
    meta_path: Path
    output_dir: Path
    analysis_method: str
    fdr_cutoff: float
    log2fc_cutoff: float
    summary_top_n: int
    contrasts: list[tuple[str, str]]

    @property
    def data_path(self) -> Path:
        return self.input_dir / self.global_path

    @property
    def metadata_path(self) -> Path:
        return self.meta_dir / self.meta_path


def _build_pairwise_config(config: configparser.ConfigParser) -> PairwiseConfig:
    return PairwiseConfig(
        input_dir=_get_required_path(config, "input", "input_dir"),
        global_path=_get_required_path(config, "input", "global_path"),
        meta_dir=_get_required_path(config, "input", "meta_dir"),
        meta_path=_get_required_path(config, "input", "meta_path"),
        output_dir=_get_required_path(config, "output", "output_dir"),
        analysis_method=_get_optional_value(config, "analysis", "method", "ModeratedT"),
        fdr_cutoff=_get_optional_float(config, "settings", "FDR", 0.05),
        log2fc_cutoff=_get_optional_float(config, "settings", "log2FC_threshold", 0.58),
        summary_top_n=int(_get_optional_float(config, "settings", "summary_top_n", 20)),
        contrasts=_read_contrasts(config),
    )


def _write_single_contrast_config(
    output_dir: Path,
    pair: tuple[str, str],
    cfg: PairwiseConfig,
) -> Path:
    group1, group2 = pair
    config_path = output_dir / f"config_{group1}_vs_{group2}.ini"
    parser = configparser.ConfigParser()
    parser["input"] = {
        "input_dir": f'"{cfg.input_dir}"',
        "global_path": f'"{cfg.global_path}"',
        "meta_dir": f'"{cfg.meta_dir}"',
        "meta_path": f'"{cfg.meta_path}"',
    }
    parser["output"] = {"output_dir": f'"{cfg.output_dir}"'}
    parser["task"] = {"name": '"global_diff"'}
    parser["settings"] = {
        "group1": f'"{group1}"',
        "group2": f'"{group2}"',
        "FDR": str(cfg.fdr_cutoff),
        "log2FC_threshold": str(cfg.log2fc_cutoff),
    }
    parser["analysis"] = {"method": f'"{cfg.analysis_method}"'}
    parser["steps"] = {
        "step1": '"get_sample_info"',
        "step2": '"compare_two_groups"',
        "step3": '"plot_volcano"',
        "step4": '"save_results"',
    }
    with config_path.open("w", encoding="utf-8") as handle:
        parser.write(handle)
    return config_path


def _load_group_sample_sizes(cfg: PairwiseConfig) -> dict[str, int]:
    meta = pd.read_csv(cfg.metadata_path, sep=_infer_sep(cfg.metadata_path))
    data = pd.read_csv(cfg.data_path, sep=_infer_sep(cfg.data_path))
    feature_column = "geneSymbol"
    if feature_column in data.columns:
        data = data.set_index(feature_column)
    sample_id_column = _pick_sample_column(meta, data.columns.tolist(), None)

    result: dict[str, int] = {}
    for group in sorted({group for pair in cfg.contrasts for group in pair}):
        if not group.upper().startswith("G"):
            continue
        grade = group.upper().replace("GG", "").replace("G", "")
        mask = (
            meta["Tissuetype"].astype(str).str.casefold() == "tumor"
        ) & (
            meta["FirstCategory"].astype(str) == "Sufficient Purity"
        ) & (
            pd.to_numeric(meta["BCR_Gleason_Grade"], errors="coerce") == float(grade)
        )
        result[group] = int(meta.loc[mask, sample_id_column].astype(str).isin(data.columns).sum())
    return result


def _summarize_pairwise_results(cfg: PairwiseConfig) -> None:
    groups = sorted({group for pair in cfg.contrasts for group in pair})
    hit_matrix = pd.DataFrame(0, index=groups, columns=groups, dtype=int)
    summary_rows: list[dict[str, object]] = []
    top_rows: list[pd.DataFrame] = []

    for group1, group2 in cfg.contrasts:
        diff_path = cfg.output_dir / f"{group1}_vs_{group2}_diff.tsv"
        diff = pd.read_csv(diff_path, sep="\t")
        sig = diff["Significance"].fillna("NS")
        su_count = int((sig == "S-U").sum())
        sd_count = int((sig == "S-D").sum())
        hit_matrix.loc[group1, group2] = su_count
        hit_matrix.loc[group2, group1] = sd_count

        summary_rows.append(
            {
                "contrast": f"{group1}_vs_{group2}",
                "group1": group1,
                "group2": group2,
                "n_S_U": su_count,
                "n_S_D": sd_count,
                "n_U": int((sig == "U").sum()),
                "n_D": int((sig == "D").sum()),
                "top_abs_log2fc": float(diff["AbsLog2FC(median)"].max(skipna=True)),
                "min_FDR": float(diff["FDR"].min(skipna=True)) if diff["FDR"].notna().any() else pd.NA,
            }
        )

        top_frame = diff.sort_values(["AbsLog2FC(median)", "FDR"], ascending=[False, True]).head(cfg.summary_top_n).copy()
        top_frame.insert(0, "contrast", f"{group1}_vs_{group2}")
        top_rows.append(top_frame)

    summary_df = pd.DataFrame(summary_rows).sort_values(["top_abs_log2fc", "min_FDR"], ascending=[False, True])
    summary_df.to_csv(cfg.output_dir / "pairwise_summary.tsv", sep="\t", index=False)

    top_effects = pd.concat(top_rows, ignore_index=True) if top_rows else pd.DataFrame()
    top_effects.to_csv(cfg.output_dir / "pairwise_top_effects.tsv", sep="\t", index=False)

    hit_matrix.to_csv(cfg.output_dir / "pairwise_sig_hit_matrix.tsv", sep="\t", index_label="group")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(
        hit_matrix,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        linewidths=0.5,
        square=True,
        cbar_kws={"label": "Directional significant protein count"},
        ax=ax,
    )
    ax.set_title("Pairwise Significant Protein Counts (directional)")
    ax.set_xlabel("Reference group")
    ax.set_ylabel("Higher-abundance group")
    fig.tight_layout()
    fig.savefig(cfg.output_dir / "pairwise_sig_hit_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    overview_plot = summary_df.copy()
    if not overview_plot.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_y = overview_plot["min_FDR"].astype(float)
        plot_x = overview_plot["top_abs_log2fc"].astype(float)
        ax.scatter(plot_x, plot_y, s=60, color="#b35806", alpha=0.9)
        for _, row in overview_plot.iterrows():
            ax.text(
                float(row["top_abs_log2fc"]) + 0.03,
                float(row["min_FDR"]),
                str(row["contrast"]),
                fontsize=8,
                va="center",
            )
        ax.axhline(cfg.fdr_cutoff, color="#666666", linestyle="--", linewidth=1)
        ax.set_xlabel("Top absolute log2 fold change per contrast")
        ax.set_ylabel("Minimum FDR per contrast")
        ax.set_title("Pairwise Contrast Overview: Effect Size vs FDR")
        fig.tight_layout()
        fig.savefig(cfg.output_dir / "pairwise_effect_overview.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    sample_sizes = _load_group_sample_sizes(cfg)
    summary_lines = [
        "Pairwise global_diff exploratory summary",
        f"analysis_method={cfg.analysis_method}",
        f"fdr_cutoff={cfg.fdr_cutoff}",
        f"log2FC_threshold={cfg.log2fc_cutoff}",
        "group_sample_sizes=" + ", ".join(f"{group}:{sample_sizes.get(group, 'NA')}" for group in groups),
        "outputs:",
        "- pairwise_summary.tsv: one row per explicit contrast with hit counts and top effect metrics.",
        "- pairwise_top_effects.tsv: top effect-size rows for each contrast.",
        "- pairwise_sig_hit_matrix.tsv/png: directional count summary across contrasts.",
        "- pairwise_effect_overview.png: contrast-level effect-size vs FDR overview.",
    ]
    (cfg.output_dir / "pairwise_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def run_global_diff_pairwise(config_path: Path) -> Path:
    config = _read_config(config_path)
    cfg = _build_pairwise_config(config)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    for pair in cfg.contrasts:
        single_config_path = _write_single_contrast_config(cfg.output_dir, pair, cfg)
        run_global_diff(single_config_path)

    _summarize_pairwise_results(cfg)
    output_path = cfg.output_dir / "pairwise_sig_hit_matrix.png"
    print(f"global_diff_pairwise output written to {output_path}")
    return output_path
