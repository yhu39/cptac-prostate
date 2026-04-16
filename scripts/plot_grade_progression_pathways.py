from __future__ import annotations

import argparse
import configparser
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
import seaborn as sns


PATTERN_GRADE_ORDER = ["GG2", "GG3", "GG4", "GG5"]
HEATMAP_GROUP_ORDER = ["normal", "GG1", "GG2", "GG3", "GG4", "GG5"]
HEATMAP_GROUP_LABELS = {
    "normal": "Normal",
    "GG1": "GG1",
    "GG2": "GG2",
    "GG3": "GG3",
    "GG4": "GG4",
    "GG5": "GG5",
}
MODULE_ORDER = [
    "Tumor epithelial / cell-state",
    "Nucleolar / RNA processing",
    "Secretory / glycan trafficking",
    "Mitochondrial / metabolic",
    "ECM / stromal remodeling",
]
MODULE_COLORS = {
    "Tumor epithelial / cell-state": "#355070",
    "Nucleolar / RNA processing": "#6d597a",
    "Secretory / glycan trafficking": "#2a9d8f",
    "Mitochondrial / metabolic": "#e09f3e",
    "ECM / stromal remodeling": "#bc4749",
}
MODULE_GENES = {
    "Tumor epithelial / cell-state": ["CGREF1", "EPCAM", "TSPAN13", "VSTM2L", "ZBTB7B"],
    "Nucleolar / RNA processing": ["DKC1", "GAR1", "NHP2", "NOLC1"],
    "Secretory / glycan trafficking": [
        "ENTPD5",
        "ERGIC1",
        "GALNT7",
        "GLYATL1",
        "GMDS",
        "GOLM1",
        "SLC37A1",
        "SLC4A4",
        "UAP1",
    ],
    "Mitochondrial / metabolic": ["AMACR", "COA4", "COX17", "FABP5", "FASN", "FBP2", "FMC1", "PYCR1"],
    "ECM / stromal remodeling": [
        "COMP",
        "LAMC3",
        "MARCKSL1",
        "PLA2G2A",
        "POSTN",
        "SFRP4",
        "THBS4",
        "TMSB15B",
        "UGDH",
    ],
}
PATTERN_COLORS = {
    "Trunk": "#1f78b4",
    "Late gain": "#d95f02",
    "Early loss": "#33a02c",
    "Variable shared": "#7f7f7f",
}
PLOT_PREFIX = "grade_progression_pathway_map"
CONFIG_FILENAME = "config_grade_progression_pathway.ini"
SUMMARY_FILENAME = "grade_progression_pathway_summary_.md"
EDITABLE_PDF_SUFFIX = "_editable_text.pdf"


@dataclass(frozen=True)
class ProgressionInputs:
    run_dir: Path
    output_dir: Path
    plot_prefix: str
    heatmap_groups: tuple[str, ...]
    summary_filename: str
    config_filename: str


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def _parse_csv_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _read_config(config_path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.read(config_path, encoding="utf-8")
    return parser


def _inputs_from_config(config_path: Path) -> ProgressionInputs:
    config = _read_config(config_path)
    run_dir = Path(_strip_quotes(config.get("input", "run_dir"))).resolve()
    output_dir = Path(_strip_quotes(config.get("output", "output_dir"))).resolve()
    plot_prefix = _strip_quotes(config.get("settings", "plot_prefix", fallback=PLOT_PREFIX))
    heatmap_groups = _parse_csv_list(
        _strip_quotes(config.get("settings", "heatmap_groups", fallback="normal,GG1,GG2,GG3,GG4,GG5"))
    )
    summary_filename = _strip_quotes(
        config.get("settings", "summary_filename", fallback=SUMMARY_FILENAME)
    )
    config_filename = _strip_quotes(
        config.get("settings", "config_filename", fallback=CONFIG_FILENAME)
    )
    return ProgressionInputs(
        run_dir=run_dir,
        output_dir=output_dir,
        plot_prefix=plot_prefix,
        heatmap_groups=heatmap_groups,
        summary_filename=summary_filename,
        config_filename=config_filename,
    )


def _merge_cli_overrides(args: argparse.Namespace, inputs: ProgressionInputs) -> ProgressionInputs:
    run_dir = args.run_dir.resolve() if args.run_dir is not None else inputs.run_dir
    output_dir = args.output_dir.resolve() if args.output_dir is not None else inputs.output_dir
    plot_prefix = args.plot_prefix if args.plot_prefix != PLOT_PREFIX else inputs.plot_prefix
    heatmap_groups = (
        _parse_csv_list(args.heatmap_groups)
        if args.heatmap_groups != "normal,GG1,GG2,GG3,GG4,GG5"
        else inputs.heatmap_groups
    )
    summary_filename = (
        args.summary_filename if args.summary_filename != SUMMARY_FILENAME else inputs.summary_filename
    )
    config_filename = (
        args.config_filename if args.config_filename != CONFIG_FILENAME else inputs.config_filename
    )
    return ProgressionInputs(
        run_dir=run_dir,
        output_dir=output_dir,
        plot_prefix=plot_prefix,
        heatmap_groups=heatmap_groups,
        summary_filename=summary_filename,
        config_filename=config_filename,
    )


def parse_args() -> ProgressionInputs:
    parser = argparse.ArgumentParser(
        description=(
            "Build a progression-focused heatmap from GG2-GG5 tumor-overlap S-U proteins."
        )
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional config ini path.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory containing global_diff_summary_heatmap.tsv and genes_*_vs_normal folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the figure and summary tables. Defaults to --run-dir.",
    )
    parser.add_argument(
        "--plot-prefix",
        default=PLOT_PREFIX,
        help=f"Output filename prefix (default: {PLOT_PREFIX}).",
    )
    parser.add_argument(
        "--heatmap-groups",
        default="normal,GG1,GG2,GG3,GG4,GG5",
        help="Comma-separated heatmap group order.",
    )
    parser.add_argument(
        "--summary-filename",
        default=SUMMARY_FILENAME,
        help=f"Summary markdown filename (default: {SUMMARY_FILENAME}).",
    )
    parser.add_argument(
        "--config-filename",
        default=CONFIG_FILENAME,
        help=f"Written config filename (default: {CONFIG_FILENAME}).",
    )
    args = parser.parse_args()

    if args.config is not None:
        return _merge_cli_overrides(args, _inputs_from_config(args.config.resolve()))

    if args.run_dir is None:
        msg = "--run-dir is required when --config is not provided."
        raise ValueError(msg)

    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or args.run_dir).resolve()
    heatmap_groups = _parse_csv_list(args.heatmap_groups)
    return ProgressionInputs(
        run_dir=run_dir,
        output_dir=output_dir,
        plot_prefix=args.plot_prefix,
        heatmap_groups=heatmap_groups,
        summary_filename=args.summary_filename,
        config_filename=args.config_filename,
    )


def _validate_groups(groups: tuple[str, ...]) -> None:
    missing = [group for group in groups if group not in HEATMAP_GROUP_LABELS]
    if missing:
        msg = f"Unsupported heatmap groups: {', '.join(missing)}"
        raise ValueError(msg)


def _read_gene_list(path: Path) -> set[str]:
    if not path.exists():
        msg = f"Gene list was not found: {path}"
        raise FileNotFoundError(msg)
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _read_grade_sets(run_dir: Path) -> dict[str, set[str]]:
    tumor_set = _read_gene_list(run_dir / "genes_tumor_vs_normal" / "S-U_genes.txt")
    grade_sets: dict[str, set[str]] = {}
    for grade in PATTERN_GRADE_ORDER:
        grade_set = _read_gene_list(run_dir / f"genes_{grade}_vs_normal" / "S-U_genes.txt")
        grade_sets[grade] = grade_set & tumor_set
    return grade_sets


def _pattern_category(pattern: str) -> str:
    if pattern == "1111":
        return "Trunk"
    if pattern in {"0111", "0011", "0001", "0101"}:
        return "Late gain"
    if pattern in {"1110", "1100", "1000"}:
        return "Early loss"
    return "Variable shared"


def _assign_module(gene: str) -> str:
    for module_name, genes in MODULE_GENES.items():
        if gene in genes:
            return module_name
    msg = f"Gene '{gene}' does not have a module assignment."
    raise ValueError(msg)


def _sample_count(run_dir: Path, grade: str) -> int:
    path = run_dir / f"{grade}_vs_normal_sample_info.tsv"
    frame = pd.read_csv(path, sep="\t")
    mask = frame["group"].astype(str).str.upper() == grade.upper()
    return int(frame.loc[mask, "sample"].astype(str).nunique())


def build_protein_frame(
    run_dir: Path,
    heatmap_groups: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, int]]:
    _validate_groups(heatmap_groups)
    grade_sets = _read_grade_sets(run_dir)
    union_genes = sorted(set.union(*grade_sets.values()))

    heatmap_path = run_dir / "global_diff_summary_heatmap.tsv"
    heatmap = pd.read_csv(heatmap_path, sep="\t")
    required_columns = {"gene", *heatmap_groups}
    missing_columns = sorted(required_columns - set(heatmap.columns))
    if missing_columns:
        msg = f"Missing columns in {heatmap_path}: {', '.join(missing_columns)}"
        raise ValueError(msg)

    trend = (
        heatmap.loc[heatmap["gene"].isin(union_genes), ["gene", *heatmap_groups]]
        .groupby("gene", as_index=False)
        .median()
    )
    trend["pattern"] = trend["gene"].map(
        lambda gene: "".join("1" if gene in grade_sets[grade] else "0" for grade in PATTERN_GRADE_ORDER)
    )
    trend["pattern_category"] = trend["pattern"].map(_pattern_category)
    trend["module"] = trend["gene"].map(_assign_module)
    trend["delta_G5_G2"] = trend["GG5"] - trend["GG2"]
    trend["monotonic_up"] = (
        (trend["GG2"] <= trend["GG3"])
        & (trend["GG3"] <= trend["GG4"])
        & (trend["GG4"] <= trend["GG5"])
    )
    trend["monotonic_down"] = (
        (trend["GG2"] >= trend["GG3"])
        & (trend["GG3"] >= trend["GG4"])
        & (trend["GG4"] >= trend["GG5"])
    )

    z_values = []
    for _, row in trend.iterrows():
        values = row[list(heatmap_groups)].to_numpy(dtype=float)
        std = float(values.std(ddof=0))
        if np.isclose(std, 0.0):
            z_values.append(np.zeros_like(values))
        else:
            z_values.append((values - float(values.mean())) / std)
    z_frame = pd.DataFrame(z_values, columns=[f"{group}_z" for group in heatmap_groups])
    trend = pd.concat([trend.reset_index(drop=True), z_frame], axis=1)

    pattern_rank = {"Trunk": 0, "Late gain": 1, "Variable shared": 2, "Early loss": 3}
    module_rank = {name: idx for idx, name in enumerate(MODULE_ORDER)}
    trend["module_rank"] = trend["module"].map(module_rank)
    trend["pattern_rank"] = trend["pattern_category"].map(pattern_rank)
    trend = trend.sort_values(
        ["module_rank", "pattern_rank", "delta_G5_G2", "gene"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)

    sample_counts = {grade: _sample_count(run_dir, grade) for grade in PATTERN_GRADE_ORDER}
    return trend, sample_counts


def build_module_frame(
    protein_frame: pd.DataFrame,
    heatmap_groups: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for module_name in MODULE_ORDER:
        module_sub = protein_frame.loc[protein_frame["module"] == module_name].copy()
        row: dict[str, object] = {
            "module": module_name,
            "n_genes": len(module_sub),
            "genes": ";".join(module_sub["gene"].tolist()),
        }
        for group in heatmap_groups:
            row[f"{group}_z"] = float(module_sub[f"{group}_z"].mean())
        for pattern_name in ["Trunk", "Late gain", "Early loss", "Variable shared"]:
            key = pattern_name.casefold().replace(" ", "_").replace("-", "_") + "_n"
            row[key] = int((module_sub["pattern_category"] == pattern_name).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def save_tables(
    output_dir: Path,
    plot_prefix: str,
    protein_frame: pd.DataFrame,
    module_frame: pd.DataFrame,
    heatmap_groups: tuple[str, ...],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    protein_cols = [
        "gene",
        "module",
        "pattern",
        "pattern_category",
        *heatmap_groups,
        *(f"{group}_z" for group in heatmap_groups),
        "delta_G5_G2",
        "monotonic_up",
        "monotonic_down",
    ]
    protein_path = output_dir / f"{plot_prefix}_proteins.tsv"
    module_path = output_dir / f"{plot_prefix}_modules.tsv"
    protein_frame.loc[:, protein_cols].to_csv(protein_path, sep="\t", index=False)
    module_frame.to_csv(module_path, sep="\t", index=False)
    return protein_path, module_path


def write_config(output_dir: Path, inputs: ProgressionInputs) -> Path:
    config = configparser.ConfigParser()
    config["input"] = {"run_dir": str(inputs.run_dir)}
    config["output"] = {"output_dir": str(inputs.output_dir)}
    config["settings"] = {
        "plot_prefix": inputs.plot_prefix,
        "heatmap_groups": ",".join(inputs.heatmap_groups),
        "summary_filename": inputs.summary_filename,
        "config_filename": inputs.config_filename,
    }
    config_path = output_dir / inputs.config_filename
    with config_path.open("w", encoding="utf-8") as handle:
        config.write(handle)
    return config_path


def write_summary(
    output_dir: Path,
    inputs: ProgressionInputs,
    protein_frame: pd.DataFrame,
    module_frame: pd.DataFrame,
    sample_counts: dict[str, int],
    figure_png: Path,
    editable_figure_pdf: Path,
    protein_path: Path,
    module_path: Path,
    config_path: Path,
) -> Path:
    pattern_counts = protein_frame["pattern_category"].value_counts()
    top_increasing = protein_frame.sort_values("delta_G5_G2", ascending=False).head(6)["gene"].tolist()
    top_decreasing = protein_frame.sort_values("delta_G5_G2", ascending=True).head(6)["gene"].tolist()
    monotonic_up = protein_frame.loc[protein_frame["monotonic_up"], "gene"].tolist()
    monotonic_down = protein_frame.loc[protein_frame["monotonic_down"], "gene"].tolist()
    module_delta = (
        module_frame.assign(delta_G5_G2=module_frame["GG5_z"] - module_frame["GG2_z"])
        .sort_values("delta_G5_G2", ascending=False)
        .reset_index(drop=True)
    )
    strongest_module = str(module_delta.loc[0, "module"])

    summary_lines = [
        "# Grade Progression Pathway Summary",
        "",
        "## Generated files",
        f"- Figure PNG: `{figure_png.name}`",
        f"- Editable-text PDF: `{editable_figure_pdf.name}`",
        f"- Protein table: `{protein_path.name}`",
        f"- Module table: `{module_path.name}`",
        f"- Config: `{config_path.name}`",
        "",
        "## Figure reading guide",
        "- 左侧彩色块表示功能模块分组。",
        f"- heatmap 列顺序为：{', '.join(HEATMAP_GROUP_LABELS[group] for group in inputs.heatmap_groups)}。",
        "- 每个蛋白在 heatmap 中的颜色是按该蛋白在所有列中的相对表达做 z-score 后得到，因此更适合看轨迹，不直接代表原始 fold change。",
        "- gene name 放在 heatmap 右侧，便于和最右边的 `Set-membership pattern` 条带对应。",
        "- `Set-membership pattern` 只基于 GG2-GG5 的 tumor-overlap `S-U` 集合定义，不受 Normal 和 GG1 表达列影响。",
        "",
        "## Biological interpretation",
        f"- 最明显上升的功能方向是 `{strongest_module}`，对应的代表蛋白包括 {', '.join(top_increasing)}。",
        f"- 最明显下降的蛋白主要包括 {', '.join(top_decreasing)}，更偏向分泌/糖基化或早期状态相关功能。",
        f"- 单调上升蛋白：{', '.join(monotonic_up) if monotonic_up else 'None'}。",
        f"- 单调下降蛋白：{', '.join(monotonic_down) if monotonic_down else 'None'}。",
        (
            f"- 样本数提示：GG2 n={sample_counts['GG2']}, GG3 n={sample_counts['GG3']}, "
            f"GG4 n={sample_counts['GG4']}, GG5 n={sample_counts['GG5']}；GG4 结果建议谨慎解读。"
        ),
        "",
        "## Set-membership pattern definitions",
        "- 位点定义顺序固定为 `GG2, GG3, GG4, GG5`；`1` 表示该蛋白属于对应 grade 的 tumor-overlap `S-U` 集合，`0` 表示不属于。",
        f"- `Trunk` (`1111`)：GG2-GG5 全部存在，代表稳定的肿瘤主干程序。当前共有 {int(pattern_counts.get('Trunk', 0))} 个蛋白。",
        (
            f"- `Late gain` (`0111`, `0011`, `0001`, `0101`)：在 GG2 不明显，随后在更高 grade 出现，"
            f"代表晚期获得的进展程序。当前共有 {int(pattern_counts.get('Late gain', 0))} 个蛋白。"
        ),
        (
            f"- `Early loss` (`1110`, `1100`, `1000`)：在早期 grade 出现，但随着 grade 升高逐渐丢失，"
            f"代表早期状态或较分化功能。当前共有 {int(pattern_counts.get('Early loss', 0))} 个蛋白。"
        ),
        (
            f"- `Variable shared`（其余模式，如 `1101`）：在多个 grade 间共享，但不符合简单的持续获得或持续丢失，"
            f"代表更波动的共享程序。当前共有 {int(pattern_counts.get('Variable shared', 0))} 个蛋白。"
        ),
        "",
        "## Working model",
        "- 这组结果支持“稳定主干程序 + 晚期 ECM/基质重塑与代谢增强”的两阶段进展模型。",
        "- Normal 和 GG1 列提供了更早期背景，因此可以更直观看到哪些蛋白从正常/低级别状态一路升高到 GG5。",
    ]
    summary_path = output_dir / inputs.summary_filename
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8-sig")
    return summary_path


def _draw_module_blocks(ax: plt.Axes, protein_frame: pd.DataFrame) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(len(protein_frame), 0)
    ax.axis("off")

    current_row = 0
    for module_name in MODULE_ORDER:
        row_count = int((protein_frame["module"] == module_name).sum())
        ax.add_patch(
            Rectangle(
                (0, current_row),
                1,
                row_count,
                facecolor=MODULE_COLORS[module_name],
                edgecolor="white",
                linewidth=1.2,
            )
        )
        ax.text(
            0.5,
            current_row + row_count / 2,
            fill(module_name.replace(" / ", "\n"), width=16),
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            fontweight="bold",
        )
        current_row += row_count


def _draw_pattern_strip(ax: plt.Axes, protein_frame: pd.DataFrame) -> None:
    colors = [to_rgba(PATTERN_COLORS[label]) for label in protein_frame["pattern_category"]]
    ax.imshow(np.array(colors).reshape(len(colors), 1, 4), aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Set-membership\npattern", fontsize=10, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_heatmap(
    ax: plt.Axes,
    protein_frame: pd.DataFrame,
    heatmap_groups: tuple[str, ...],
    color_limits: tuple[float, float],
) -> plt.AxesImage:
    values = protein_frame[[f"{group}_z" for group in heatmap_groups]].to_numpy(dtype=float)
    image = ax.imshow(
        values,
        cmap="RdBu_r",
        vmin=color_limits[0],
        vmax=color_limits[1],
        aspect="auto",
    )
    ax.set_xticks(range(len(heatmap_groups)))
    ax.set_xticklabels([HEATMAP_GROUP_LABELS[group] for group in heatmap_groups], fontsize=11)
    ax.set_yticks(range(len(protein_frame)))
    ax.set_yticklabels(protein_frame["gene"].tolist(), fontsize=9)
    ax.yaxis.tick_right()
    ax.tick_params(axis="y", labelright=True, labelleft=False, length=0, pad=5)
    ax.set_title(
        "Figure B. Protein trajectories across Normal, GG1, and GG2-GG5",
        loc="left",
        fontsize=14,
        pad=10,
        fontweight="bold",
    )
    ax.set_xlabel("Group", fontsize=11)
    ax.set_xticks(np.arange(-0.5, len(heatmap_groups), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(protein_frame), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    if "GG1" in heatmap_groups and "GG2" in heatmap_groups:
        gg1_idx = heatmap_groups.index("GG1")
        ax.axvline(gg1_idx + 0.5, color="black", linewidth=1.0, linestyle="--")

    boundary = 0
    for module_name in MODULE_ORDER[:-1]:
        boundary += int((protein_frame["module"] == module_name).sum())
        ax.axhline(boundary - 0.5, color="black", linewidth=1.2)

    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def plot_figure(
    output_dir: Path,
    plot_prefix: str,
    protein_frame: pd.DataFrame,
    sample_counts: dict[str, int],
    heatmap_groups: tuple[str, ...],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white")

    z_columns = [f"{group}_z" for group in heatmap_groups]
    z_min = float(protein_frame[z_columns].min().min())
    z_max = float(protein_frame[z_columns].max().max())
    limit = max(abs(z_min), abs(z_max), 1.0)
    color_limits = (-limit, limit)

    fig = plt.figure(figsize=(16.0, 11.0))
    grid = fig.add_gridspec(
        nrows=1,
        ncols=6,
        width_ratios=[2.25, 6.8, 1.25, 0.42, 0.30, 1.75],
        wspace=0.05,
    )

    ax_module = fig.add_subplot(grid[0, 0])
    _draw_module_blocks(ax_module, protein_frame)

    ax_heatmap = fig.add_subplot(grid[0, 1])
    image = _draw_heatmap(ax_heatmap, protein_frame, heatmap_groups, color_limits)

    ax_gap = fig.add_subplot(grid[0, 2])
    ax_gap.axis("off")

    ax_pattern = fig.add_subplot(grid[0, 3])
    _draw_pattern_strip(ax_pattern, protein_frame)

    ax_cbar = fig.add_subplot(grid[0, 4])
    colorbar = fig.colorbar(image, cax=ax_cbar)
    colorbar.set_label("Within-protein z-score", fontsize=10)
    colorbar.ax.tick_params(labelsize=9)

    ax_legend = fig.add_subplot(grid[0, 5])
    ax_legend.axis("off")
    pattern_handles = [
        Patch(facecolor=PATTERN_COLORS[name], edgecolor="none", label=name)
        for name in ["Trunk", "Late gain", "Early loss", "Variable shared"]
    ]
    ax_legend.legend(
        handles=pattern_handles,
        loc="upper left",
        frameon=False,
        title="Pattern legend",
        fontsize=10,
        title_fontsize=10,
    )

    sample_text = ", ".join(f"{grade} n={sample_counts[grade]}" for grade in PATTERN_GRADE_ORDER)
    fig.suptitle(
        "Tumor-overlap S-U proteins across prostate cancer grade progression",
        x=0.02,
        y=0.97,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.02,
        0.015,
        (
            "Heatmap columns include Normal and GG1 baselines. "
            "Set-membership pattern is defined only from GG2-GG5 tumor-overlap S-U membership. "
            f"Grade-group sample counts: {sample_text}."
        ),
        ha="left",
        va="bottom",
        fontsize=10,
    )

    png_path = output_dir / f"{plot_prefix}.png"
    editable_pdf_path = output_dir / f"{plot_prefix}{EDITABLE_PDF_SUFFIX}"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    original_pdf_fonttype = plt.rcParams.get("pdf.fonttype", 3)
    original_ps_fonttype = plt.rcParams.get("ps.fonttype", 3)
    try:
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42
        fig.savefig(editable_pdf_path, dpi=300, bbox_inches="tight")
    finally:
        plt.rcParams["pdf.fonttype"] = original_pdf_fonttype
        plt.rcParams["ps.fonttype"] = original_ps_fonttype
    plt.close(fig)
    return png_path, editable_pdf_path


def main() -> int:
    inputs = parse_args()
    protein_frame, sample_counts = build_protein_frame(inputs.run_dir, inputs.heatmap_groups)
    module_frame = build_module_frame(protein_frame, inputs.heatmap_groups)
    protein_path, module_path = save_tables(
        inputs.output_dir,
        inputs.plot_prefix,
        protein_frame,
        module_frame,
        inputs.heatmap_groups,
    )
    config_path = write_config(inputs.output_dir, inputs)
    png_path, editable_pdf_path = plot_figure(
        inputs.output_dir,
        inputs.plot_prefix,
        protein_frame,
        sample_counts,
        inputs.heatmap_groups,
    )
    summary_path = write_summary(
        inputs.output_dir,
        inputs,
        protein_frame,
        module_frame,
        sample_counts,
        png_path,
        editable_pdf_path,
        protein_path,
        module_path,
        config_path,
    )
    print(f"Saved figure: {png_path}")
    print(f"Saved editable-text PDF: {editable_pdf_path}")
    print(f"Saved protein table: {protein_path}")
    print(f"Saved module table: {module_path}")
    print(f"Saved config: {config_path}")
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
