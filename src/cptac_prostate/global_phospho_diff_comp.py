from __future__ import annotations

import configparser
from dataclasses import dataclass, field
from itertools import combinations
from math import ceil
from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
from venn import venn


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


def _get_optional_int(
    config: configparser.ConfigParser,
    section: str,
    option: str,
    default: int,
) -> int:
    if not config.has_section(section) or not config.has_option(section, option):
        return default
    return int(_strip_quotes(config.get(section, option)))


def _parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_group_name(group_name: str) -> str:
    group_name = group_name.strip()
    if group_name.casefold() == "tumor":
        return "tumor"

    match = re.fullmatch(r"g{1,2}(\d+)", group_name, flags=re.IGNORECASE)
    if match is not None:
        return f"GG{match.group(1)}"
    return group_name


def _group_title(group_name: str) -> str:
    if group_name.casefold() == "tumor":
        return "Tumor vs Normal"
    return f"{group_name} vs Normal"


def _group_prefix(group_name: str) -> str:
    if group_name.casefold() == "tumor":
        return "tumor_vs_normal"
    return f"{group_name}_vs_normal"


def _candidate_group_tokens(group_name: str) -> list[str]:
    normalized = _normalize_group_name(group_name)
    if normalized.casefold() == "tumor":
        return ["tumor"]

    match = re.fullmatch(r"GG(\d+)", normalized, flags=re.IGNORECASE)
    if match is None:
        return [group_name, normalized]

    grade_value = match.group(1)
    return [normalized, f"G{grade_value}"]


def _read_gene_list(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    return {line.strip() for line in text.splitlines() if line.strip()}


def _read_genes_from_diff(diff_path: Path, significance_label: str) -> set[str]:
    diff = pd.read_csv(diff_path, sep="\t")
    if "Gene" not in diff.columns or "Significance" not in diff.columns:
        msg = f"Diff result file is missing Gene/Significance columns: {diff_path}"
        raise ValueError(msg)
    return set(
        diff.loc[diff["Significance"] == significance_label, "Gene"]
        .dropna()
        .astype(str)
        .tolist()
    )


def _load_gene_set(output_dir: Path, group_name: str, significance_label: str) -> set[str]:
    label_filename = f"{significance_label}_genes.txt"
    for token in _candidate_group_tokens(group_name):
        gene_path = output_dir / f"genes_{token}_vs_normal" / label_filename
        if gene_path.exists():
            return _read_gene_list(gene_path)

    for token in _candidate_group_tokens(group_name):
        diff_path = output_dir / f"{token}_vs_normal_diff.tsv"
        if diff_path.exists():
            return _read_genes_from_diff(diff_path, significance_label)

    msg = f"Unable to find results for group '{group_name}' under {output_dir}."
    raise FileNotFoundError(msg)


def _compute_exclusive_overlap_rows(group_sets: dict[str, set[str]]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    set_names = list(group_sets.keys())
    for r in range(1, len(set_names) + 1):
        for combo in combinations(set_names, r):
            intersection = set.intersection(*(group_sets[name] for name in combo))
            other_names = [name for name in set_names if name not in combo]
            exclusive = intersection.copy()
            for other in other_names:
                exclusive -= group_sets[other]
            results.append(
                {
                    "combination": "&".join(combo),
                    "n_sets": len(combo),
                    "count": len(exclusive),
                    "genes": sorted(exclusive),
                }
            )
    return results


@dataclass
class GlobalPhosphoDiffCompConfig:
    global_output_dir: Path
    phospho_st_output_dir: Path
    py_output_dir: Path
    output_dir: Path
    group_names: list[str] = field(default_factory=lambda: ["tumor", "GG2", "GG3", "GG4", "GG5"])
    significance_label: str = "S-U"
    global_label: str = "global"
    phospho_st_label: str = "phospho_ST"
    py_label: str = "pY"
    facet_ncols: int = 3


@dataclass
class GlobalPhosphoDiffCompState:
    cfg: GlobalPhosphoDiffCompConfig
    group_sets: dict[str, dict[str, set[str]]] = field(default_factory=dict)
    summary_table: pd.DataFrame | None = None
    membership_table: pd.DataFrame | None = None


def _build_config(config: configparser.ConfigParser) -> GlobalPhosphoDiffCompConfig:
    group_names = _parse_csv_list(
        _get_optional_value(config, "settings", "group_names", "tumor,GG2,GG3,GG4,GG5")
    )
    return GlobalPhosphoDiffCompConfig(
        global_output_dir=_get_required_path(config, "input", "global_output_dir"),
        phospho_st_output_dir=_get_required_path(config, "input", "phospho_st_output_dir"),
        py_output_dir=_get_required_path(config, "input", "py_output_dir"),
        output_dir=_get_required_path(config, "output", "output_dir"),
        group_names=group_names,
        significance_label=_get_optional_value(config, "settings", "significance_label", "S-U"),
        global_label=_get_optional_value(config, "settings", "global_label", "global"),
        phospho_st_label=_get_optional_value(config, "settings", "phospho_st_label", "phospho_ST"),
        py_label=_get_optional_value(config, "settings", "py_label", "pY"),
        facet_ncols=_get_optional_int(config, "settings", "facet_ncols", 3),
    )


def _load_group_sets(state: GlobalPhosphoDiffCompState) -> dict[str, dict[str, set[str]]]:
    if state.group_sets:
        return state.group_sets

    result: dict[str, dict[str, set[str]]] = {}
    for group_name in state.cfg.group_names:
        normalized_group = _normalize_group_name(group_name)
        result[normalized_group] = {
            state.cfg.global_label: _load_gene_set(
                state.cfg.global_output_dir,
                normalized_group,
                state.cfg.significance_label,
            ),
            state.cfg.phospho_st_label: _load_gene_set(
                state.cfg.phospho_st_output_dir,
                normalized_group,
                state.cfg.significance_label,
            ),
            state.cfg.py_label: _load_gene_set(
                state.cfg.py_output_dir,
                normalized_group,
                state.cfg.significance_label,
            ),
        }
    state.group_sets = result
    return result


def _draw_venn_panel(
    ax: plt.Axes,
    source_sets: dict[str, set[str]],
    title: str,
    *,
    fontsize: int,
    legend_loc: str | None,
) -> None:
    if any(source_sets.values()):
        venn(
            source_sets,
            ax=ax,
            fontsize=fontsize,
            legend_loc=legend_loc,
        )
    else:
        ax.text(0.5, 0.5, "No genes available", ha="center", va="center")
        ax.axis("off")
    ax.set_title(title)


def _save_faceted_venn_figure(
    output_dir: Path,
    group_sets: dict[str, dict[str, set[str]]],
    cfg: GlobalPhosphoDiffCompConfig,
) -> None:
    n_panels = len(group_sets)
    ncols = max(1, cfg.facet_ncols)
    nrows = max(1, ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.6 * ncols, 4.8 * nrows))
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, (group_name, source_sets) in zip(axes_flat, group_sets.items(), strict=False):
        title = (
            f"{_group_title(group_name)}\n"
            f"{cfg.global_label}={len(source_sets[cfg.global_label])}, "
            f"{cfg.phospho_st_label}={len(source_sets[cfg.phospho_st_label])}, "
            f"{cfg.py_label}={len(source_sets[cfg.py_label])}"
        )
        _draw_venn_panel(
            ax,
            source_sets,
            title,
            fontsize=10,
            legend_loc=None,
        )

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    fig.suptitle(
        (
            f"{cfg.significance_label} Gene Overlap Across Global, "
            f"Phospho-ST, and pY Comparisons"
        ),
        fontsize=16,
        y=0.98,
    )
    fig.text(
        0.5,
        0.945,
        f"Set labels: {cfg.global_label}, {cfg.phospho_st_label}, {cfg.py_label}",
        ha="center",
        va="center",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.92))
    fig.savefig(
        output_dir / f"global_phospho_diff_comp_{cfg.significance_label}_venn_facets.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def venn_diagram(state: GlobalPhosphoDiffCompState) -> GlobalPhosphoDiffCompState:
    output_dir = state.cfg.output_dir
    group_sets = _load_group_sets(state)

    summary_rows: list[dict[str, object]] = []
    membership_rows: list[dict[str, object]] = []
    label_to_column = {
        state.cfg.global_label: "in_global",
        state.cfg.phospho_st_label: "in_phospho_st",
        state.cfg.py_label: "in_pY",
    }

    for group_name, source_sets in group_sets.items():
        file_prefix = f"{_group_prefix(group_name)}_{state.cfg.significance_label}"
        fig, ax = plt.subplots(figsize=(9, 8))
        _draw_venn_panel(
            ax,
            source_sets,
            f"{state.cfg.significance_label} Gene Overlap: {_group_title(group_name)}",
            fontsize=12,
            legend_loc="upper right",
        )
        fig.tight_layout()
        fig.savefig(output_dir / f"{file_prefix}_venn.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        overlap_rows = _compute_exclusive_overlap_rows(source_sets)
        overlap_table = pd.DataFrame(overlap_rows).sort_values(["n_sets", "combination"]).reset_index(drop=True)
        overlap_table["genes"] = overlap_table["genes"].apply(lambda values: ";".join(values))
        overlap_table.to_csv(output_dir / f"{file_prefix}_overlap.tsv", sep="\t", index=False)

        union_genes = sorted(set.union(*source_sets.values()))
        for gene in union_genes:
            membership = [label for label, genes in source_sets.items() if gene in genes]
            row = {
                "group": group_name,
                "comparison": _group_prefix(group_name),
                "gene": gene,
                "membership": "&".join(membership),
            }
            for label, column_name in label_to_column.items():
                row[column_name] = gene in source_sets[label]
            membership_rows.append(row)

        summary_rows.append(
            {
                "group": group_name,
                "comparison": _group_prefix(group_name),
                "significance_label": state.cfg.significance_label,
                "n_global": len(source_sets[state.cfg.global_label]),
                "n_phospho_st": len(source_sets[state.cfg.phospho_st_label]),
                "n_pY": len(source_sets[state.cfg.py_label]),
                "n_global_phospho_st": len(
                    source_sets[state.cfg.global_label] & source_sets[state.cfg.phospho_st_label]
                ),
                "n_global_pY": len(source_sets[state.cfg.global_label] & source_sets[state.cfg.py_label]),
                "n_phospho_st_pY": len(
                    source_sets[state.cfg.phospho_st_label] & source_sets[state.cfg.py_label]
                ),
                "n_all_three": len(
                    source_sets[state.cfg.global_label]
                    & source_sets[state.cfg.phospho_st_label]
                    & source_sets[state.cfg.py_label]
                ),
            }
        )

    _save_faceted_venn_figure(output_dir, group_sets, state.cfg)

    summary_table = pd.DataFrame(summary_rows)
    membership_table = pd.DataFrame(membership_rows)
    summary_table.to_csv(output_dir / "global_phospho_diff_comp_summary.tsv", sep="\t", index=False)
    membership_table.to_csv(output_dir / "global_phospho_diff_comp_membership.tsv", sep="\t", index=False)

    state.summary_table = summary_table
    state.membership_table = membership_table
    return state


STEP_FUNCTIONS = {
    "venn_diagram": venn_diagram,
}


def run_global_phospho_diff_comp(config_path: Path) -> Path:
    config = _read_config(config_path)
    cfg = _build_config(config)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    state = GlobalPhosphoDiffCompState(cfg=cfg)
    for step_name in _get_step_names(config):
        print(step_name)
        if step_name not in STEP_FUNCTIONS:
            msg = f"Unsupported global_phospho_diff_comp step: {step_name}"
            raise ValueError(msg)
        state = STEP_FUNCTIONS[step_name](state)

    output_path = cfg.output_dir / "global_phospho_diff_comp_summary.tsv"
    print(f"global_phospho_diff_comp output written to {output_path}")
    return output_path
