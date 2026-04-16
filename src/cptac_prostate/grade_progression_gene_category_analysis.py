from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
import re
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
import seaborn as sns


PATTERN_GRADE_ORDER = ("GG2", "GG3", "GG4", "GG5")
DEFAULT_HEATMAP_GROUPS = ("normal", "GG1", "GG2", "GG3", "GG4", "GG5")
HEATMAP_GROUP_LABELS = {
    "normal": "Normal",
    "GG1": "GG1",
    "GG2": "GG2",
    "GG3": "GG3",
    "GG4": "GG4",
    "GG5": "GG5",
}
CATEGORY_ORDER = (
    "Tumor epithelial / cell-state",
    "Nucleolar / RNA processing",
    "Secretory / glycan trafficking",
    "Mitochondrial / metabolic",
    "ECM / stromal remodeling",
)
CATEGORY_COLORS = {
    "Tumor epithelial / cell-state": "#355070",
    "Nucleolar / RNA processing": "#6d597a",
    "Secretory / glycan trafficking": "#2a9d8f",
    "Mitochondrial / metabolic": "#e09f3e",
    "ECM / stromal remodeling": "#bc4749",
}
PATTERN_COLORS = {
    "Trunk": "#1f78b4",
    "Late gain": "#d95f02",
    "Early loss": "#33a02c",
    "Variable shared": "#7f7f7f",
}
DEFAULT_ENRICHMENT_LIBRARIES = (
    "MSigDB_Hallmark_2020",
    "GO_Biological_Process_2023",
    "KEGG_2021_Human",
    "Reactome_2022",
)
LIBRARY_LABELS = {
    "MSigDB_Hallmark_2020": "Hallmark",
    "GO_Biological_Process_2023": "GO_BP",
    "KEGG_2021_Human": "KEGG",
    "Reactome_2022": "Reactome",
}
PRIMARY_CATEGORY_BY_GENE = {
    "AMACR": "Mitochondrial / metabolic",
    "CGREF1": "Tumor epithelial / cell-state",
    "COA4": "Mitochondrial / metabolic",
    "COMP": "ECM / stromal remodeling",
    "COX17": "Mitochondrial / metabolic",
    "DKC1": "Nucleolar / RNA processing",
    "ENTPD5": "Secretory / glycan trafficking",
    "EPCAM": "Tumor epithelial / cell-state",
    "ERGIC1": "Secretory / glycan trafficking",
    "FABP5": "Mitochondrial / metabolic",
    "FASN": "Mitochondrial / metabolic",
    "FBP2": "Mitochondrial / metabolic",
    "FMC1": "Mitochondrial / metabolic",
    "GALNT7": "Secretory / glycan trafficking",
    "GAR1": "Nucleolar / RNA processing",
    "GLYATL1": "Mitochondrial / metabolic",
    "GMDS": "Secretory / glycan trafficking",
    "GOLM1": "Secretory / glycan trafficking",
    "LAMC3": "ECM / stromal remodeling",
    "MARCKSL1": "ECM / stromal remodeling",
    "NHP2": "Nucleolar / RNA processing",
    "NOLC1": "Nucleolar / RNA processing",
    "PLA2G2A": "ECM / stromal remodeling",
    "POSTN": "ECM / stromal remodeling",
    "PYCR1": "Mitochondrial / metabolic",
    "SFRP4": "ECM / stromal remodeling",
    "SLC37A1": "Secretory / glycan trafficking",
    "SLC4A4": "Tumor epithelial / cell-state",
    "THBS4": "ECM / stromal remodeling",
    "TMSB15B": "ECM / stromal remodeling",
    "TSPAN13": "Tumor epithelial / cell-state",
    "UAP1": "Secretory / glycan trafficking",
    "UGDH": "Secretory / glycan trafficking",
    "VSTM2L": "Tumor epithelial / cell-state",
    "ZBTB7B": "Tumor epithelial / cell-state",
}
SECONDARY_CATEGORY_BY_GENE = {
    "CGREF1": "ECM / stromal remodeling",
    "ENTPD5": "Mitochondrial / metabolic",
    "FABP5": "Tumor epithelial / cell-state",
    "GMDS": "Mitochondrial / metabolic",
    "GOLM1": "Tumor epithelial / cell-state",
    "MARCKSL1": "Tumor epithelial / cell-state",
    "PLA2G2A": "Secretory / glycan trafficking",
    "SLC37A1": "Mitochondrial / metabolic",
    "SLC4A4": "Secretory / glycan trafficking",
    "TMSB15B": "Tumor epithelial / cell-state",
    "UAP1": "Mitochondrial / metabolic",
    "UGDH": "Mitochondrial / metabolic",
    "VSTM2L": "Secretory / glycan trafficking",
}
REVIEW_NOTES = {
    "CGREF1": "Cell-adhesion and growth-arrest annotations support cell-state; secreted property leaves ECM overlap as secondary.",
    "GLYATL1": "Kept in metabolic because reviewed function is acyltransferase activity in glutamine metabolism.",
    "MARCKSL1": "Placed in ECM/stromal because actin remodeling and migration dominate the reviewed annotation.",
    "PLA2G2A": "Placed in ECM/stromal because extracellular phospholipase and integrin/tissue-remodeling functions dominate.",
    "SLC4A4": "Placed in cell-state because basolateral bicarbonate transport is a differentiated epithelial feature.",
    "TMSB15B": "Low-to-medium confidence invasion/cytoskeleton marker; kept in ECM/stromal as migration-associated.",
    "UGDH": "Placed in secretory/glycan because UDP-glucuronate and glycosaminoglycan biosynthesis are the dominant annotations.",
    "VSTM2L": "Sparse annotation; low-confidence cell-state assignment retained until stronger tissue-specific evidence is added.",
    "ZBTB7B": "Lineage transcription-factor logic supports cell-state, but prostate-specific evidence remains limited.",
}
PMID_PATTERN = re.compile(r"PubMed:(\d+)")
PLOT_PREFIX = "grade_progression_gene_category_heatmap"
SUMMARY_FILENAME = "tumor_overlap_gene_category_summary.md"
RUBRIC_FILENAME = "tumor_overlap_gene_category_scoring_rubric.md"
TEMPLATE_FILENAME = "tumor_overlap_gene_category_evidence_template.tsv"
EVIDENCE_FILENAME = "tumor_overlap_gene_category_evidence_first_pass.tsv"
CATEGORY_AUDIT_FILENAME = "tumor_overlap_gene_category_category_audit.tsv"
ENRICHMENT_ALL_FILENAME = "tumor_overlap_gene_category_enrichment_all_terms.tsv"
ENRICHMENT_SIG_FILENAME = "tumor_overlap_gene_category_enrichment_significant_terms.tsv"
ENRICHMENT_GENE_SUPPORT_FILENAME = "tumor_overlap_gene_category_enrichment_gene_support.tsv"
CONFIG_SNAPSHOT_FILENAME = "config_grade_progression_gene_category_analysis.ini"
EDITABLE_PDF_SUFFIX = "_editable_text.pdf"


CATEGORY_KEYWORDS = {
    "Tumor epithelial / cell-state": {
        "annotation": (
            "cell adhesion",
            "tight junction",
            "epithelial",
            "differentiation",
            "growth arrest",
            "lineage",
            "transcription regulator",
            "stem cell",
            "cell cycle",
            "proliferation",
        ),
        "localization": (
            "cell junction",
            "lateral cell membrane",
            "basolateral cell membrane",
            "cell membrane",
            "membrane",
            "nucleus",
        ),
        "database": (
            "cell adhesion",
            "stem cell",
            "differentiation",
            "regulation of gene expression",
            "transcription",
            "cell cycle",
            "nucleus organization",
        ),
    },
    "Nucleolar / RNA processing": {
        "annotation": (
            "nucleol",
            "cajal body",
            "ribonucleoprotein",
            "ribosome biogenesis",
            "rna-binding",
            "rrna processing",
            "pseudouridine",
            "telomerase",
            "snorna",
            "scarna",
        ),
        "localization": ("nucleus, nucleolus", "nucleus, cajal body", "nucleus"),
        "database": (
            "rna processing",
            "rrna",
            "ribosome biogenesis",
            "telomerase",
            "nucleolus organization",
            "pseudouridine",
        ),
    },
    "Secretory / glycan trafficking": {
        "annotation": (
            "endoplasmic reticulum",
            "golgi",
            "glycosyl",
            "glycan",
            "glycoprotein",
            "udp-",
            "udp-glcnac",
            "udp-glucuronate",
            "galnac",
            "hexosamine",
            "sugar nucleotide",
            "gdp-l-fucose",
            "gdp-mannose",
            "vesicle",
            "glucose-6-phosphate",
        ),
        "localization": (
            "endoplasmic reticulum",
            "golgi",
            "secreted",
            "er-golgi",
            "golgi apparatus",
        ),
        "database": (
            "glycosyl",
            "golgi",
            "vesicle-mediated transport",
            "udp",
            "udp-glucuronate",
            "udp-n-acetylglucosamine",
            "fucose",
            "glycosaminoglycan",
            "protein folding",
        ),
    },
    "Mitochondrial / metabolic": {
        "annotation": (
            "mitochondr",
            "peroxisome",
            "fatty acid",
            "lipid",
            "oxidoreductase",
            "respiratory chain",
            "gluconeogenesis",
            "biosynthetic process",
            "acyltransferase",
            "metabolic",
        ),
        "localization": ("mitochondr", "peroxisome", "cytoplasm"),
        "database": (
            "metabolic process",
            "biosynthetic process",
            "respiratory chain",
            "fatty acid",
            "gluconeogenesis",
            "mitochondrial",
            "cholesterol",
            "beta-oxidation",
        ),
    },
    "ECM / stromal remodeling": {
        "annotation": (
            "extracellular matrix",
            "ecm",
            "extracellular",
            "basement membrane",
            "cell migration",
            "actin cytoskeleton",
            "lamellipodium",
            "filopodium",
            "integrin",
            "inflammatory",
            "tissue regeneration",
            "wnt signaling",
            "phospholipase",
            "matrix",
            "adhesion",
            "tissue remodeling",
        ),
        "localization": (
            "extracellular matrix",
            "basement membrane",
            "secreted",
            "cell membrane",
            "cytoskeleton",
        ),
        "database": (
            "extracellular matrix",
            "extracellular matrix organization",
            "cell adhesion",
            "migration",
            "tissue remodeling",
            "actin filament organization",
            "basement membrane",
            "wnt signaling",
            "integrin",
            "inflammatory response",
            "phospholipid",
        ),
    },
}


@dataclass(frozen=True)
class GeneCategoryConfig:
    run_dir: Path
    output_dir: Path
    annotation_tsv: Path
    plot_prefix: str = PLOT_PREFIX
    summary_filename: str = SUMMARY_FILENAME
    rubric_filename: str = RUBRIC_FILENAME
    template_filename: str = TEMPLATE_FILENAME
    evidence_filename: str = EVIDENCE_FILENAME
    category_audit_filename: str = CATEGORY_AUDIT_FILENAME
    enrichment_all_filename: str = ENRICHMENT_ALL_FILENAME
    enrichment_sig_filename: str = ENRICHMENT_SIG_FILENAME
    enrichment_gene_support_filename: str = ENRICHMENT_GENE_SUPPORT_FILENAME
    config_snapshot_filename: str = CONFIG_SNAPSHOT_FILENAME
    heatmap_groups: tuple[str, ...] = DEFAULT_HEATMAP_GROUPS
    enrichment_libraries: tuple[str, ...] = DEFAULT_ENRICHMENT_LIBRARIES
    adj_p_cutoff: float = 0.05
    min_overlap_hits: int = 2
    top_terms_per_group: int = 5


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


def _build_config(config: configparser.ConfigParser) -> GeneCategoryConfig:
    return GeneCategoryConfig(
        run_dir=_get_required_path(config, "input", "run_dir"),
        output_dir=_get_required_path(config, "output", "output_dir"),
        annotation_tsv=_get_required_path(config, "input", "annotation_tsv"),
        plot_prefix=_get_optional_value(config, "settings", "plot_prefix", PLOT_PREFIX),
        summary_filename=_get_optional_value(config, "settings", "summary_filename", SUMMARY_FILENAME),
        rubric_filename=_get_optional_value(config, "settings", "rubric_filename", RUBRIC_FILENAME),
        template_filename=_get_optional_value(config, "settings", "template_filename", TEMPLATE_FILENAME),
        evidence_filename=_get_optional_value(config, "settings", "evidence_filename", EVIDENCE_FILENAME),
        category_audit_filename=_get_optional_value(
            config,
            "settings",
            "category_audit_filename",
            CATEGORY_AUDIT_FILENAME,
        ),
        enrichment_all_filename=_get_optional_value(
            config,
            "settings",
            "enrichment_all_filename",
            ENRICHMENT_ALL_FILENAME,
        ),
        enrichment_sig_filename=_get_optional_value(
            config,
            "settings",
            "enrichment_sig_filename",
            ENRICHMENT_SIG_FILENAME,
        ),
        enrichment_gene_support_filename=_get_optional_value(
            config,
            "settings",
            "enrichment_gene_support_filename",
            ENRICHMENT_GENE_SUPPORT_FILENAME,
        ),
        config_snapshot_filename=_get_optional_value(
            config,
            "settings",
            "config_snapshot_filename",
            CONFIG_SNAPSHOT_FILENAME,
        ),
        heatmap_groups=_parse_csv_list(
            _get_optional_value(
                config,
                "settings",
                "heatmap_groups",
                ",".join(DEFAULT_HEATMAP_GROUPS),
            )
        ),
        enrichment_libraries=_parse_csv_list(
            _get_optional_value(
                config,
                "settings",
                "enrichment_libraries",
                ",".join(DEFAULT_ENRICHMENT_LIBRARIES),
            )
        ),
        adj_p_cutoff=float(_get_optional_value(config, "settings", "adj_p_cutoff", "0.05")),
        min_overlap_hits=int(float(_get_optional_value(config, "settings", "min_overlap_hits", "2"))),
        top_terms_per_group=int(float(_get_optional_value(config, "settings", "top_terms_per_group", "5"))),
    )


def _read_gene_list(path: Path) -> set[str]:
    if not path.exists():
        msg = f"Gene list was not found: {path}"
        raise FileNotFoundError(msg)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    return {line.strip() for line in text.splitlines() if line.strip()}


def _read_grade_sets(run_dir: Path) -> tuple[set[str], dict[str, set[str]]]:
    tumor_set = _read_gene_list(run_dir / "genes_tumor_vs_normal" / "S-U_genes.txt")
    grade_sets: dict[str, set[str]] = {}
    for grade in PATTERN_GRADE_ORDER:
        grade_set = _read_gene_list(run_dir / f"genes_{grade}_vs_normal" / "S-U_genes.txt")
        grade_sets[grade] = grade_set & tumor_set
    return tumor_set, grade_sets


def _pattern_category(pattern: str) -> str:
    if pattern == "1111":
        return "Trunk"
    if pattern in {"0111", "0011", "0001", "0101"}:
        return "Late gain"
    if pattern in {"1110", "1100", "1000"}:
        return "Early loss"
    return "Variable shared"


def _sample_count(run_dir: Path, grade: str) -> int:
    path = run_dir / f"{grade}_vs_normal_sample_info.tsv"
    frame = pd.read_csv(path, sep="\t")
    mask = frame["group"].astype(str).str.upper() == grade.upper()
    return int(frame.loc[mask, "sample"].astype(str).nunique())


def _validate_groups(groups: tuple[str, ...]) -> None:
    missing = [group for group in groups if group not in HEATMAP_GROUP_LABELS]
    if missing:
        msg = f"Unsupported heatmap groups: {', '.join(missing)}"
        raise ValueError(msg)


def _extract_pmids(text: str) -> str:
    pmids = []
    for pmid in PMID_PATTERN.findall(text or ""):
        if pmid not in pmids:
            pmids.append(pmid)
    return ";".join(pmids[:6])


def _safe_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_text(value: object) -> str:
    return _safe_text(value).casefold()


def _count_hits(text: str, keywords: tuple[str, ...], cap: int) -> tuple[int, list[str]]:
    hits = [keyword for keyword in keywords if keyword.casefold() in text]
    dedup_hits = list(dict.fromkeys(hits))
    return min(cap, len(dedup_hits)), dedup_hits[:cap]


def build_expression_frame(
    run_dir: Path,
    heatmap_groups: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, set[str]], dict[str, int]]:
    _validate_groups(heatmap_groups)
    _, grade_sets = _read_grade_sets(run_dir)
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
    trend["delta_G5_G2"] = trend["GG5"] - trend["GG2"]
    trend["mean_tumor_grade"] = trend[list(PATTERN_GRADE_ORDER)].mean(axis=1)

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

    sample_counts = {grade: _sample_count(run_dir, grade) for grade in PATTERN_GRADE_ORDER}
    return trend, grade_sets, sample_counts


def _annotation_frame(annotation_tsv: Path) -> pd.DataFrame:
    frame = pd.read_csv(annotation_tsv, sep="\t").fillna("")
    required = {
        "gene",
        "protein_name",
        "function_text",
        "subcellular_location",
        "keywords",
        "go_process",
        "reactome",
        "kegg",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        msg = f"Annotation TSV is missing columns: {', '.join(missing)}"
        raise ValueError(msg)
    return frame


def _trajectory_score(row: pd.Series, category: str) -> int:
    pattern_category = str(row["pattern_category"])
    delta = float(row["delta_G5_G2"])
    normal_value = float(row["normal"])
    tumor_mean = float(row["mean_tumor_grade"])

    if category == "ECM / stromal remodeling":
        return int(delta > 0 or pattern_category == "Late gain")
    if category == "Mitochondrial / metabolic":
        return int(delta > 0 or pattern_category in {"Late gain", "Trunk"})
    if category == "Nucleolar / RNA processing":
        return int(delta >= 0 or pattern_category in {"Late gain", "Trunk"})
    if category == "Secretory / glycan trafficking":
        return int(delta < 0 or pattern_category == "Early loss")
    if category == "Tumor epithelial / cell-state":
        return int(pattern_category == "Trunk" or tumor_mean > normal_value)
    return 0


def build_evidence_frame(cfg: GeneCategoryConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, set[str]], dict[str, int]]:
    expression_frame, grade_sets, sample_counts = build_expression_frame(cfg.run_dir, cfg.heatmap_groups)
    annotation_frame = _annotation_frame(cfg.annotation_tsv)
    frame = expression_frame.merge(annotation_frame, on="gene", how="left")

    if frame[["protein_name", "function_text", "subcellular_location"]].isna().any().any():
        msg = "Some genes are missing annotation rows in annotation_tsv."
        raise ValueError(msg)

    records: list[dict[str, object]] = []
    score_columns: dict[str, str] = {}
    for category in CATEGORY_ORDER:
        label = (
            category.casefold()
            .replace(" / ", "_")
            .replace(" ", "_")
            .replace("-", "_")
        )
        score_columns[category] = label

    for _, row in frame.iterrows():
        gene = str(row["gene"])
        annotation_text = " ".join(
            [
                _normalize_text(row["protein_name"]),
                _normalize_text(row["function_text"]),
                _normalize_text(row["keywords"]),
            ]
        )
        localization_text = _normalize_text(row["subcellular_location"])
        database_text = " ".join(
            [
                _normalize_text(row["go_process"]),
                _normalize_text(row["reactome"]),
                _normalize_text(row["kegg"]),
            ]
        )

        category_support: dict[str, int] = {}
        category_evidence: dict[str, tuple[str, str, str]] = {}
        for category, keyword_groups in CATEGORY_KEYWORDS.items():
            annotation_score, annotation_hits = _count_hits(
                annotation_text,
                keyword_groups["annotation"],
                cap=2,
            )
            localization_score, localization_hits = _count_hits(
                localization_text,
                keyword_groups["localization"],
                cap=1,
            )
            database_score, database_hits = _count_hits(
                database_text,
                keyword_groups["database"],
                cap=2,
            )
            category_support[category] = annotation_score + localization_score + database_score
            category_evidence[category] = (
                "; ".join(annotation_hits),
                "; ".join(localization_hits),
                "; ".join(database_hits),
            )

        ranked_categories = sorted(
            CATEGORY_ORDER,
            key=lambda category: (category_support[category], -CATEGORY_ORDER.index(category)),
            reverse=True,
        )
        top_support_category = ranked_categories[0]
        primary_category = PRIMARY_CATEGORY_BY_GENE[gene]
        secondary_category = SECONDARY_CATEGORY_BY_GENE.get(gene, "")
        primary_support_score = category_support[primary_category]
        secondary_support_score = category_support[secondary_category] if secondary_category else 0
        trajectory_score = _trajectory_score(row, primary_category)
        conflict_penalty = -1 if secondary_category and secondary_support_score >= max(primary_support_score - 1, 1) else 0
        primary_score = primary_support_score + trajectory_score + conflict_penalty
        confidence = "high" if primary_score >= 5 and conflict_penalty == 0 else "medium" if primary_score >= 4 else "low"

        primary_annotation_evidence, primary_localization_evidence, primary_database_evidence = category_evidence[primary_category]
        review_note = REVIEW_NOTES.get(gene, "")
        if category_support[top_support_category] > primary_support_score:
            mismatch_note = f"Top keyword-support category is {top_support_category}."
            review_note = f"{review_note} {mismatch_note}".strip()

        record = {
            "gene": gene,
            "protein_name": _safe_text(row["protein_name"]),
            "pmid_support": _extract_pmids(_safe_text(row["function_text"])),
            "primary_category": primary_category,
            "primary_score": primary_score,
            "primary_support_score": primary_support_score,
            "secondary_category": secondary_category,
            "secondary_score": secondary_support_score,
            "top_support_category": top_support_category,
            "top_support_score": category_support[top_support_category],
            "manual_primary_matches_top_support": primary_support_score == category_support[top_support_category],
            "trajectory_score": trajectory_score,
            "conflict_penalty": conflict_penalty,
            "confidence": confidence,
            "pattern": row["pattern"],
            "pattern_category": row["pattern_category"],
            "delta_G5_G2": row["delta_G5_G2"],
            "function_text": _safe_text(row["function_text"]),
            "subcellular_location": _safe_text(row["subcellular_location"]),
            "keywords": _safe_text(row["keywords"]),
            "go_process": _safe_text(row["go_process"]),
            "reactome": _safe_text(row["reactome"]),
            "kegg": _safe_text(row["kegg"]),
            "primary_annotation_evidence": primary_annotation_evidence,
            "primary_localization_evidence": primary_localization_evidence,
            "primary_database_evidence": primary_database_evidence,
            "review_note": review_note,
            "curator_primary_category": "",
            "curator_secondary_category": "",
            "curator_confidence": "",
            "curator_note": "",
        }
        for group in cfg.heatmap_groups:
            record[group] = row[group]
            record[f"{group}_z"] = row[f"{group}_z"]
        for category in CATEGORY_ORDER:
            record[f"support_score__{score_columns[category]}"] = category_support[category]
        records.append(record)

    evidence = pd.DataFrame(records)
    pattern_rank = {"Trunk": 0, "Late gain": 1, "Variable shared": 2, "Early loss": 3}
    category_rank = {category: idx for idx, category in enumerate(CATEGORY_ORDER)}
    evidence["category_rank"] = evidence["primary_category"].map(category_rank)
    evidence["pattern_rank"] = evidence["pattern_category"].map(pattern_rank)
    evidence = evidence.sort_values(
        ["category_rank", "pattern_rank", "delta_G5_G2", "gene"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)

    template = evidence.copy()
    for column in [
        "primary_category",
        "primary_score",
        "primary_support_score",
        "secondary_category",
        "secondary_score",
        "confidence",
        "trajectory_score",
        "conflict_penalty",
        "review_note",
    ]:
        template[column] = ""
    return evidence, template, grade_sets, sample_counts


def build_enrichment_groups(evidence: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    union_genes = sorted(evidence["gene"].unique().tolist())
    rows.append(
        {
            "group_name": "union__tumor_overlap_35_genes",
            "group_type": "union",
            "group_label": "All 35 tumor-overlap genes",
            "genes": union_genes,
            "gene_count": len(union_genes),
        }
    )

    for category in CATEGORY_ORDER:
        genes = sorted(evidence.loc[evidence["primary_category"] == category, "gene"].tolist())
        rows.append(
            {
                "group_name": f"category__{category}",
                "group_type": "category",
                "group_label": category,
                "genes": genes,
                "gene_count": len(genes),
            }
        )

    for pattern_name in ("Trunk", "Late gain", "Early loss", "Variable shared"):
        genes = sorted(evidence.loc[evidence["pattern_category"] == pattern_name, "gene"].tolist())
        if len(genes) < 3:
            continue
        rows.append(
            {
                "group_name": f"pattern__{pattern_name}",
                "group_type": "pattern",
                "group_label": pattern_name,
                "genes": genes,
                "gene_count": len(genes),
            }
        )
    return pd.DataFrame(rows)


def _parse_overlap_hits(value: str) -> int:
    if not value or "/" not in value:
        return 0
    numerator, _ = value.split("/", 1)
    return int(float(numerator))


def _run_single_enrichr(
    gene_list: list[str],
    library_name: str,
    adj_p_cutoff: float,
) -> pd.DataFrame:
    try:
        import gseapy
    except ImportError as exc:
        msg = "gseapy is required for gene-category enrichment support."
        raise ImportError(msg) from exc

    if len(gene_list) < 2:
        return pd.DataFrame()

    last_error: Exception | None = None
    for attempt in range(5):
        try:
            result = gseapy.enrichr(
                gene_list=gene_list,
                gene_sets=library_name,
                organism="human",
                outdir=None,
                cutoff=1.0,
            ).results.copy()
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
    result["is_significant"] = result["Adjusted P-value"] <= adj_p_cutoff
    result["library_label"] = LIBRARY_LABELS.get(library_name, library_name)
    return result


def run_support_enrichment(cfg: GeneCategoryConfig, evidence: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    group_frame = build_enrichment_groups(evidence)
    all_results: list[pd.DataFrame] = []

    for _, group_row in group_frame.iterrows():
        gene_list = list(group_row["genes"])
        for library_name in cfg.enrichment_libraries:
            result = _run_single_enrichr(gene_list, library_name, cfg.adj_p_cutoff)
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
                        "library_label",
                    ]
                )
            result["library_name"] = library_name
            result["group_name"] = group_row["group_name"]
            result["group_type"] = group_row["group_type"]
            result["group_label"] = group_row["group_label"]
            result["input_gene_count"] = int(group_row["gene_count"])
            result["input_genes"] = ";".join(gene_list)
            all_results.append(result)

    enrichment_all = pd.concat(all_results, ignore_index=True)
    enrichment_sig = enrichment_all.loc[
        (enrichment_all["is_significant"]) & (enrichment_all["overlap_hits"] >= cfg.min_overlap_hits)
    ].copy()
    if not enrichment_sig.empty:
        enrichment_sig = (
            enrichment_sig.sort_values(
                ["group_type", "group_label", "library_label", "neg_log10_fdr"],
                ascending=[True, True, True, False],
            )
            .groupby(["group_name", "library_name"], group_keys=False)
            .head(cfg.top_terms_per_group)
            .reset_index(drop=True)
        )

    gene_support_rows: list[dict[str, object]] = []
    for gene in evidence["gene"].tolist():
        gene_terms_by_library: dict[str, list[str]] = {}
        gene_groups = set()
        gene_subset = enrichment_sig.loc[enrichment_sig["Genes"].astype(str).str.contains(fr"\b{gene}\b", regex=True)]
        for _, row in gene_subset.iterrows():
            label = f"{row['Term']} [{row['group_label']}]"
            gene_terms_by_library.setdefault(str(row["library_label"]), []).append(label)
            gene_groups.add(str(row["group_label"]))

        gene_support_rows.append(
            {
                "gene": gene,
                "support_groups": "; ".join(sorted(gene_groups)),
                "Hallmark_support": "; ".join(dict.fromkeys(gene_terms_by_library.get("Hallmark", []))),
                "GO_BP_support": "; ".join(dict.fromkeys(gene_terms_by_library.get("GO_BP", []))),
                "KEGG_support": "; ".join(dict.fromkeys(gene_terms_by_library.get("KEGG", []))),
                "Reactome_support": "; ".join(dict.fromkeys(gene_terms_by_library.get("Reactome", []))),
            }
        )
    gene_support = pd.DataFrame(gene_support_rows)
    return enrichment_all, enrichment_sig, gene_support


def build_category_audit(evidence: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for category in CATEGORY_ORDER:
        subset = evidence.loc[evidence["primary_category"] == category].copy()
        rows.append(
            {
                "category": category,
                "n_genes": int(len(subset)),
                "genes": ";".join(subset["gene"].tolist()),
                "high_confidence_n": int((subset["confidence"] == "high").sum()),
                "medium_confidence_n": int((subset["confidence"] == "medium").sum()),
                "low_confidence_n": int((subset["confidence"] == "low").sum()),
                "support_mismatch_n": int((~subset["manual_primary_matches_top_support"]).sum()),
                "mean_primary_score": round(float(subset["primary_score"].mean()), 3),
                "mean_delta_G5_G2": round(float(subset["delta_G5_G2"].mean()), 3),
                "pattern_counts": "; ".join(
                    f"{pattern}={count}"
                    for pattern, count in subset["pattern_category"].value_counts().sort_index().items()
                ),
            }
        )
    return pd.DataFrame(rows)


def _save_table(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, sep="\t", index=False)


def write_outputs(
    cfg: GeneCategoryConfig,
    evidence: pd.DataFrame,
    template: pd.DataFrame,
    category_audit: pd.DataFrame,
    enrichment_all: pd.DataFrame,
    enrichment_sig: pd.DataFrame,
    gene_support: pd.DataFrame,
) -> dict[str, Path]:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    evidence_out = evidence.merge(gene_support, on="gene", how="left")
    template_out = template.merge(gene_support, on="gene", how="left")

    paths = {
        "evidence": cfg.output_dir / cfg.evidence_filename,
        "template": cfg.output_dir / cfg.template_filename,
        "category_audit": cfg.output_dir / cfg.category_audit_filename,
        "enrichment_all": cfg.output_dir / cfg.enrichment_all_filename,
        "enrichment_sig": cfg.output_dir / cfg.enrichment_sig_filename,
        "gene_support": cfg.output_dir / cfg.enrichment_gene_support_filename,
    }
    _save_table(paths["evidence"], evidence_out)
    _save_table(paths["template"], template_out)
    _save_table(paths["category_audit"], category_audit)
    _save_table(paths["enrichment_all"], enrichment_all)
    _save_table(paths["enrichment_sig"], enrichment_sig)
    _save_table(paths["gene_support"], gene_support)
    return paths


def write_config_snapshot(cfg: GeneCategoryConfig) -> Path:
    parser = configparser.ConfigParser()
    parser["task"] = {"name": "grade_progression_gene_category_analysis"}
    parser["input"] = {
        "run_dir": str(cfg.run_dir),
        "annotation_tsv": str(cfg.annotation_tsv),
    }
    parser["output"] = {"output_dir": str(cfg.output_dir)}
    parser["settings"] = {
        "plot_prefix": cfg.plot_prefix,
        "summary_filename": cfg.summary_filename,
        "rubric_filename": cfg.rubric_filename,
        "template_filename": cfg.template_filename,
        "evidence_filename": cfg.evidence_filename,
        "category_audit_filename": cfg.category_audit_filename,
        "enrichment_all_filename": cfg.enrichment_all_filename,
        "enrichment_sig_filename": cfg.enrichment_sig_filename,
        "enrichment_gene_support_filename": cfg.enrichment_gene_support_filename,
        "config_snapshot_filename": cfg.config_snapshot_filename,
        "heatmap_groups": ",".join(cfg.heatmap_groups),
        "enrichment_libraries": ",".join(cfg.enrichment_libraries),
        "adj_p_cutoff": str(cfg.adj_p_cutoff),
        "min_overlap_hits": str(cfg.min_overlap_hits),
        "top_terms_per_group": str(cfg.top_terms_per_group),
    }
    config_path = cfg.output_dir / cfg.config_snapshot_filename
    with config_path.open("w", encoding="utf-8-sig") as handle:
        parser.write(handle)
    return config_path


def write_rubric(cfg: GeneCategoryConfig) -> Path:
    lines = [
        "# Tumor-overlap Gene Category Scoring Rubric",
        "",
        "## Fixed primary categories",
        "- Tumor epithelial / cell-state",
        "- Nucleolar / RNA processing",
        "- Secretory / glycan trafficking",
        "- Mitochondrial / metabolic",
        "- ECM / stromal remodeling",
        "",
        "## Support-score definition",
        "- Each category gets a keyword-based support score from reviewed annotation text.",
        "- `annotation` component: 0-2 from protein name, function text, and UniProt keywords.",
        "- `localization` component: 0-1 from subcellular location.",
        "- `database` component: 0-2 from GO process, Reactome, and KEGG fields.",
        "- `support_score__*` columns in the evidence table are the sum of those three components.",
        "",
        "## Final classification score",
        "- `primary_score = primary_support_score + trajectory_score + conflict_penalty`.",
        "- `trajectory_score`: 0-1 based on whether the expression trajectory is consistent with the proposed primary category.",
        "- `conflict_penalty`: 0 or -1 when the secondary category has nearly the same support as the primary category.",
        "",
        "## Confidence labels",
        "- `high`: primary_score >= 5 and no conflict penalty.",
        "- `medium`: primary_score >= 4.",
        "- `low`: primary_score < 4.",
        "",
        "## Set-membership pattern definitions",
        "- Pattern order is fixed as `GG2, GG3, GG4, GG5`.",
        "- `Trunk` = `1111`.",
        "- `Late gain` = `0111`, `0011`, `0001`, `0101`.",
        "- `Early loss` = `1110`, `1100`, `1000`.",
        "- `Variable shared` = all remaining shared patterns such as `1101`.",
        "",
        "## Review policy",
        "- `top_support_category` is the strongest keyword-derived category and is reported for audit, not as a forced override.",
        "- `manual_primary_matches_top_support = False` flags genes that need closer review in the next iteration.",
    ]
    path = cfg.output_dir / cfg.rubric_filename
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return path


def write_summary(
    cfg: GeneCategoryConfig,
    evidence: pd.DataFrame,
    category_audit: pd.DataFrame,
    enrichment_sig: pd.DataFrame,
    sample_counts: dict[str, int],
    output_paths: dict[str, Path],
    figure_png: Path,
    figure_pdf: Path,
    rubric_path: Path,
    config_path: Path,
) -> Path:
    mismatch_genes = evidence.loc[~evidence["manual_primary_matches_top_support"], "gene"].tolist()
    low_confidence_genes = evidence.loc[evidence["confidence"] == "low", "gene"].tolist()
    top_up = evidence.sort_values("delta_G5_G2", ascending=False).head(6)["gene"].tolist()
    top_down = evidence.sort_values("delta_G5_G2", ascending=True).head(6)["gene"].tolist()

    enrichment_lines: list[str] = []
    if not enrichment_sig.empty:
        union_terms = enrichment_sig.loc[enrichment_sig["group_type"] == "union"].copy()
        if not union_terms.empty:
            for library in ["Hallmark", "GO_BP", "KEGG", "Reactome"]:
                subset = union_terms.loc[union_terms["library_label"] == library].head(3)
                if subset.empty:
                    continue
                enrichment_lines.append(
                    f"- {library}: " + "; ".join(
                        f"{term} (FDR={adj_p:.3g})"
                        for term, adj_p in zip(subset["Term"], subset["Adjusted P-value"])
                    )
                )

    lines = [
        "# Tumor-overlap Gene Category Analysis Summary",
        "",
        "## Generated files",
        f"- Figure PNG: `{figure_png.name}`",
        f"- Editable-text PDF: `{figure_pdf.name}`",
        f"- Evidence table: `{output_paths['evidence'].name}`",
        f"- Evidence template: `{output_paths['template'].name}`",
        f"- Category audit: `{output_paths['category_audit'].name}`",
        f"- Enrichment all terms: `{output_paths['enrichment_all'].name}`",
        f"- Enrichment significant terms: `{output_paths['enrichment_sig'].name}`",
        f"- Gene-level enrichment support: `{output_paths['gene_support'].name}`",
        f"- Rubric: `{rubric_path.name}`",
        f"- Config snapshot: `{config_path.name}`",
        "",
        "## Scope",
        f"- Total genes classified: {int(len(evidence))}.",
        (
            f"- Sample counts used for progression context: GG2 n={sample_counts['GG2']}, "
            f"GG3 n={sample_counts['GG3']}, GG4 n={sample_counts['GG4']}, GG5 n={sample_counts['GG5']}."
        ),
        "- Expression columns shown in the heatmap: Normal, GG1, GG2, GG3, GG4, GG5.",
        "",
        "## First-pass category calls",
    ]
    for _, row in category_audit.iterrows():
        lines.append(
            f"- {row['category']}: n={row['n_genes']} ({row['high_confidence_n']} high / {row['medium_confidence_n']} medium / {row['low_confidence_n']} low)."
        )
    lines.extend(
        [
            "",
            "## Progression readout",
            f"- Largest positive GG5-GG2 shifts: {', '.join(top_up)}.",
            f"- Largest negative GG5-GG2 shifts: {', '.join(top_down)}.",
            f"- Manual-vs-top-support mismatches needing review: {', '.join(mismatch_genes) if mismatch_genes else 'None'}.",
            f"- Low-confidence genes: {', '.join(low_confidence_genes) if low_confidence_genes else 'None'}.",
            "",
            "## Enrichment support",
        ]
    )
    if enrichment_lines:
        lines.extend(enrichment_lines)
    else:
        lines.append("- No significant enrichment terms met the configured FDR and overlap thresholds.")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "- Primary categories remain manually curated first-pass calls.",
            "- The support scores and enrichment tables are audit layers to stabilize the scheme and expose weak assignments.",
        ]
    )
    summary_path = cfg.output_dir / cfg.summary_filename
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return summary_path


def _draw_category_blocks(ax: plt.Axes, evidence: pd.DataFrame) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(len(evidence), 0)
    ax.axis("off")
    current_row = 0
    for category in CATEGORY_ORDER:
        row_count = int((evidence["primary_category"] == category).sum())
        ax.add_patch(
            Rectangle(
                (0, current_row),
                1,
                row_count,
                facecolor=CATEGORY_COLORS[category],
                edgecolor="white",
                linewidth=1.2,
            )
        )
        ax.text(
            0.5,
            current_row + row_count / 2,
            category.replace(" / ", "\n"),
            ha="center",
            va="center",
            fontsize=11.5,
            color="white",
            fontweight="bold",
        )
        current_row += row_count


def _category_boundaries(evidence: pd.DataFrame) -> list[int]:
    boundaries: list[int] = []
    current_row = 0
    for category in CATEGORY_ORDER:
        current_row += int((evidence["primary_category"] == category).sum())
        if current_row < len(evidence):
            boundaries.append(current_row)
    return boundaries


def _draw_module_separators(ax: plt.Axes, boundaries: list[int], color: str = "black", linewidth: float = 1.2) -> None:
    for boundary in boundaries:
        ax.axhline(boundary - 0.5, color=color, linewidth=linewidth)


def _draw_heatmap(
    ax: plt.Axes,
    evidence: pd.DataFrame,
    heatmap_groups: tuple[str, ...],
    color_limits: tuple[float, float],
    boundaries: list[int],
) -> plt.AxesImage:
    values = evidence[[f"{group}_z" for group in heatmap_groups]].to_numpy(dtype=float)
    image = ax.imshow(values, cmap="RdBu_r", vmin=color_limits[0], vmax=color_limits[1], aspect="auto")
    ax.set_xticks(np.arange(len(heatmap_groups)))
    ax.set_xticklabels([HEATMAP_GROUP_LABELS[group] for group in heatmap_groups], rotation=0, fontsize=11)
    ax.set_yticks([])
    ax.set_xlabel("Group", fontsize=11)
    for x in range(len(heatmap_groups) + 1):
        ax.axvline(x - 0.5, color="white", linewidth=0.8)
    for y in range(len(evidence) + 1):
        ax.axhline(y - 0.5, color="white", linewidth=0.4, alpha=0.35)
    if "GG1" in heatmap_groups and "GG2" in heatmap_groups:
        gg1_idx = heatmap_groups.index("GG1")
        ax.axvline(gg1_idx + 0.5, color="black", linewidth=1.0, linestyle="--", alpha=0.75)
    _draw_module_separators(ax, boundaries)
    return image


def _draw_gene_labels(ax: plt.Axes, evidence: pd.DataFrame, boundaries: list[int]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(len(evidence), 0)
    ax.axis("off")
    for idx, gene in enumerate(evidence["gene"].tolist()):
        ax.text(0, idx + 0.5, gene, va="center", ha="left", fontsize=11.2)


def _draw_pattern_strip(ax: plt.Axes, evidence: pd.DataFrame, boundaries: list[int]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(len(evidence), 0)
    ax.set_facecolor("white")
    for idx, label in enumerate(evidence["pattern_category"].tolist()):
        ax.add_patch(
            Rectangle(
                (0, idx),
                1,
                1,
                facecolor=PATTERN_COLORS[label],
                edgecolor="white",
                linewidth=0.6,
            )
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Set-membership\npattern", fontsize=11, pad=8)
    _draw_module_separators(ax, boundaries)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_heatmap(cfg: GeneCategoryConfig, evidence: pd.DataFrame) -> tuple[Path, Path]:
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    sns.set_theme(style="white")

    figure = plt.figure(figsize=(14.0, max(8.6, len(evidence) * 0.34)))
    grid = figure.add_gridspec(
        nrows=1,
        ncols=6,
        width_ratios=[1.45, 4.15, 0.95, 0.22, 0.18, 1.0],
        wspace=0.06,
    )
    block_ax = figure.add_subplot(grid[0, 0])
    heatmap_ax = figure.add_subplot(grid[0, 1])
    gene_ax = figure.add_subplot(grid[0, 2])
    pattern_ax = figure.add_subplot(grid[0, 3])
    colorbar_ax = figure.add_subplot(grid[0, 4])
    legend_ax = figure.add_subplot(grid[0, 5])
    legend_ax.axis("off")

    color_limits = (-2.5, 2.5)
    boundaries = _category_boundaries(evidence)
    _draw_category_blocks(block_ax, evidence)
    image = _draw_heatmap(heatmap_ax, evidence, cfg.heatmap_groups, color_limits, boundaries)
    _draw_gene_labels(gene_ax, evidence, boundaries)
    _draw_pattern_strip(pattern_ax, evidence, boundaries)

    cbar = figure.colorbar(image, cax=colorbar_ax)
    cbar.set_label("Within-protein z-score", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    pattern_handles = [Patch(facecolor=PATTERN_COLORS[name], edgecolor="none", label=name) for name in PATTERN_COLORS]
    legend_ax.legend(
        handles=pattern_handles,
        loc="upper left",
        title="Pattern legend",
        fontsize=11,
        title_fontsize=12,
        frameon=False,
    )
    figure.suptitle(
        "Tumor-overlap S-U proteins across prostate cancer grade progression",
        fontsize=15,
        fontweight="bold",
        x=0.02,
        ha="left",
        y=0.995,
    )
    heatmap_ax.set_title(
        "Figure B. Protein trajectories across Normal, GG1, and GG2-GG5",
        fontsize=11.5,
        fontweight="bold",
        pad=8,
    )
    figure.text(
        0.02,
        0.02,
        "Heatmap columns include Normal and GG1 baselines. Set-membership pattern is defined only from GG2-GG5 tumor-overlap S-U membership.",
        fontsize=8.6,
    )
    figure.subplots_adjust(bottom=0.1, top=0.91, left=0.04, right=0.985)

    png_path = cfg.output_dir / f"{cfg.plot_prefix}.png"
    pdf_path = cfg.output_dir / f"{cfg.plot_prefix}{EDITABLE_PDF_SUFFIX}"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def run_grade_progression_gene_category_analysis(config_path: Path) -> None:
    cfg = _build_config(_read_config(config_path))
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    evidence, template, _, sample_counts = build_evidence_frame(cfg)
    enrichment_all, enrichment_sig, gene_support = run_support_enrichment(cfg, evidence)
    category_audit = build_category_audit(evidence)
    output_paths = write_outputs(
        cfg=cfg,
        evidence=evidence,
        template=template,
        category_audit=category_audit,
        enrichment_all=enrichment_all,
        enrichment_sig=enrichment_sig,
        gene_support=gene_support,
    )
    config_snapshot_path = write_config_snapshot(cfg)
    rubric_path = write_rubric(cfg)
    figure_png, figure_pdf = plot_heatmap(cfg, evidence)
    write_summary(
        cfg=cfg,
        evidence=evidence,
        category_audit=category_audit,
        enrichment_sig=enrichment_sig,
        sample_counts=sample_counts,
        output_paths=output_paths,
        figure_png=figure_png,
        figure_pdf=figure_pdf,
        rubric_path=rubric_path,
        config_path=config_snapshot_path,
    )
