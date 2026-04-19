from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection


DEFAULT_STROMAL_MARKERS = [
    "POSTN",
    "COMP",
    "THBS4",
    "SFRP4",
    "COL1A1",
    "COL1A2",
    "DCN",
    "LUM",
    "ACTA2",
    "TAGLN",
]

DEFAULT_TUMOR_MARKERS = [
    "EPCAM",
    "AMACR",
    "ENTPD5",
    "GOLM1",
    "KRT8",
    "KRT18",
    "FOLH1",
]

SIGNATURE_TERM_NAMES = {
    "myogenesis": ["Myogenesis"],
    "coagulation": ["Coagulation"],
    "emt": ["Epithelial Mesenchymal Transition"],
    "stromal_ecm": ["Focal adhesion", "ECM-receptor interaction"],
}

FOCUS_HALLMARK_TERMS = [
    "Myogenesis",
    "Coagulation",
    "Epithelial Mesenchymal Transition",
    "MYC Targets V1",
    "MYC Targets V2",
    "E2F Targets",
    "Oxidative Phosphorylation",
]

FOCUS_KEGG_TERMS = [
    "Focal adhesion",
    "ECM-receptor interaction",
    "Complement and coagulation cascades",
]


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


def _get_optional_path(
    config: configparser.ConfigParser,
    section: str,
    option: str,
) -> Path | None:
    if not config.has_section(section) or not config.has_option(section, option):
        return None
    value = _strip_quotes(config.get(section, option))
    return Path(value) if value else None


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


def _get_optional_list(
    config: configparser.ConfigParser,
    section: str,
    option: str,
    default: list[str],
) -> list[str]:
    if not config.has_section(section) or not config.has_option(section, option):
        return default
    raw = _strip_quotes(config.get(section, option))
    return [item.strip() for item in raw.split(",") if item.strip()]


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
    msg = f"Unsupported file format for {path}."
    raise ValueError(msg)


def _safe_fdrcorrect(pvalues: pd.Series) -> pd.Series:
    corrected = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = pvalues.notna()
    if valid.any():
        corrected.loc[valid] = fdrcorrection(pvalues.loc[valid])[1]
    return corrected


def _safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    frame = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(frame) < 3:
        return np.nan, np.nan, int(len(frame))
    if frame.iloc[:, 0].nunique() < 2 or frame.iloc[:, 1].nunique() < 2:
        return np.nan, np.nan, int(len(frame))
    rho, pvalue = stats.spearmanr(frame.iloc[:, 0], frame.iloc[:, 1])
    return float(rho), float(pvalue), int(len(frame))


def _read_gmt(path: Path) -> dict[str, list[str]]:
    gene_sets: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            term = fields[0].strip()
            genes = [gene.strip().upper() for gene in fields[2:] if gene.strip()]
            if term and genes:
                gene_sets[term] = genes
    return gene_sets


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _split_gene_field(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [gene.strip().upper() for gene in text.split(";") if gene.strip()]


def _contains_case_insensitive(text: object, patterns: list[str]) -> bool:
    value = str(text).casefold()
    return any(pattern.casefold() in value for pattern in patterns)


def _normalize_pathway_term(value: object) -> str:
    text = str(value).strip()
    text = text.replace("Homo sapiens", "")
    parts = [part for part in text.split() if not part.lower().startswith("hsa")]
    return " ".join(parts).casefold()


def _select_best_sample_column(
    meta: pd.DataFrame,
    labels: pd.Index | pd.Series | list[str],
    preferred: str | None,
) -> tuple[str | None, int]:
    label_set = {str(label) for label in labels}
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(["SampleID", "common_ID", "sample_id", "sample"])
    candidates.extend(
        [
            column for column in meta.columns
            if column not in candidates and meta[column].dtype == object
        ]
    )

    best_column: str | None = None
    best_overlap = 0
    for column in candidates:
        if column not in meta.columns:
            continue
        overlap = int(meta[column].astype(str).isin(label_set).sum())
        if overlap > best_overlap:
            best_overlap = overlap
            best_column = column
    return best_column, best_overlap


def _resolve_base_diff_inputs(base_config_path: Path) -> dict[str, Any]:
    config = _read_config(base_config_path)
    input_dir = _get_required_path(config, "input", "input_dir")
    global_path = _get_required_path(config, "input", "global_path")
    meta_dir = _get_required_path(config, "input", "meta_dir")
    meta_path = _get_required_path(config, "input", "meta_path")
    sample_id_column = _get_optional_value(config, "input", "sample_id_column", "") or None
    return {
        "matrix_path": input_dir / global_path,
        "metadata_path": meta_dir / meta_path,
        "feature_column": _get_optional_value(config, "input", "feature_column", "geneSymbol"),
        "sample_id_column": sample_id_column,
        "group_column": _get_optional_value(config, "input", "group_column", "Tissuetype"),
        "tumor_label": _get_optional_value(config, "input", "group1", "tumor"),
        "normal_label": _get_optional_value(config, "input", "group2", "normal"),
        "purity_column": _get_optional_value(config, "input", "purity_column", "Purity"),
    }


@dataclass
class PurityValidationConfig:
    output_dir: Path
    matrix_path: Path
    metadata_path: Path
    base_diff_config: Path | None = None
    hallmark_terms_path: Path | None = None
    kegg_gsea_results_path: Path | None = None
    original_protein_results_path: Path | None = None
    original_pathway_results_path: Path | None = None
    hallmark_gmt_path: Path | None = None
    kegg_gmt_path: Path | None = None
    feature_column: str = "geneSymbol"
    sample_id_column: str | None = None
    group_column: str = "Tissuetype"
    tumor_label: str = "tumor"
    normal_label: str = "normal"
    purity_column: str = "Purity"
    batch_column: str | None = None
    tumor_filter_column: str | None = None
    tumor_filter_value: str | None = None
    candidate_protein_mode: str = "preset_markers"
    candidate_protein_path: Path | None = None
    candidate_protein_gene_column: str = "gene"
    candidate_protein_list: list[str] | None = None
    high_purity_quantile: float = 0.5
    min_group_size: int = 3
    gsea_permutation_num: int = 1000
    gsea_min_size: int = 15
    gsea_max_size: int = 500
    gsea_seed: int = 123
    substantial_effect_change_pct: float = 30.0
    protein_significance_fdr: float = 0.05
    protein_effect_size_cutoff: float = 0.58
    pathway_significance_fdr: float = 0.05
    stromal_markers: list[str] | None = None
    tumor_markers: list[str] | None = None


def _load_config(config_path: Path) -> PurityValidationConfig:
    config = _read_config(config_path)
    base_diff_config = _get_optional_path(config, "input", "base_diff_config")
    resolved = _resolve_base_diff_inputs(base_diff_config) if base_diff_config else {}

    matrix_path = _get_optional_path(config, "input", "matrix_path") or resolved.get("matrix_path")
    metadata_path = _get_optional_path(config, "input", "metadata_path") or resolved.get("metadata_path")
    if matrix_path is None or metadata_path is None:
        msg = "Provide [input] matrix_path and metadata_path, or [input] base_diff_config."
        raise ValueError(msg)

    return PurityValidationConfig(
        output_dir=_get_required_path(config, "output", "output_dir"),
        matrix_path=matrix_path,
        metadata_path=metadata_path,
        base_diff_config=base_diff_config,
        hallmark_terms_path=_get_optional_path(config, "input", "hallmark_terms_path"),
        kegg_gsea_results_path=_get_optional_path(config, "input", "kegg_gsea_results_path"),
        original_protein_results_path=_get_optional_path(config, "input", "original_protein_results_path"),
        original_pathway_results_path=_get_optional_path(config, "input", "original_pathway_results_path"),
        hallmark_gmt_path=_get_optional_path(config, "input", "hallmark_gmt_path"),
        kegg_gmt_path=_get_optional_path(config, "input", "kegg_gmt_path"),
        feature_column=_get_optional_value(
            config,
            "settings",
            "feature_column",
            resolved.get("feature_column", "geneSymbol"),
        ),
        sample_id_column=(
            _get_optional_value(config, "settings", "sample_id_column", "") or resolved.get("sample_id_column")
        ),
        group_column=_get_optional_value(
            config,
            "settings",
            "group_column",
            resolved.get("group_column", "Tissuetype"),
        ),
        tumor_label=_get_optional_value(
            config,
            "settings",
            "tumor_label",
            resolved.get("tumor_label", "tumor"),
        ),
        normal_label=_get_optional_value(
            config,
            "settings",
            "normal_label",
            resolved.get("normal_label", "normal"),
        ),
        purity_column=_get_optional_value(
            config,
            "settings",
            "purity_column",
            resolved.get("purity_column", "Purity"),
        ),
        batch_column=_get_optional_value(config, "settings", "batch_column", "") or None,
        tumor_filter_column=_get_optional_value(config, "settings", "tumor_filter_column", "") or None,
        tumor_filter_value=_get_optional_value(config, "settings", "tumor_filter_value", "") or None,
        candidate_protein_mode=_get_optional_value(config, "settings", "candidate_protein_mode", "preset_markers"),
        candidate_protein_path=_get_optional_path(config, "settings", "candidate_protein_path"),
        candidate_protein_gene_column=_get_optional_value(config, "settings", "candidate_protein_gene_column", "gene"),
        candidate_protein_list=[gene.upper() for gene in _get_optional_list(config, "settings", "candidate_protein_list", [])],
        high_purity_quantile=_get_optional_float(config, "settings", "high_purity_quantile", 0.5),
        min_group_size=_get_optional_int(config, "settings", "min_group_size", 3),
        gsea_permutation_num=_get_optional_int(config, "settings", "gsea_permutation_num", 1000),
        gsea_min_size=_get_optional_int(config, "settings", "gsea_min_size", 15),
        gsea_max_size=_get_optional_int(config, "settings", "gsea_max_size", 500),
        gsea_seed=_get_optional_int(config, "settings", "gsea_seed", 123),
        substantial_effect_change_pct=_get_optional_float(
            config,
            "settings",
            "substantial_effect_change_pct",
            30.0,
        ),
        protein_significance_fdr=_get_optional_float(config, "settings", "protein_significance_fdr", 0.05),
        protein_effect_size_cutoff=_get_optional_float(config, "settings", "protein_effect_size_cutoff", 0.58),
        pathway_significance_fdr=_get_optional_float(config, "settings", "pathway_significance_fdr", 0.05),
        stromal_markers=[gene.upper() for gene in _get_optional_list(
            config,
            "settings",
            "stromal_markers",
            DEFAULT_STROMAL_MARKERS,
        )],
        tumor_markers=[gene.upper() for gene in _get_optional_list(
            config,
            "settings",
            "tumor_markers",
            DEFAULT_TUMOR_MARKERS,
        )],
    )


def _prepare_matrix_and_metadata(cfg: PurityValidationConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    matrix_raw = _read_table_auto(cfg.matrix_path)
    meta_raw = _read_table_auto(cfg.metadata_path)

    wide_sample_col, wide_overlap = _select_best_sample_column(meta_raw, matrix_raw.columns.astype(str), cfg.sample_id_column)
    first_column = matrix_raw.columns[0]
    long_sample_col, long_overlap = _select_best_sample_column(meta_raw, matrix_raw[first_column].astype(str), cfg.sample_id_column)

    if wide_overlap <= 0 and long_overlap <= 0:
        msg = "Unable to align matrix samples to metadata. No overlapping sample identifiers were detected."
        raise ValueError(msg)

    orientation = "proteins_in_rows_samples_in_columns"
    matrix = matrix_raw.copy()
    if long_overlap > wide_overlap:
        orientation = "samples_in_rows_proteins_in_columns"
        sample_column_in_matrix = first_column
        duplicated_samples = int(matrix[sample_column_in_matrix].astype(str).duplicated().sum())
        matrix = matrix.set_index(sample_column_in_matrix)
        matrix = matrix.apply(pd.to_numeric, errors="coerce").T
        matrix.index = matrix.index.astype(str).str.strip().str.upper()
        matrix = matrix.groupby(level=0).mean()
        sample_id_column = long_sample_col
        duplicated_features = 0
    else:
        feature_column = cfg.feature_column if cfg.feature_column in matrix.columns else matrix.columns[0]
        matrix[feature_column] = matrix[feature_column].astype(str).str.strip().str.upper()
        duplicated_features = int(matrix[feature_column].duplicated().sum())
        if duplicated_features > 0:
            matrix = matrix.groupby(feature_column, as_index=False).mean(numeric_only=True)
        sample_columns = [column for column in matrix.columns if column != feature_column]
        duplicated_samples = int(pd.Index(sample_columns).duplicated().sum())
        matrix = matrix.set_index(feature_column)
        matrix = matrix.apply(pd.to_numeric, errors="coerce")
        if duplicated_samples > 0:
            matrix = matrix.T.groupby(level=0).mean().T
        sample_id_column = wide_sample_col

    if sample_id_column is None:
        msg = "Unable to determine the metadata sample ID column."
        raise ValueError(msg)

    meta = meta_raw.copy()
    meta[sample_id_column] = meta[sample_id_column].astype(str)
    duplicated_metadata_samples = int(meta[sample_id_column].duplicated().sum())
    if duplicated_metadata_samples > 0:
        meta = meta.drop_duplicates(subset=[sample_id_column], keep="first")

    matched_samples = [sample for sample in matrix.columns if sample in set(meta[sample_id_column])]
    matrix = matrix.loc[:, matched_samples]
    matrix = matrix.dropna(axis=0, how="all")
    meta = meta.loc[meta[sample_id_column].isin(matched_samples)].copy()
    meta = meta.set_index(sample_id_column).loc[matched_samples].copy()

    clean_groups = meta[cfg.group_column].astype(str).str.casefold()
    purity_numeric = pd.to_numeric(meta[cfg.purity_column], errors="coerce")
    meta["group"] = clean_groups
    meta["purity_numeric"] = purity_numeric
    meta["is_tumor"] = (clean_groups == cfg.tumor_label.casefold()).astype(int)
    meta["is_normal"] = (clean_groups == cfg.normal_label.casefold()).astype(int)
    meta["passes_tumor_filter"] = 1
    if cfg.tumor_filter_column and cfg.tumor_filter_value:
        if cfg.tumor_filter_column not in meta.columns:
            msg = f"Tumor filter column '{cfg.tumor_filter_column}' was not found in metadata."
            raise ValueError(msg)
        tumor_filter_mask = meta[cfg.tumor_filter_column].astype(str).str.casefold() == cfg.tumor_filter_value.casefold()
        meta.loc[meta["is_tumor"].eq(1), "passes_tumor_filter"] = tumor_filter_mask.loc[meta["is_tumor"].eq(1)].astype(int)
        keep_mask = meta["is_normal"].eq(1) | (meta["is_tumor"].eq(1) & meta["passes_tumor_filter"].eq(1))
        meta = meta.loc[keep_mask].copy()
        matrix = matrix.loc[:, meta.index].copy()

    summary = {
        "orientation": orientation,
        "matrix_path": str(cfg.matrix_path),
        "metadata_path": str(cfg.metadata_path),
        "sample_id_column": sample_id_column,
        "n_raw_matrix_rows": int(matrix_raw.shape[0]),
        "n_raw_matrix_columns": int(matrix_raw.shape[1]),
        "n_raw_metadata_samples": int(meta_raw.shape[0]),
        "n_retained_proteins": int(matrix.shape[0]),
        "n_retained_samples": int(matrix.shape[1]),
        "n_retained_tumor": int(meta["is_tumor"].sum()),
        "n_retained_normal": int(meta["is_normal"].sum()),
        "n_tumor_with_purity": int((meta["is_tumor"].eq(1) & meta["purity_numeric"].notna()).sum()),
        "tumor_filter_column": cfg.tumor_filter_column or "",
        "tumor_filter_value": cfg.tumor_filter_value or "",
        "n_retained_tumor_passing_filter": int((meta["is_tumor"].eq(1) & meta["passes_tumor_filter"].eq(1)).sum()),
        "duplicated_features": int(duplicated_features),
        "duplicated_matrix_samples": int(duplicated_samples),
        "duplicated_metadata_samples": int(duplicated_metadata_samples),
    }
    return matrix, meta, summary


def _build_marker_availability(
    matrix: pd.DataFrame,
    stromal_markers: list[str],
    tumor_markers: list[str],
) -> pd.DataFrame:
    available = set(matrix.index)
    rows: list[dict[str, Any]] = []
    for marker_type, markers in [
        ("stromal_ecm_marker", stromal_markers),
        ("tumor_epithelial_marker", tumor_markers),
    ]:
        for gene in markers:
            rows.append(
                {
                    "marker_type": marker_type,
                    "gene": gene,
                    "present_in_matrix": gene in available,
                }
            )
    return pd.DataFrame(rows)


def _extract_signature_genes_from_terms(
    path: Path | None,
    term_patterns: dict[str, list[str]],
    gene_column_candidates: list[str],
) -> dict[str, list[str]]:
    extracted = {name: [] for name in term_patterns}
    if path is None or not path.exists():
        return extracted

    table = _read_table_auto(path)
    term_column = next(
        (
            column for column in ["Term", "term_label", "term", "Name"]
            if column in table.columns
        ),
        None,
    )
    gene_column = next((column for column in gene_column_candidates if column in table.columns), None)
    if term_column is None or gene_column is None:
        return extracted

    for signature_name, patterns in term_patterns.items():
        rows = table.loc[table[term_column].map(lambda value: _contains_case_insensitive(value, patterns))]
        genes: list[str] = []
        for value in rows[gene_column]:
            genes.extend(_split_gene_field(value))
        extracted[signature_name] = _unique(genes)
    return extracted


def _build_signature_definitions(cfg: PurityValidationConfig) -> tuple[dict[str, list[str]], pd.DataFrame]:
    signatures: dict[str, list[str]] = {
        "stromal_ecm_score": list(cfg.stromal_markers or []),
        "tumor_epithelial_score": list(cfg.tumor_markers or []),
        "myogenesis_score": [],
        "coagulation_score": [],
        "emt_score": [],
    }
    rows: list[dict[str, Any]] = []

    hallmark_from_terms = _extract_signature_genes_from_terms(
        cfg.hallmark_terms_path,
        {
            "myogenesis_score": SIGNATURE_TERM_NAMES["myogenesis"],
            "coagulation_score": SIGNATURE_TERM_NAMES["coagulation"],
            "emt_score": SIGNATURE_TERM_NAMES["emt"],
        },
        ["Genes", "Lead_genes", "genes"],
    )
    for key, genes in hallmark_from_terms.items():
        if genes:
            signatures[key] = genes
            rows.append(
                {
                    "signature": key,
                    "source": str(cfg.hallmark_terms_path),
                    "source_terms": "; ".join(SIGNATURE_TERM_NAMES[key.replace("_score", "")]),
                    "requested_genes": ";".join(genes),
                }
            )

    kegg_from_gsea = _extract_signature_genes_from_terms(
        cfg.kegg_gsea_results_path,
        {"stromal_ecm_score": SIGNATURE_TERM_NAMES["stromal_ecm"]},
        ["Lead_genes", "Genes", "genes"],
    )
    if kegg_from_gsea["stromal_ecm_score"]:
        signatures["stromal_ecm_score"] = _unique(signatures["stromal_ecm_score"] + kegg_from_gsea["stromal_ecm_score"])
        rows.append(
            {
                "signature": "stromal_ecm_score",
                "source": str(cfg.kegg_gsea_results_path),
                "source_terms": "; ".join(SIGNATURE_TERM_NAMES["stromal_ecm"]),
                "requested_genes": ";".join(signatures["stromal_ecm_score"]),
            }
        )

    if cfg.hallmark_gmt_path and cfg.hallmark_gmt_path.exists():
        hallmark_gmt = _read_gmt(cfg.hallmark_gmt_path)
        fallback_terms = {
            "myogenesis_score": "Myogenesis",
            "coagulation_score": "Coagulation",
            "emt_score": "Epithelial Mesenchymal Transition",
        }
        for signature_name, term_name in fallback_terms.items():
            if not signatures[signature_name] and term_name in hallmark_gmt:
                signatures[signature_name] = hallmark_gmt[term_name]
                rows.append(
                    {
                        "signature": signature_name,
                        "source": str(cfg.hallmark_gmt_path),
                        "source_terms": term_name,
                        "requested_genes": ";".join(signatures[signature_name]),
                    }
                )

    if cfg.kegg_gmt_path and cfg.kegg_gmt_path.exists() and len(signatures["stromal_ecm_score"]) <= len(cfg.stromal_markers or []):
        kegg_gmt = _read_gmt(cfg.kegg_gmt_path)
        fallback_genes: list[str] = []
        for term_name, genes in kegg_gmt.items():
            if _contains_case_insensitive(term_name, SIGNATURE_TERM_NAMES["stromal_ecm"]):
                fallback_genes.extend(genes)
        if fallback_genes:
            signatures["stromal_ecm_score"] = _unique(signatures["stromal_ecm_score"] + fallback_genes)
            rows.append(
                {
                    "signature": "stromal_ecm_score",
                    "source": str(cfg.kegg_gmt_path),
                    "source_terms": "; ".join(SIGNATURE_TERM_NAMES["stromal_ecm"]),
                    "requested_genes": ";".join(signatures["stromal_ecm_score"]),
                }
            )

    if not rows:
        for signature_name, genes in signatures.items():
            rows.append(
                {
                    "signature": signature_name,
                    "source": "built_in_markers",
                    "source_terms": "",
                    "requested_genes": ";".join(genes),
                }
            )
    return signatures, pd.DataFrame(rows)


def _score_signatures(
    matrix: pd.DataFrame,
    meta: pd.DataFrame,
    sample_id_column: str,
    signatures: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_means = matrix.mean(axis=1, skipna=True)
    row_stds = matrix.std(axis=1, skipna=True, ddof=0).replace(0.0, np.nan)
    z_matrix = matrix.sub(row_means, axis=0).div(row_stds, axis=0)

    score_table = meta.copy()
    score_table.insert(0, sample_id_column, score_table.index)
    definition_rows: list[dict[str, Any]] = []
    for signature_name, genes in signatures.items():
        genes_upper = _unique([gene.upper() for gene in genes])
        available = [gene for gene in genes_upper if gene in z_matrix.index]
        missing = [gene for gene in genes_upper if gene not in z_matrix.index]
        if available:
            score_table[signature_name] = z_matrix.loc[available].mean(axis=0, skipna=True).reindex(score_table.index).to_numpy()
        else:
            score_table[signature_name] = np.nan
        score_table[f"{signature_name}_n_genes"] = len(available)
        definition_rows.append(
            {
                "signature": signature_name,
                "requested_gene_count": len(genes_upper),
                "available_gene_count": len(available),
                "missing_gene_count": len(missing),
                "requested_genes": ";".join(genes_upper),
                "available_genes": ";".join(available),
                "missing_genes": ";".join(missing),
            }
        )
    return score_table.reset_index(drop=True), pd.DataFrame(definition_rows)


def _build_design_matrix(
    meta: pd.DataFrame,
    group_column: str,
    batch_column: str | None,
) -> pd.DataFrame:
    design = pd.DataFrame(index=meta.index)
    design["const"] = 1.0
    design[group_column] = pd.to_numeric(meta[group_column], errors="coerce")
    if batch_column and batch_column in meta.columns:
        batch_series = meta[batch_column].astype(str)
        if batch_series.nunique(dropna=True) > 1:
            dummies = pd.get_dummies(batch_series, prefix="batch", drop_first=True, dtype=float)
            design = pd.concat([design, dummies], axis=1)
    return design


def _fit_linear_coefficient(
    response: pd.Series,
    design: pd.DataFrame,
    coefficient: str,
    min_group_size: int,
    group_column: str,
) -> dict[str, float]:
    frame = design.copy()
    frame["response"] = pd.to_numeric(response, errors="coerce").reindex(frame.index)
    valid = frame["response"].notna() & frame.drop(columns=["response"]).notna().all(axis=1)
    frame = frame.loc[valid].copy()

    if frame.empty or group_column not in frame.columns:
        return {"beta": np.nan, "stderr": np.nan, "t_value": np.nan, "p_value": np.nan, "n": 0, "adj_r2": np.nan}
    if frame[group_column].nunique() < 2:
        return {"beta": np.nan, "stderr": np.nan, "t_value": np.nan, "p_value": np.nan, "n": int(len(frame)), "adj_r2": np.nan}

    group_counts = frame[group_column].round().value_counts()
    if group_counts.get(0.0, 0) < min_group_size or group_counts.get(1.0, 0) < min_group_size:
        return {"beta": np.nan, "stderr": np.nan, "t_value": np.nan, "p_value": np.nan, "n": int(len(frame)), "adj_r2": np.nan}

    X = frame.drop(columns=["response"]).to_numpy(dtype=float)
    y = frame["response"].to_numpy(dtype=float)
    n_obs, n_coef = X.shape
    if n_obs <= n_coef:
        return {"beta": np.nan, "stderr": np.nan, "t_value": np.nan, "p_value": np.nan, "n": int(n_obs), "adj_r2": np.nan}

    xtx = X.T @ X
    if np.linalg.matrix_rank(xtx) < n_coef:
        return {"beta": np.nan, "stderr": np.nan, "t_value": np.nan, "p_value": np.nan, "n": int(n_obs), "adj_r2": np.nan}

    xtx_inv = np.linalg.inv(xtx)
    beta = xtx_inv @ X.T @ y
    residual = y - (X @ beta)
    df_resid = n_obs - n_coef
    rss = float(residual.T @ residual)
    tss = float(((y - y.mean()) ** 2).sum())
    mse = rss / df_resid if df_resid > 0 else np.nan
    std_errors = np.sqrt(np.diag(xtx_inv) * mse) if np.isfinite(mse) else np.full(n_coef, np.nan)
    coef_index = list(frame.drop(columns=["response"]).columns).index(coefficient)
    beta_value = float(beta[coef_index])
    stderr_value = float(std_errors[coef_index]) if np.isfinite(std_errors[coef_index]) else np.nan
    if not np.isfinite(stderr_value) or stderr_value <= 0:
        t_value = np.nan
        p_value = np.nan
    else:
        t_value = float(beta_value / stderr_value)
        p_value = float(2.0 * stats.t.sf(abs(t_value), df_resid))
    r2 = np.nan if tss <= 0 else 1.0 - (rss / tss)
    adj_r2 = np.nan if not np.isfinite(r2) else 1.0 - (1.0 - r2) * ((n_obs - 1.0) / df_resid)
    return {
        "beta": beta_value,
        "stderr": stderr_value,
        "t_value": t_value,
        "p_value": p_value,
        "n": int(n_obs),
        "adj_r2": float(adj_r2) if np.isfinite(adj_r2) else np.nan,
    }


def _evaluate_feature_matrix(
    feature_matrix: pd.DataFrame,
    meta: pd.DataFrame,
    cfg: PurityValidationConfig,
    high_purity_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_meta = meta.copy()
    model_meta["group_binary"] = model_meta["is_tumor"].astype(float)
    model_meta["purity_model"] = model_meta["purity_numeric"]
    model_meta.loc[model_meta["is_normal"].eq(1), "purity_model"] = 0.0
    model_meta["high_purity_binary"] = (
        model_meta["is_tumor"].eq(1) & model_meta["purity_numeric"].ge(high_purity_threshold)
    ).astype(float)

    design_unadjusted = _build_design_matrix(model_meta, "group_binary", None)
    design_adjusted = _build_design_matrix(model_meta, "group_binary", cfg.batch_column)
    design_adjusted["purity_model"] = model_meta["purity_model"]
    adjusted_columns = [column for column in design_adjusted.columns if column != "purity_model"]
    design_adjusted = design_adjusted[adjusted_columns[:2] + ["purity_model"] + adjusted_columns[2:]]

    high_purity_mask = model_meta["is_normal"].eq(1) | model_meta["high_purity_binary"].eq(1)
    design_high_vs_normal = _build_design_matrix(
        model_meta.loc[high_purity_mask],
        "high_purity_binary",
        None,
    )

    high_low_mask = model_meta["is_tumor"].eq(1) & model_meta["purity_numeric"].notna()
    design_high_vs_low = _build_design_matrix(
        model_meta.loc[high_low_mask],
        "high_purity_binary",
        None,
    )

    unadjusted_rows: list[dict[str, Any]] = []
    adjusted_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for feature_name, values in feature_matrix.iterrows():
        unadjusted = _fit_linear_coefficient(
            values,
            design_unadjusted,
            "group_binary",
            cfg.min_group_size,
            "group_binary",
        )
        adjusted = _fit_linear_coefficient(
            values,
            design_adjusted,
            "group_binary",
            cfg.min_group_size,
            "group_binary",
        )
        high_vs_normal = _fit_linear_coefficient(
            values.loc[high_purity_mask],
            design_high_vs_normal,
            "high_purity_binary",
            cfg.min_group_size,
            "high_purity_binary",
        )
        high_vs_low = _fit_linear_coefficient(
            values.loc[high_low_mask],
            design_high_vs_low,
            "high_purity_binary",
            cfg.min_group_size,
            "high_purity_binary",
        )

        n_tumor_nonmissing = int(values.loc[model_meta["is_tumor"].eq(1)].notna().sum())
        n_normal_nonmissing = int(values.loc[model_meta["is_normal"].eq(1)].notna().sum())

        unadjusted_rows.append(
            {
                "protein": feature_name,
                "effect_size": unadjusted["beta"],
                "stderr": unadjusted["stderr"],
                "t_value": unadjusted["t_value"],
                "p_value": unadjusted["p_value"],
                "n_samples": unadjusted["n"],
                "adj_r2": unadjusted["adj_r2"],
                "n_tumor_nonmissing": n_tumor_nonmissing,
                "n_normal_nonmissing": n_normal_nonmissing,
            }
        )
        adjusted_rows.append(
            {
                "protein": feature_name,
                "effect_size": adjusted["beta"],
                "stderr": adjusted["stderr"],
                "t_value": adjusted["t_value"],
                "p_value": adjusted["p_value"],
                "n_samples": adjusted["n"],
                "adj_r2": adjusted["adj_r2"],
                "n_tumor_nonmissing": n_tumor_nonmissing,
                "n_normal_nonmissing": n_normal_nonmissing,
            }
        )

        percent_attenuation = np.nan
        if pd.notna(unadjusted["beta"]) and abs(unadjusted["beta"]) > 0:
            percent_attenuation = ((abs(unadjusted["beta"]) - abs(adjusted["beta"])) / abs(unadjusted["beta"])) * 100.0
        sign_changed = (
            pd.notna(unadjusted["beta"])
            and pd.notna(adjusted["beta"])
            and np.sign(unadjusted["beta"]) != np.sign(adjusted["beta"])
            and abs(adjusted["beta"]) > 0
        )
        comparison_rows.append(
            {
                "protein": feature_name,
                "unadjusted_effect_size": unadjusted["beta"],
                "unadjusted_p_value": unadjusted["p_value"],
                "adjusted_effect_size": adjusted["beta"],
                "adjusted_p_value": adjusted["p_value"],
                "high_purity_vs_normal_effect_size": high_vs_normal["beta"],
                "high_purity_vs_normal_p_value": high_vs_normal["p_value"],
                "high_vs_low_tumor_effect_size": high_vs_low["beta"],
                "high_vs_low_tumor_p_value": high_vs_low["p_value"],
                "percent_attenuation": percent_attenuation,
                "substantial_effect_change": bool(
                    pd.notna(percent_attenuation) and abs(percent_attenuation) >= cfg.substantial_effect_change_pct
                ) or bool(sign_changed),
                "effect_direction_changed": bool(sign_changed),
                "n_tumor_nonmissing": n_tumor_nonmissing,
                "n_normal_nonmissing": n_normal_nonmissing,
            }
        )

    diff_unadjusted = pd.DataFrame(unadjusted_rows)
    diff_unadjusted["fdr"] = _safe_fdrcorrect(diff_unadjusted["p_value"])
    diff_adjusted = pd.DataFrame(adjusted_rows)
    diff_adjusted["fdr"] = _safe_fdrcorrect(diff_adjusted["p_value"])
    comparison = pd.DataFrame(comparison_rows)
    comparison["unadjusted_fdr"] = _safe_fdrcorrect(comparison["unadjusted_p_value"])
    comparison["adjusted_fdr"] = _safe_fdrcorrect(comparison["adjusted_p_value"])
    comparison["high_purity_vs_normal_fdr"] = _safe_fdrcorrect(comparison["high_purity_vs_normal_p_value"])
    comparison["high_vs_low_tumor_fdr"] = _safe_fdrcorrect(comparison["high_vs_low_tumor_p_value"])
    comparison["significant_unadjusted"] = comparison["unadjusted_fdr"] < 0.05
    comparison["significant_adjusted"] = comparison["adjusted_fdr"] < 0.05
    comparison["significance_transition"] = np.select(
        [
            comparison["significant_unadjusted"] & comparison["significant_adjusted"],
            comparison["significant_unadjusted"] & ~comparison["significant_adjusted"],
            ~comparison["significant_unadjusted"] & comparison["significant_adjusted"],
        ],
        [
            "remain_significant_after_adjustment",
            "lose_significance_after_adjustment",
            "gain_significance_after_adjustment",
        ],
        default="not_significant_in_either_model",
    )
    return diff_unadjusted, diff_adjusted, comparison


def _compute_protein_correlations(
    matrix: pd.DataFrame,
    meta: pd.DataFrame,
    stromal_score: pd.Series,
    stromal_markers: list[str],
    tumor_markers: list[str],
) -> pd.DataFrame:
    tumor_mask = meta["is_tumor"].eq(1) & meta["purity_numeric"].notna()
    tumor_samples = meta.index[tumor_mask]
    purity = meta.loc[tumor_samples, "purity_numeric"]
    stromal = stromal_score.reindex(tumor_samples)
    rows: list[dict[str, Any]] = []
    stromal_marker_set = set(stromal_markers)
    tumor_marker_set = set(tumor_markers)

    for feature_name, values in matrix.iterrows():
        rho, p_value, n_corr = _safe_spearman(values.reindex(tumor_samples), purity)
        stromal_rho, stromal_p, _ = _safe_spearman(values.reindex(tumor_samples), stromal)
        rows.append(
            {
                "protein": feature_name,
                "purity_spearman_rho": rho,
                "purity_spearman_p_value": p_value,
                "n_tumor_with_purity": n_corr,
                "stromal_score_spearman_rho": stromal_rho,
                "stromal_score_spearman_p_value": stromal_p,
                "is_candidate_stromal_marker": feature_name in stromal_marker_set,
                "is_candidate_tumor_marker": feature_name in tumor_marker_set,
            }
        )
    result = pd.DataFrame(rows)
    result["purity_spearman_fdr"] = _safe_fdrcorrect(result["purity_spearman_p_value"])
    result["stromal_score_spearman_fdr"] = _safe_fdrcorrect(result["stromal_score_spearman_p_value"])
    return result.sort_values(
        ["is_candidate_stromal_marker", "is_candidate_tumor_marker", "purity_spearman_fdr", "protein"],
        ascending=[False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def _evaluate_signatures(
    score_table: pd.DataFrame,
    cfg: PurityValidationConfig,
    high_purity_threshold: float,
) -> pd.DataFrame:
    meta = score_table.set_index(score_table.columns[0]).copy()
    tumor_samples = meta.index[meta["is_tumor"].eq(1) & meta["purity_numeric"].notna()]
    stromal_score = meta["stromal_ecm_score"]
    purity = meta.loc[tumor_samples, "purity_numeric"]
    signature_names = [
        "stromal_ecm_score",
        "tumor_epithelial_score",
        "myogenesis_score",
        "coagulation_score",
        "emt_score",
    ]
    signature_matrix = meta[signature_names].T.copy()
    _, _, comparison = _evaluate_feature_matrix(signature_matrix, meta, cfg, high_purity_threshold)

    rows: list[dict[str, Any]] = []
    for signature_name in signature_names:
        values = meta[signature_name]
        rho, p_value, n_corr = _safe_spearman(values.loc[tumor_samples], purity)
        if signature_name == "stromal_ecm_score":
            stromal_rho = 1.0
            stromal_p = 0.0
        else:
            stromal_rho, stromal_p, _ = _safe_spearman(values.loc[tumor_samples], stromal_score.loc[tumor_samples])
        feature_row = comparison.loc[comparison["protein"] == signature_name].iloc[0]
        rows.append(
            {
                "signature": signature_name,
                "purity_spearman_rho": rho,
                "purity_spearman_p_value": p_value,
                "n_tumor_with_purity": n_corr,
                "stromal_score_spearman_rho": stromal_rho,
                "stromal_score_spearman_p_value": stromal_p,
                "unadjusted_effect_size": feature_row["unadjusted_effect_size"],
                "unadjusted_p_value": feature_row["unadjusted_p_value"],
                "adjusted_effect_size": feature_row["adjusted_effect_size"],
                "adjusted_p_value": feature_row["adjusted_p_value"],
                "high_purity_vs_normal_effect_size": feature_row["high_purity_vs_normal_effect_size"],
                "high_purity_vs_normal_p_value": feature_row["high_purity_vs_normal_p_value"],
                "high_vs_low_tumor_effect_size": feature_row["high_vs_low_tumor_effect_size"],
                "high_vs_low_tumor_p_value": feature_row["high_vs_low_tumor_p_value"],
                "percent_attenuation": feature_row["percent_attenuation"],
            }
        )
    result = pd.DataFrame(rows)
    result["purity_spearman_fdr"] = _safe_fdrcorrect(result["purity_spearman_p_value"])
    result["stromal_score_spearman_fdr"] = _safe_fdrcorrect(result["stromal_score_spearman_p_value"])
    result["unadjusted_fdr"] = _safe_fdrcorrect(result["unadjusted_p_value"])
    result["adjusted_fdr"] = _safe_fdrcorrect(result["adjusted_p_value"])
    result["high_purity_vs_normal_fdr"] = _safe_fdrcorrect(result["high_purity_vs_normal_p_value"])
    result["high_vs_low_tumor_fdr"] = _safe_fdrcorrect(result["high_vs_low_tumor_p_value"])

    classifications: list[str] = []
    for row in result.itertuples():
        purity_sensitive = (
            pd.notna(row.purity_spearman_rho)
            and row.purity_spearman_rho < 0
            and pd.notna(row.purity_spearman_fdr)
            and row.purity_spearman_fdr < 0.05
            and pd.notna(row.stromal_score_spearman_rho)
            and row.stromal_score_spearman_rho > 0
            and pd.notna(row.percent_attenuation)
            and row.percent_attenuation >= cfg.substantial_effect_change_pct
        )
        tumor_intrinsic = (
            pd.notna(row.adjusted_fdr)
            and row.adjusted_fdr < 0.05
            and pd.notna(row.high_purity_vs_normal_fdr)
            and row.high_purity_vs_normal_fdr < 0.05
        )
        if purity_sensitive:
            classifications.append("likely_purity_stromal_sensitive")
        elif tumor_intrinsic:
            classifications.append("likely_tumor_intrinsic")
        else:
            classifications.append("mixed_or_uncertain")
    result["classification"] = classifications
    return result


def _plot_regression_scatter(
    x: pd.Series,
    y: pd.Series,
    x_label: str,
    y_label: str,
    title: str,
    subtitle: str,
    output_path: Path,
) -> None:
    frame = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if frame.empty:
        return
    plt.figure(figsize=(5.2, 4.0))
    plt.scatter(frame.iloc[:, 0], frame.iloc[:, 1], s=25, alpha=0.85, color="#2563eb", edgecolor="none")
    if len(frame) >= 2 and frame.iloc[:, 0].nunique() >= 2:
        slope, intercept = np.polyfit(frame.iloc[:, 0], frame.iloc[:, 1], 1)
        xs = np.linspace(frame.iloc[:, 0].min(), frame.iloc[:, 0].max(), 100)
        plt.plot(xs, intercept + slope * xs, color="#dc2626", linewidth=1.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.gcf().text(0.13, 0.92, subtitle, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def _plot_signature_scatter_plots(
    score_table: pd.DataFrame,
    signature_results: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tumor_frame = score_table.loc[score_table["is_tumor"].eq(1)].copy()
    summary = signature_results.set_index("signature")
    for signature_name in summary.index:
        row = summary.loc[signature_name]
        subtitle = (
            f"rho={row['purity_spearman_rho']:.3f}, FDR={row['purity_spearman_fdr']:.3g}"
            if pd.notna(row["purity_spearman_rho"])
            else "rho=NA"
        )
        _plot_regression_scatter(
            tumor_frame["purity_numeric"],
            tumor_frame[signature_name],
            "Tumor purity",
            signature_name.replace("_", " "),
            signature_name.replace("_", " ").title(),
            subtitle,
            output_dir / f"{signature_name}_vs_purity.png",
        )


def _plot_protein_scatter_plots(
    matrix: pd.DataFrame,
    meta: pd.DataFrame,
    protein_results: pd.DataFrame,
    proteins: list[str],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tumor_mask = meta["is_tumor"].eq(1) & meta["purity_numeric"].notna()
    tumor_samples = meta.index[tumor_mask]
    lookup = protein_results.set_index("protein")
    for protein in proteins:
        if protein not in matrix.index or protein not in lookup.index:
            continue
        row = lookup.loc[protein]
        subtitle = (
            f"rho={row['purity_spearman_rho']:.3f}, FDR={row['purity_spearman_fdr']:.3g}"
            if pd.notna(row["purity_spearman_rho"])
            else "rho=NA"
        )
        _plot_regression_scatter(
            meta.loc[tumor_samples, "purity_numeric"],
            matrix.loc[protein, tumor_samples].T,
            "Tumor purity",
            protein,
            protein,
            subtitle,
            output_dir / f"{protein}_vs_purity.png",
        )


def _plot_candidate_beta_comparison(comparison: pd.DataFrame, output_path: Path) -> None:
    candidate_frame = comparison.loc[
        comparison["is_selected_candidate"].fillna(False)
    ].copy()
    candidate_frame = candidate_frame.dropna(subset=["unadjusted_effect_size", "adjusted_effect_size"])
    if candidate_frame.empty:
        return
    candidate_frame = candidate_frame.sort_values("unadjusted_effect_size")
    y = np.arange(len(candidate_frame))
    plt.figure(figsize=(7.2, max(4.0, 0.4 * len(candidate_frame))))
    for idx, row in enumerate(candidate_frame.itertuples()):
        plt.plot(
            [row.adjusted_effect_size, row.unadjusted_effect_size],
            [idx, idx],
            color="#cbd5e1",
            linewidth=1.5,
        )
    plt.scatter(candidate_frame["unadjusted_effect_size"], y, color="#2563eb", s=40, label="Unadjusted")
    plt.scatter(candidate_frame["adjusted_effect_size"], y, color="#dc2626", s=40, label="Purity-adjusted")
    plt.axvline(0.0, color="#94a3b8", linestyle="--", linewidth=1.0)
    plt.yticks(y, candidate_frame["protein"])
    plt.xlabel("Tumor vs NAT effect size")
    plt.ylabel("")
    plt.legend(frameon=False, loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def _plot_signature_group_summary(
    score_table: pd.DataFrame,
    high_purity_threshold: float,
    output_path: Path,
) -> None:
    plot_frame = score_table.copy()
    plot_frame["purity_group"] = np.where(
        plot_frame["is_normal"].eq(1),
        "NAT",
        np.where(
            plot_frame["purity_numeric"].ge(high_purity_threshold),
            "High-purity tumor",
            "Low-purity tumor",
        ),
    )
    signature_names = [
        "stromal_ecm_score",
        "tumor_epithelial_score",
        "myogenesis_score",
        "coagulation_score",
        "emt_score",
    ]
    fig, axes = plt.subplots(len(signature_names), 1, figsize=(7.0, 2.6 * len(signature_names)), sharex=True)
    for ax, signature_name in zip(np.atleast_1d(axes), signature_names):
        grouped = [
            pd.to_numeric(plot_frame.loc[plot_frame["purity_group"] == label, signature_name], errors="coerce").dropna().to_numpy()
            for label in ["NAT", "Low-purity tumor", "High-purity tumor"]
        ]
        ax.boxplot(grouped, tick_labels=["NAT", "Low", "High"], patch_artist=True)
        ax.set_ylabel(signature_name.replace("_score", "").replace("_", "\n"))
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close(fig)


def _normalize_gsea_results(results: pd.DataFrame) -> pd.DataFrame:
    normalized = results.copy().reset_index(drop=True)
    for column in ["ES", "NES", "NOM p-val", "FDR q-val", "FWER p-val"]:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    lead_gene_column = next((column for column in ["Lead_genes", "ledge_genes"] if column in normalized.columns), None)
    if lead_gene_column and lead_gene_column != "Lead_genes":
        normalized = normalized.rename(columns={lead_gene_column: "Lead_genes"})
    if "Lead_genes" not in normalized.columns:
        normalized["Lead_genes"] = ""
    return normalized


def _run_prerank_gsea(
    ranking: pd.Series,
    gene_sets: Path,
    output_dir: Path,
    permutation_num: int,
    min_size: int,
    max_size: int,
    seed: int,
) -> pd.DataFrame:
    import gseapy

    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = ranking.dropna().sort_values(ascending=False).reset_index()
    ranked.columns = ["gene", "ranking_metric"]
    ranked.to_csv(output_dir / "ranked_gene_list.tsv", sep="\t", index=False)
    prerank_result = gseapy.prerank(
        rnk=ranked,
        gene_sets=str(gene_sets),
        permutation_num=permutation_num,
        min_size=min_size,
        max_size=max_size,
        seed=seed,
        outdir=None,
        verbose=True,
    )
    results = _normalize_gsea_results(prerank_result.res2d)
    results.to_csv(output_dir / "gsea_full_results.tsv", sep="\t", index=False)
    return results


def _build_gsea_focus_table(
    unadjusted_hallmark: pd.DataFrame | None,
    adjusted_hallmark: pd.DataFrame | None,
    unadjusted_kegg: pd.DataFrame | None,
    adjusted_kegg: pd.DataFrame | None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for collection, focus_terms, unadjusted, adjusted in [
        ("hallmark", FOCUS_HALLMARK_TERMS, unadjusted_hallmark, adjusted_hallmark),
        ("kegg", FOCUS_KEGG_TERMS, unadjusted_kegg, adjusted_kegg),
    ]:
        for term in focus_terms:
            un_row = None
            adj_row = None
            if unadjusted is not None and "Term" in unadjusted.columns:
                matches = unadjusted.loc[unadjusted["Term"].map(lambda value: _contains_case_insensitive(value, [term]))]
                if not matches.empty:
                    un_row = matches.iloc[0]
            if adjusted is not None and "Term" in adjusted.columns:
                matches = adjusted.loc[adjusted["Term"].map(lambda value: _contains_case_insensitive(value, [term]))]
                if not matches.empty:
                    adj_row = matches.iloc[0]
            rows.append(
                {
                    "collection": collection,
                    "focus_term": term,
                    "unadjusted_term": un_row["Term"] if un_row is not None else np.nan,
                    "unadjusted_nes": un_row["NES"] if un_row is not None else np.nan,
                    "unadjusted_fdr": un_row["FDR q-val"] if un_row is not None else np.nan,
                    "adjusted_term": adj_row["Term"] if adj_row is not None else np.nan,
                    "adjusted_nes": adj_row["NES"] if adj_row is not None else np.nan,
                    "adjusted_fdr": adj_row["FDR q-val"] if adj_row is not None else np.nan,
                }
            )
    table = pd.DataFrame(rows)
    table["retained_after_adjustment"] = table["adjusted_fdr"] < 0.25
    table["attenuated_after_adjustment"] = (
        table["unadjusted_fdr"].lt(0.25)
        & table["adjusted_fdr"].ge(0.25)
    ) | (
        table["unadjusted_nes"].abs().sub(table["adjusted_nes"].abs()).fillna(0.0) > 0.3
    )
    return table


def _load_custom_candidate_proteins(cfg: PurityValidationConfig) -> list[str]:
    proteins = [gene.upper() for gene in (cfg.candidate_protein_list or []) if gene]
    if cfg.candidate_protein_path is None:
        return _unique(proteins)
    table = _read_table_auto(cfg.candidate_protein_path)
    if cfg.candidate_protein_gene_column not in table.columns:
        msg = (
            f"Candidate protein gene column '{cfg.candidate_protein_gene_column}' was not found in "
            f"{cfg.candidate_protein_path}."
        )
        raise ValueError(msg)
    proteins.extend(
        table[cfg.candidate_protein_gene_column].dropna().astype(str).str.strip().str.upper().tolist()
    )
    return _unique(proteins)


def _build_candidate_selection_table(
    cfg: PurityValidationConfig,
    matrix: pd.DataFrame,
    original_protein_loss: pd.DataFrame,
    original_pathway_loss: pd.DataFrame,
) -> pd.DataFrame:
    available = set(matrix.index.astype(str))
    protein_sig_mask = original_protein_loss["original_significant"].astype("boolean").fillna(False)

    selected_rows: list[dict[str, Any]] = []
    mode = cfg.candidate_protein_mode.strip().casefold()

    if mode == "preset_markers":
        proteins = _unique((cfg.stromal_markers or []) + (cfg.tumor_markers or []))
        for protein in proteins:
            selected_rows.append(
                {
                    "protein": protein,
                    "candidate_mode": "preset_markers",
                    "candidate_reason": "preset_stromal_or_tumor_marker",
                    "present_in_matrix": protein in available,
                }
            )
    elif mode == "original_significant":
        for protein in original_protein_loss.loc[protein_sig_mask, "protein"].astype(str).tolist():
            selected_rows.append(
                {
                    "protein": protein,
                    "candidate_mode": "original_significant",
                    "candidate_reason": "original_significant_protein",
                    "present_in_matrix": protein in available,
                }
            )
    elif mode == "original_significant_leading_edge":
        leading_edge_genes: set[str] = set()
        for lead_genes in original_pathway_loss.loc[
            original_pathway_loss["original_fdr"].lt(cfg.pathway_significance_fdr), "lead_genes"
        ]:
            leading_edge_genes.update(_split_gene_field(lead_genes))
        selected = original_protein_loss.loc[
            protein_sig_mask & original_protein_loss["protein"].isin(leading_edge_genes), "protein"
        ].astype(str).tolist()
        for protein in selected:
            selected_rows.append(
                {
                    "protein": protein,
                    "candidate_mode": "original_significant_leading_edge",
                    "candidate_reason": "original_significant_and_significant_pathway_leading_edge",
                    "present_in_matrix": protein in available,
                }
            )
    elif mode == "custom_list":
        for protein in _load_custom_candidate_proteins(cfg):
            selected_rows.append(
                {
                    "protein": protein,
                    "candidate_mode": "custom_list",
                    "candidate_reason": "custom_candidate_protein",
                    "present_in_matrix": protein in available,
                }
            )
    else:
        msg = (
            "candidate_protein_mode must be one of: preset_markers, original_significant, "
            "original_significant_leading_edge, custom_list."
        )
        raise ValueError(msg)

    result = pd.DataFrame(selected_rows)
    if result.empty:
        return pd.DataFrame(columns=["protein", "candidate_mode", "candidate_reason", "present_in_matrix"])
    result["protein"] = result["protein"].astype(str).str.strip().str.upper()
    result = result.drop_duplicates(subset=["protein"], keep="first").reset_index(drop=True)
    return result


def _find_first_column(columns: pd.Index, candidates: list[str]) -> str | None:
    lookup = {str(column).casefold(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.casefold() in lookup:
            return lookup[candidate.casefold()]
    return None


def _build_original_protein_loss_table(
    original_results_path: Path | None,
    adjusted_comparison: pd.DataFrame,
) -> pd.DataFrame:
    if original_results_path is None or not original_results_path.exists():
        return pd.DataFrame()

    original = _read_table_auto(original_results_path).copy()
    gene_column = _find_first_column(original.columns, ["Gene", "gene", "protein", "Protein"])
    fdr_column = _find_first_column(original.columns, ["FDR", "fdr", "adj_pval", "padj"])
    significance_column = _find_first_column(original.columns, ["Significance", "significance"])
    effect_column = _find_first_column(original.columns, ["Log2FC(mean)", "Log2FC(median)", "effect_size"])
    if gene_column is None or (fdr_column is None and significance_column is None):
        return pd.DataFrame()

    original["protein"] = original[gene_column].astype(str).str.strip().str.upper()
    significance_labels = (
        original[significance_column].astype(str).str.strip().str.upper()
        if significance_column is not None
        else pd.Series(np.nan, index=original.index, dtype=object)
    )
    if fdr_column is not None:
        original["original_fdr"] = pd.to_numeric(original[fdr_column], errors="coerce")
        if significance_column is not None:
            original["original_significant"] = significance_labels.isin(["S-U", "S-D"])
        else:
            original["original_significant"] = original["original_fdr"] < 0.05
    else:
        original["original_fdr"] = np.nan
        original["original_significant"] = significance_labels.isin(["S-U", "S-D"])
    original["original_significance_label"] = significance_labels
    original["original_tumor_up"] = significance_labels.eq("S-U")
    original["original_nat_up"] = significance_labels.eq("S-D")
    original["original_effect_size"] = (
        pd.to_numeric(original[effect_column], errors="coerce") if effect_column is not None else np.nan
    )

    merged = adjusted_comparison.merge(
        original[
            [
                "protein",
                "original_fdr",
                "original_significant",
                "original_significance_label",
                "original_tumor_up",
                "original_nat_up",
                "original_effect_size",
            ]
        ],
        on="protein",
        how="left",
    )
    original_sig = merged["original_significant"].astype("boolean").fillna(False)
    merged["lost_after_purity_adjustment_vs_original"] = original_sig & merged["adjusted_fdr"].ge(0.05)
    return merged.sort_values(
        ["lost_after_purity_adjustment_vs_original", "adjusted_fdr", "original_fdr", "protein"],
        ascending=[False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def _build_original_pathway_loss_table(
    original_results_path: Path | None,
    adjusted_results: list[tuple[str, pd.DataFrame | None]],
    pathway_significance_fdr: float,
) -> pd.DataFrame:
    if original_results_path is None or not original_results_path.exists():
        return pd.DataFrame()

    original = _read_table_auto(original_results_path).copy()
    term_column = _find_first_column(original.columns, ["Term", "term", "Name"])
    fdr_column = _find_first_column(original.columns, ["FDR q-val", "fdr_q_val", "FDR", "fdr"])
    nes_column = _find_first_column(original.columns, ["NES", "nes"])
    lead_column = _find_first_column(original.columns, ["Lead_genes", "Genes", "genes"])
    if term_column is None or fdr_column is None:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    original["Term"] = original[term_column].astype(str).str.strip()
    original["original_fdr"] = pd.to_numeric(original[fdr_column], errors="coerce")
    original["original_nes"] = pd.to_numeric(original[nes_column], errors="coerce") if nes_column is not None else np.nan
    original["lead_genes"] = original[lead_column] if lead_column is not None else np.nan
    original["original_significant"] = original["original_fdr"] < pathway_significance_fdr

    adjusted_lookup: dict[str, tuple[str, pd.Series]] = {}
    for collection, frame in adjusted_results:
        if frame is None or frame.empty or "Term" not in frame.columns:
            continue
        local = frame.copy()
        local["Term"] = local["Term"].astype(str).str.strip()
        local["normalized_term"] = local["Term"].map(_normalize_pathway_term)
        for _, row in local.iterrows():
            adjusted_lookup[str(row["normalized_term"])] = (collection, row)

    for row in original.itertuples():
        collection, adjusted_row = adjusted_lookup.get(_normalize_pathway_term(row.Term), ("unknown", None))
        adjusted_fdr = np.nan
        adjusted_nes = np.nan
        if adjusted_row is not None:
            adjusted_fdr = pd.to_numeric(adjusted_row.get("FDR q-val"), errors="coerce")
            adjusted_nes = pd.to_numeric(adjusted_row.get("NES"), errors="coerce")
        rows.append(
            {
                "collection": collection,
                "pathway": row.Term,
                "original_fdr": row.original_fdr,
                "original_nes": row.original_nes,
                "original_significant": bool(row.original_significant),
                "adjusted_fdr": adjusted_fdr,
                "adjusted_nes": adjusted_nes,
                "lost_after_purity_adjustment_vs_original": bool(row.original_significant) and pd.notna(adjusted_fdr) and adjusted_fdr >= pathway_significance_fdr,
                "missing_in_adjusted_rerun": pd.isna(adjusted_fdr),
                "lead_genes": row.lead_genes,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["lost_after_purity_adjustment_vs_original", "adjusted_fdr", "original_fdr", "pathway"],
        ascending=[False, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def _fmt(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _fmt_g(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.3g}"


def _write_report(
    cfg: PurityValidationConfig,
    summary: dict[str, Any],
    marker_availability: pd.DataFrame,
    signature_results: pd.DataFrame,
    comparison: pd.DataFrame,
    high_purity_threshold: float,
    gsea_focus: pd.DataFrame,
    original_protein_loss: pd.DataFrame,
    original_pathway_loss: pd.DataFrame,
) -> Path:
    report_path = cfg.output_dir / "purity_validation_summary.md"

    missing_markers = marker_availability.loc[~marker_availability["present_in_matrix"], "gene"].tolist()
    signature_lookup = signature_results.set_index("signature")
    candidate_lookup = comparison.loc[
        comparison["is_selected_candidate"].fillna(False)
    ].copy()
    purity_sensitive_proteins = candidate_lookup.loc[
        (candidate_lookup["purity_spearman_rho"] < 0)
        & (candidate_lookup["purity_spearman_fdr"] < 0.05)
        & (candidate_lookup["stromal_score_spearman_rho"] > 0)
        & (
            (candidate_lookup["significance_transition"] == "lose_significance_after_adjustment")
            | candidate_lookup["substantial_effect_change"]
        )
    ].sort_values(["purity_spearman_fdr", "percent_attenuation"], na_position="last")
    tumor_intrinsic_proteins = candidate_lookup.loc[
        (candidate_lookup["adjusted_fdr"] < 0.05)
        & (candidate_lookup["high_purity_vs_normal_fdr"] < 0.05)
    ].sort_values(["adjusted_fdr", "high_purity_vs_normal_fdr"], na_position="last")

    purity_sensitive_signatures = signature_results.loc[
        signature_results["classification"] == "likely_purity_stromal_sensitive"
    ]
    tumor_intrinsic_signatures = signature_results.loc[
        signature_results["classification"] == "likely_tumor_intrinsic"
    ]

    lines = [
        "# Purity Validation Summary",
        "",
        "## Inputs and retained data",
        f"- Matrix: `{cfg.matrix_path}`",
        f"- Metadata: `{cfg.metadata_path}`",
        f"- Orientation detected: `{summary['orientation']}`",
        f"- Sample ID column used: `{summary['sample_id_column']}`",
        f"- Retained proteins: {summary['n_retained_proteins']}",
        f"- Retained samples: {summary['n_retained_samples']}",
        f"- Retained tumor samples: {summary['n_retained_tumor']}",
        f"- Retained NAT/normal samples: {summary['n_retained_normal']}",
        f"- Tumor samples with numeric purity: {summary['n_tumor_with_purity']}",
        f"- Tumor filter: `{summary['tumor_filter_column'] or 'none'}` = `{summary['tumor_filter_value'] or 'none'}`",
        f"- Candidate protein mode: `{cfg.candidate_protein_mode}`",
        f"- High-purity threshold: {_fmt(high_purity_threshold)} (tumor purity median)",
        "",
        "## Marker availability",
        f"- Missing requested markers: {', '.join(missing_markers) if missing_markers else 'none'}",
        "",
        "## Signature-level purity associations",
    ]
    for signature_name in [
        "stromal_ecm_score",
        "tumor_epithelial_score",
        "myogenesis_score",
        "coagulation_score",
        "emt_score",
    ]:
        row = signature_lookup.loc[signature_name]
        lines.append(
            "- "
            + f"{signature_name}: rho={_fmt(row['purity_spearman_rho'])}, purity FDR={_fmt_g(row['purity_spearman_fdr'])}, "
            + f"stromal-score rho={_fmt(row['stromal_score_spearman_rho'])}, attenuation={_fmt(row['percent_attenuation'], 1)}%"
        )

    lines.extend(
        [
            "",
            "## Differential summary",
            f"- Proteins significant without purity adjustment (FDR < 0.05): {(comparison['unadjusted_fdr'] < 0.05).sum()}",
            f"- Proteins significant after purity adjustment (FDR < 0.05): {(comparison['adjusted_fdr'] < 0.05).sum()}",
            f"- Proteins that lost significance after purity adjustment: {(comparison['significance_transition'] == 'lose_significance_after_adjustment').sum()}",
            f"- Originally significant proteins that became non-significant after purity adjustment: {int(original_protein_loss['lost_after_purity_adjustment_vs_original'].sum()) if not original_protein_loss.empty else 0}",
            f"- Originally significant pathways that became non-significant after purity adjustment: {int(original_pathway_loss['lost_after_purity_adjustment_vs_original'].sum()) if not original_pathway_loss.empty else 0}",
            "",
            "## Originally significant proteins that became non-significant after purity adjustment",
        ]
    )
    if original_protein_loss.empty:
        lines.append("- Original protein result table was not available.")
    else:
        for row in original_protein_loss.loc[
            original_protein_loss["lost_after_purity_adjustment_vs_original"]
        ].head(15).itertuples():
            lines.append(
                "- "
                + f"Protein `{row.protein}`: original FDR={_fmt_g(row.original_fdr)}, adjusted FDR={_fmt_g(row.adjusted_fdr)}, "
                + f"original effect={_fmt(row.original_effect_size)}, adjusted effect={_fmt(row.adjusted_effect_size)}"
            )

    lines.extend(
        [
            "",
            "## Originally significant pathways that became non-significant after purity adjustment",
        ]
    )
    if original_pathway_loss.empty:
        lines.append("- Original pathway result table was not available.")
    else:
        for row in original_pathway_loss.loc[
            original_pathway_loss["lost_after_purity_adjustment_vs_original"]
        ].head(15).itertuples():
            lines.append(
                "- "
                + f"Pathway `{row.pathway}`: original FDR={_fmt_g(row.original_fdr)}, adjusted FDR={_fmt_g(row.adjusted_fdr)}, "
                + f"original NES={_fmt(row.original_nes)}, adjusted NES={_fmt(row.adjusted_nes)}"
            )

    lines.extend(
        [
            "",
            "## GSEA focus pathways after purity adjustment",
        ]
    )
    if gsea_focus.empty:
        lines.append("- GSEA rerun was not available.")
    else:
        for row in gsea_focus.itertuples():
            lines.append(
                "- "
                + f"{row.collection}/{row.focus_term}: "
                + f"unadjusted NES={_fmt(row.unadjusted_nes)}, unadjusted FDR={_fmt_g(row.unadjusted_fdr)}, "
                + f"adjusted NES={_fmt(row.adjusted_nes)}, adjusted FDR={_fmt_g(row.adjusted_fdr)}"
            )

    lines.extend(["", "## Final interpretation", "", "### A. Likely purity/stromal-sensitive signals"])
    if purity_sensitive_signatures.empty and purity_sensitive_proteins.empty:
        lines.append("- No signature or candidate protein met the full purity/stromal-sensitive rule set.")
    else:
        for row in purity_sensitive_signatures.itertuples():
            lines.append(
                "- "
                + f"Signature `{row.signature}`: negative purity correlation, positive stromal-score correlation, and attenuation after purity adjustment."
            )
        for row in purity_sensitive_proteins.head(12).itertuples():
            lines.append(
                "- "
                + f"Protein `{row.protein}`: rho={_fmt(row.purity_spearman_rho)}, "
                + f"stromal-score rho={_fmt(row.stromal_score_spearman_rho)}, "
                + f"attenuation={_fmt(row.percent_attenuation, 1)}%."
            )

    lines.extend(["", "### B. Likely tumor-intrinsic signals"])
    if tumor_intrinsic_signatures.empty and tumor_intrinsic_proteins.empty:
        lines.append("- No signature or candidate protein met the strict tumor-intrinsic rule set.")
    else:
        for row in tumor_intrinsic_signatures.itertuples():
            lines.append(
                "- "
                + f"Signature `{row.signature}`: retained significance after purity adjustment and in the high-purity tumor vs NAT comparison."
            )
        for row in tumor_intrinsic_proteins.head(12).itertuples():
            lines.append(
                "- "
                + f"Protein `{row.protein}`: adjusted FDR={_fmt_g(row.adjusted_fdr)}, "
                + f"high-purity vs NAT FDR={_fmt_g(row.high_purity_vs_normal_fdr)}."
            )

    lines.extend(
        [
            "",
            "## Interpretation note",
            "- Purity-sensitive signals were treated as stromal-admixture or microenvironment-linked candidates, not automatically dismissed as technical contamination.",
            "- Signals that remained in high-purity tumors after purity adjustment were prioritized as more compatible with tumor-intrinsic or stable tumor-associated biology.",
            "",
            "## Main output files",
            "- `cleaned_metadata.csv`",
            "- `sample_signature_scores.csv`",
            "- `protein_purity_correlations.csv`",
            "- `signature_purity_correlations.csv`",
            "- `diff_unadjusted.csv`",
            "- `diff_purity_adjusted.csv`",
            "- `diff_comparison_adjusted_vs_unadjusted.csv`",
            "- `proteins_original_sig_lost_after_purity_adjustment.csv`",
            "- `pathways_original_sig_lost_after_purity_adjustment.csv`",
            "- `plots/`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _write_config_snapshot(cfg: PurityValidationConfig) -> None:
    lines = [
        "[task]",
        'name = "purity_validation"',
        "",
        "[input]",
    ]
    if cfg.base_diff_config is not None:
        lines.append(f'base_diff_config = "{cfg.base_diff_config}"')
        lines.extend(
        [
            f'matrix_path = "{cfg.matrix_path}"',
            f'metadata_path = "{cfg.metadata_path}"',
            f'hallmark_terms_path = "{cfg.hallmark_terms_path}"' if cfg.hallmark_terms_path else "hallmark_terms_path = ",
            f'kegg_gsea_results_path = "{cfg.kegg_gsea_results_path}"' if cfg.kegg_gsea_results_path else "kegg_gsea_results_path = ",
            f'original_protein_results_path = "{cfg.original_protein_results_path}"' if cfg.original_protein_results_path else "original_protein_results_path = ",
            f'original_pathway_results_path = "{cfg.original_pathway_results_path}"' if cfg.original_pathway_results_path else "original_pathway_results_path = ",
            f'hallmark_gmt_path = "{cfg.hallmark_gmt_path}"' if cfg.hallmark_gmt_path else "hallmark_gmt_path = ",
            f'kegg_gmt_path = "{cfg.kegg_gmt_path}"' if cfg.kegg_gmt_path else "kegg_gmt_path = ",
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
            f'candidate_protein_mode = "{cfg.candidate_protein_mode}"',
            f'candidate_protein_path = "{cfg.candidate_protein_path}"' if cfg.candidate_protein_path else "candidate_protein_path = ",
            f'candidate_protein_gene_column = "{cfg.candidate_protein_gene_column}"',
            f'candidate_protein_list = "{",".join(cfg.candidate_protein_list or [])}"',
            f"high_purity_quantile = {cfg.high_purity_quantile}",
            f"min_group_size = {cfg.min_group_size}",
            f"gsea_permutation_num = {cfg.gsea_permutation_num}",
            f"gsea_min_size = {cfg.gsea_min_size}",
            f"gsea_max_size = {cfg.gsea_max_size}",
            f"gsea_seed = {cfg.gsea_seed}",
            f"substantial_effect_change_pct = {cfg.substantial_effect_change_pct}",
            f"protein_significance_fdr = {cfg.protein_significance_fdr}",
            f"protein_effect_size_cutoff = {cfg.protein_effect_size_cutoff}",
            f"pathway_significance_fdr = {cfg.pathway_significance_fdr}",
            f'stromal_markers = "{",".join(cfg.stromal_markers or [])}"',
            f'tumor_markers = "{",".join(cfg.tumor_markers or [])}"',
        ]
    )
    (cfg.output_dir / "config.ini").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_purity_validation(config_path: Path) -> Path:
    cfg = _load_config(config_path)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[purity_validation] loading matrix: {cfg.matrix_path}")
    print(f"[purity_validation] loading metadata: {cfg.metadata_path}")
    matrix, meta, summary = _prepare_matrix_and_metadata(cfg)
    print(
        "[purity_validation] aligned "
        f"{summary['n_retained_samples']} samples and {summary['n_retained_proteins']} proteins "
        f"using sample column '{summary['sample_id_column']}'"
    )

    marker_availability = _build_marker_availability(
        matrix,
        cfg.stromal_markers or [],
        cfg.tumor_markers or [],
    )
    signatures, signature_sources = _build_signature_definitions(cfg)
    score_table, signature_definitions = _score_signatures(
        matrix,
        meta,
        summary["sample_id_column"],
        signatures,
    )

    tumor_purity = meta.loc[meta["is_tumor"].eq(1), "purity_numeric"].dropna()
    if tumor_purity.empty:
        msg = "No tumor purity values were available after alignment."
        raise ValueError(msg)
    high_purity_threshold = float(tumor_purity.quantile(cfg.high_purity_quantile))
    score_table["high_purity_threshold"] = high_purity_threshold
    score_table["high_purity_group"] = (
        score_table["is_tumor"].eq(1) & score_table["purity_numeric"].ge(high_purity_threshold)
    ).astype(int)

    print("[purity_validation] computing protein-level purity correlations")
    protein_correlations = _compute_protein_correlations(
        matrix,
        meta,
        score_table.set_index(summary["sample_id_column"])["stromal_ecm_score"],
        cfg.stromal_markers or [],
        cfg.tumor_markers or [],
    )

    print("[purity_validation] fitting unadjusted and purity-adjusted protein models")
    diff_unadjusted, diff_adjusted, comparison = _evaluate_feature_matrix(
        matrix,
        meta,
        cfg,
        high_purity_threshold,
    )
    comparison = comparison.merge(
        protein_correlations,
        on="protein",
        how="left",
    )
    comparison["is_candidate_stromal_marker"] = comparison["protein"].isin(cfg.stromal_markers or [])
    comparison["is_candidate_tumor_marker"] = comparison["protein"].isin(cfg.tumor_markers or [])

    signature_results = _evaluate_signatures(score_table, cfg, high_purity_threshold)

    plots_dir = cfg.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _plot_signature_scatter_plots(score_table, signature_results, plots_dir / "signature_vs_purity")
    _plot_protein_scatter_plots(
        matrix,
        meta,
        protein_correlations,
        ["POSTN", "COMP", "THBS4", "SFRP4"],
        plots_dir / "protein_vs_purity",
    )
    _plot_signature_group_summary(score_table, high_purity_threshold, plots_dir / "signature_by_purity_group.png")

    hallmark_unadjusted = None
    hallmark_adjusted = None
    kegg_unadjusted = None
    kegg_adjusted = None
    if cfg.hallmark_gmt_path and cfg.hallmark_gmt_path.exists():
        print(f"[purity_validation] running hallmark preranked GSEA from {cfg.hallmark_gmt_path}")
        ranking_unadjusted = diff_unadjusted.set_index("protein")["t_value"]
        ranking_adjusted = diff_adjusted.set_index("protein")["t_value"]
        hallmark_unadjusted = _run_prerank_gsea(
            ranking_unadjusted,
            cfg.hallmark_gmt_path,
            cfg.output_dir / "gsea_hallmark_unadjusted",
            cfg.gsea_permutation_num,
            cfg.gsea_min_size,
            cfg.gsea_max_size,
            cfg.gsea_seed,
        )
        hallmark_adjusted = _run_prerank_gsea(
            ranking_adjusted,
            cfg.hallmark_gmt_path,
            cfg.output_dir / "gsea_hallmark_adjusted",
            cfg.gsea_permutation_num,
            cfg.gsea_min_size,
            cfg.gsea_max_size,
            cfg.gsea_seed,
        )

    if cfg.kegg_gmt_path and cfg.kegg_gmt_path.exists():
        print(f"[purity_validation] running KEGG preranked GSEA from {cfg.kegg_gmt_path}")
        ranking_unadjusted = diff_unadjusted.set_index("protein")["t_value"]
        ranking_adjusted = diff_adjusted.set_index("protein")["t_value"]
        kegg_unadjusted = _run_prerank_gsea(
            ranking_unadjusted,
            cfg.kegg_gmt_path,
            cfg.output_dir / "gsea_kegg_unadjusted",
            cfg.gsea_permutation_num,
            cfg.gsea_min_size,
            cfg.gsea_max_size,
            cfg.gsea_seed,
        )
        kegg_adjusted = _run_prerank_gsea(
            ranking_adjusted,
            cfg.kegg_gmt_path,
            cfg.output_dir / "gsea_kegg_adjusted",
            cfg.gsea_permutation_num,
            cfg.gsea_min_size,
            cfg.gsea_max_size,
            cfg.gsea_seed,
        )

    gsea_focus = _build_gsea_focus_table(
        hallmark_unadjusted,
        hallmark_adjusted,
        kegg_unadjusted,
        kegg_adjusted,
    )
    original_protein_loss = _build_original_protein_loss_table(cfg.original_protein_results_path, comparison)
    original_pathway_loss = _build_original_pathway_loss_table(
        cfg.original_pathway_results_path,
        [
            ("hallmark", hallmark_adjusted),
            ("kegg", kegg_adjusted),
        ],
        cfg.pathway_significance_fdr,
    )
    candidate_selection = _build_candidate_selection_table(cfg, matrix, original_protein_loss, original_pathway_loss)
    comparison = comparison.merge(
        candidate_selection[["protein", "candidate_mode", "candidate_reason", "present_in_matrix"]],
        on="protein",
        how="left",
    )
    comparison["is_selected_candidate"] = comparison["candidate_mode"].notna()
    original_protein_loss = original_protein_loss.merge(
        candidate_selection[["protein", "candidate_mode", "candidate_reason", "present_in_matrix"]],
        on="protein",
        how="left",
    )
    original_protein_loss["is_selected_candidate"] = original_protein_loss["candidate_mode"].notna()
    _plot_candidate_beta_comparison(comparison, plots_dir / "candidate_beta_before_after_adjustment.png")

    print("[purity_validation] writing output tables")
    cleaned_metadata = meta.copy()
    cleaned_metadata.insert(0, summary["sample_id_column"], cleaned_metadata.index)
    cleaned_metadata["high_purity_threshold"] = high_purity_threshold
    cleaned_metadata["high_purity_group"] = (
        cleaned_metadata["is_tumor"].eq(1) & cleaned_metadata["purity_numeric"].ge(high_purity_threshold)
    ).astype(int)
    cleaned_metadata.to_csv(cfg.output_dir / "cleaned_metadata.csv", index=False)
    marker_availability.to_csv(cfg.output_dir / "marker_availability.csv", index=False)
    signature_sources.to_csv(cfg.output_dir / "signature_sources.csv", index=False)
    signature_definitions.to_csv(cfg.output_dir / "signature_gene_sets.csv", index=False)
    score_table.to_csv(cfg.output_dir / "sample_signature_scores.csv", index=False)
    protein_correlations.to_csv(cfg.output_dir / "protein_purity_correlations.csv", index=False)
    signature_results.to_csv(cfg.output_dir / "signature_purity_correlations.csv", index=False)
    diff_unadjusted.to_csv(cfg.output_dir / "diff_unadjusted.csv", index=False)
    diff_adjusted.to_csv(cfg.output_dir / "diff_purity_adjusted.csv", index=False)
    comparison.to_csv(cfg.output_dir / "diff_comparison_adjusted_vs_unadjusted.csv", index=False)
    candidate_selection.to_csv(cfg.output_dir / "candidate_protein_selection.csv", index=False)
    comparison[
        [
            "protein",
            "high_purity_vs_normal_effect_size",
            "high_purity_vs_normal_p_value",
            "high_purity_vs_normal_fdr",
        ]
    ].to_csv(cfg.output_dir / "diff_high_purity_vs_normal.csv", index=False)
    comparison[
        [
            "protein",
            "high_vs_low_tumor_effect_size",
            "high_vs_low_tumor_p_value",
            "high_vs_low_tumor_fdr",
        ]
    ].to_csv(cfg.output_dir / "diff_high_vs_low_purity_tumors.csv", index=False)
    gsea_focus.to_csv(cfg.output_dir / "gsea_focus_pathways.csv", index=False)
    original_protein_loss.to_csv(cfg.output_dir / "proteins_original_sig_lost_after_purity_adjustment.csv", index=False)
    original_pathway_loss.to_csv(cfg.output_dir / "pathways_original_sig_lost_after_purity_adjustment.csv", index=False)
    pd.DataFrame([summary | {"high_purity_threshold": high_purity_threshold}]).to_csv(
        cfg.output_dir / "analysis_summary.csv",
        index=False,
    )

    _write_config_snapshot(cfg)
    report_path = _write_report(
        cfg,
        summary,
        marker_availability,
        signature_results,
        comparison,
        high_purity_threshold,
        gsea_focus,
        original_protein_loss,
        original_pathway_loss,
    )
    print(f"[purity_validation] finished; report written to {report_path}")
    return report_path
