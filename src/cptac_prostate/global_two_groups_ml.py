from __future__ import annotations

import configparser
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize


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
    return int(float(_strip_quotes(config.get(section, option))))


def _get_optional_bool(
    config: configparser.ConfigParser,
    section: str,
    option: str,
    default: bool,
) -> bool:
    if not config.has_section(section) or not config.has_option(section, option):
        return default
    return config.getboolean(section, option)


def _infer_sep(path: Path) -> str:
    return "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","


def _resolve_matrix_path(config: configparser.ConfigParser) -> Path:
    input_dir = _get_optional_path(config, "input", "input_dir")
    matrix_path = _get_optional_path(config, "input", "matrix_path")
    if matrix_path is None:
        matrix_path = _get_optional_path(config, "input", "global_path")
    if matrix_path is None:
        msg = "Config file needs [input] matrix_path or [input] global_path."
        raise ValueError(msg)
    if matrix_path.is_absolute() or input_dir is None:
        return matrix_path
    return input_dir / matrix_path


def _resolve_metadata_path(config: configparser.ConfigParser) -> Path:
    meta_dir = _get_optional_path(config, "input", "meta_dir")
    meta_path = _get_optional_path(config, "input", "metadata_path")
    if meta_path is None:
        meta_path = _get_optional_path(config, "input", "meta_path")
    if meta_path is None:
        msg = "Config file needs [input] metadata_path or [input] meta_path."
        raise ValueError(msg)
    if meta_path.is_absolute() or meta_dir is None:
        return meta_path
    return meta_dir / meta_path


def _parse_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _canonicalize_sample_id(value: object) -> str:
    text = str(value).strip()
    if text == "" or text.casefold() == "nan":
        return ""
    return re.sub(r"[^A-Za-z0-9]+", "", text).upper()


def _unique_canonical_map(values: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    duplicates: set[str] = set()
    for value in values:
        key = _canonicalize_sample_id(value)
        if key == "":
            continue
        if key in mapping and mapping[key] != value:
            duplicates.add(key)
            continue
        mapping[key] = value
    for key in duplicates:
        mapping.pop(key, None)
    return mapping


def _score_overlap(left_values: pd.Series, right_values: list[str]) -> tuple[int, int]:
    left = {
        str(value).strip()
        for value in left_values.dropna().astype(str)
        if str(value).strip() and str(value).strip().casefold() != "nan"
    }
    right = {
        str(value).strip()
        for value in right_values
        if str(value).strip() and str(value).strip().casefold() != "nan"
    }
    exact = len(left.intersection(right))
    left_canonical = {_canonicalize_sample_id(value) for value in left}
    right_canonical = {_canonicalize_sample_id(value) for value in right}
    canonical = len(left_canonical.intersection(right_canonical).difference({""}))
    return exact, canonical


def _build_palette(labels: pd.Series) -> dict[str, tuple[float, float, float]]:
    unique_labels = sorted(labels.astype(str).unique().tolist())
    colors = sns.color_palette("tab10", n_colors=max(len(unique_labels), 1))
    return {label: colors[idx] for idx, label in enumerate(unique_labels)}


def _safe_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    return numeric


def _format_metric(value: object) -> str:
    numeric = _safe_float(value)
    if numeric is not None:
        return f"{numeric:.4f}"
    return str(value)


def _frame_to_markdown(frame: pd.DataFrame, max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No rows_"

    subset = frame if max_rows is None else frame.head(max_rows)
    headers = [str(column) for column in subset.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in subset.iterrows():
        lines.append("| " + " | ".join(_format_metric(row[column]) for column in subset.columns) + " |")
    return "\n".join(lines)


@dataclass
class GlobalTwoGroupsMLConfig:
    matrix_path: Path
    metadata_path: Path
    output_dir: Path
    group_column: str | None
    sample_id_column: str | None
    feature_column: str | None
    groups: list[str] | None
    max_missing_fraction: float = 0.5
    imputation_method: str = "median"
    variance_quantile: float = 0.2
    min_features_after_variance: int = 200
    standardize: bool = True
    top_n_features: int = 30
    cv_folds: int = 5
    random_state: int = 42
    rf_estimators: int = 500
    rf_min_samples_leaf: int = 1
    logistic_c: float = 0.5


@dataclass
class MatrixLayout:
    orientation: str
    metadata_sample_column: str
    feature_column: str | None
    sample_row_column: str | None
    exact_overlap: int
    canonical_overlap: int


def _build_config(config: configparser.ConfigParser) -> GlobalTwoGroupsMLConfig:
    groups = _parse_list(_get_optional_value(config, "settings", "groups", ""))
    if not groups:
        group1 = _get_optional_value(config, "settings", "group1", "")
        group2 = _get_optional_value(config, "settings", "group2", "")
        groups = [group for group in [group1, group2] if group]

    return GlobalTwoGroupsMLConfig(
        matrix_path=_resolve_matrix_path(config),
        metadata_path=_resolve_metadata_path(config),
        output_dir=_get_required_path(config, "output", "output_dir"),
        group_column=(
            _get_optional_value(config, "input", "group_column", "").strip()
            or _get_optional_value(config, "settings", "group_column", "").strip()
            or None
        ),
        sample_id_column=(
            _get_optional_value(config, "input", "sample_id_column", "").strip()
            or _get_optional_value(config, "settings", "sample_id_column", "").strip()
            or None
        ),
        feature_column=(
            _get_optional_value(config, "input", "feature_column", "").strip()
            or _get_optional_value(config, "settings", "feature_column", "").strip()
            or None
        ),
        groups=groups or None,
        max_missing_fraction=_get_optional_float(config, "analysis", "max_missing_fraction", 0.5),
        imputation_method=_get_optional_value(config, "analysis", "imputation_method", "median"),
        variance_quantile=_get_optional_float(config, "analysis", "variance_quantile", 0.2),
        min_features_after_variance=_get_optional_int(
            config,
            "analysis",
            "min_features_after_variance",
            200,
        ),
        standardize=_get_optional_bool(config, "analysis", "standardize", True),
        top_n_features=_get_optional_int(config, "analysis", "top_n_features", 30),
        cv_folds=_get_optional_int(config, "analysis", "cv_folds", 5),
        random_state=_get_optional_int(config, "analysis", "random_state", 42),
        rf_estimators=_get_optional_int(config, "analysis", "rf_estimators", 500),
        rf_min_samples_leaf=_get_optional_int(config, "analysis", "rf_min_samples_leaf", 1),
        logistic_c=_get_optional_float(config, "analysis", "logistic_c", 0.5),
    )


def _pick_group_column(
    metadata: pd.DataFrame,
    preferred: str | None,
    requested_groups: list[str] | None,
) -> str:
    if preferred:
        if preferred not in metadata.columns:
            msg = f"Metadata column '{preferred}' was not found."
            raise ValueError(msg)
        return preferred

    if requested_groups:
        requested = {group.casefold() for group in requested_groups}
        best_column: str | None = None
        best_score = -1
        for column in metadata.columns:
            series = metadata[column].dropna().astype(str).str.strip()
            if series.empty:
                continue
            overlap = len({value.casefold() for value in series.unique()}.intersection(requested))
            if overlap > best_score:
                best_score = overlap
                best_column = str(column)
        if best_column is not None and best_score > 0:
            return best_column

    for column in ["group", "Group", "label", "Label", "class", "Class", "Tissuetype", "Grade_Group"]:
        if column in metadata.columns:
            return column

    msg = "Unable to infer a metadata group-label column. Set [input] group_column."
    raise ValueError(msg)


def _candidate_metadata_sample_columns(metadata: pd.DataFrame, preferred: str | None) -> list[str]:
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(["SampleID", "sample_id", "sample", "sampleID", "common_ID"])
    object_columns = [
        str(column)
        for column in metadata.columns
        if metadata[column].dtype == "object" or pd.api.types.is_string_dtype(metadata[column])
    ]
    candidates.extend(object_columns)

    ordered_candidates: list[str] = []
    for candidate in candidates:
        if candidate in metadata.columns and candidate not in ordered_candidates:
            ordered_candidates.append(candidate)
    return ordered_candidates


def _detect_matrix_layout(
    raw_matrix: pd.DataFrame,
    metadata: pd.DataFrame,
    preferred_meta_sample_column: str | None,
    preferred_feature_column: str | None,
) -> MatrixLayout:
    first_column = str(raw_matrix.columns[0])
    feature_candidates: list[str] = []
    if preferred_feature_column and preferred_feature_column in raw_matrix.columns:
        feature_candidates.append(preferred_feature_column)
    if first_column not in feature_candidates:
        feature_candidates.append(first_column)

    sample_row_candidates: list[str] = [first_column]
    for column in ["SampleID", "sample_id", "sample", "sampleID", "common_ID"]:
        if column in raw_matrix.columns and column not in sample_row_candidates:
            sample_row_candidates.append(column)

    best_layout: MatrixLayout | None = None
    for metadata_sample_column in _candidate_metadata_sample_columns(metadata, preferred_meta_sample_column):
        sample_values = metadata[metadata_sample_column]
        for feature_column in feature_candidates:
            matrix_sample_columns = [str(column).strip() for column in raw_matrix.columns if column != feature_column]
            exact, canonical = _score_overlap(sample_values, matrix_sample_columns)
            candidate = MatrixLayout(
                orientation="feature_rows",
                metadata_sample_column=metadata_sample_column,
                feature_column=feature_column,
                sample_row_column=None,
                exact_overlap=exact,
                canonical_overlap=canonical,
            )
            if best_layout is None or (candidate.exact_overlap, candidate.canonical_overlap) > (
                best_layout.exact_overlap,
                best_layout.canonical_overlap,
            ):
                best_layout = candidate

        for sample_row_column in sample_row_candidates:
            matrix_sample_rows = raw_matrix[sample_row_column].dropna().astype(str).tolist()
            exact, canonical = _score_overlap(sample_values, matrix_sample_rows)
            candidate = MatrixLayout(
                orientation="sample_rows",
                metadata_sample_column=metadata_sample_column,
                feature_column=None,
                sample_row_column=sample_row_column,
                exact_overlap=exact,
                canonical_overlap=canonical,
            )
            if best_layout is None or (candidate.exact_overlap, candidate.canonical_overlap) > (
                best_layout.exact_overlap,
                best_layout.canonical_overlap,
            ):
                best_layout = candidate

    if best_layout is None or max(best_layout.exact_overlap, best_layout.canonical_overlap) <= 0:
        msg = "Unable to detect matrix orientation from metadata sample IDs."
        raise ValueError(msg)

    return best_layout


def _match_samples(
    matrix_sample_ids: list[str],
    metadata: pd.DataFrame,
    metadata_sample_column: str,
) -> pd.DataFrame:
    matrix_ids = [str(sample_id).strip() for sample_id in matrix_sample_ids]
    exact_lookup = {sample_id: sample_id for sample_id in matrix_ids}
    canonical_lookup = _unique_canonical_map(matrix_ids)
    order_lookup = {sample_id: order for order, sample_id in enumerate(matrix_ids)}

    matched_records: list[dict[str, object]] = []
    seen_matrix_ids: set[str] = set()
    duplicate_matrix_ids: set[str] = set()
    for metadata_index, row in metadata.iterrows():
        raw_sample_id = str(row[metadata_sample_column]).strip()
        if not raw_sample_id or raw_sample_id.casefold() == "nan":
            continue

        matched_matrix_id = exact_lookup.get(raw_sample_id)
        match_type = "exact"
        if matched_matrix_id is None:
            matched_matrix_id = canonical_lookup.get(_canonicalize_sample_id(raw_sample_id))
            match_type = "canonical"
        if matched_matrix_id is None:
            continue
        if matched_matrix_id in seen_matrix_ids:
            duplicate_matrix_ids.add(matched_matrix_id)
            continue
        seen_matrix_ids.add(matched_matrix_id)
        matched_records.append(
            {
                "metadata_index": metadata_index,
                "metadata_sample_id": raw_sample_id,
                "matrix_sample_id": matched_matrix_id,
                "match_type": match_type,
                "matrix_order": order_lookup[matched_matrix_id],
            }
        )

    if duplicate_matrix_ids:
        duplicates = ", ".join(sorted(duplicate_matrix_ids))
        msg = f"Metadata rows map to the same matrix sample ID: {duplicates}"
        raise ValueError(msg)

    if not matched_records:
        msg = "No overlapping sample IDs were found between the matrix and metadata."
        raise ValueError(msg)

    return pd.DataFrame(matched_records).sort_values("matrix_order").reset_index(drop=True)


def _prepare_inputs(
    cfg: GlobalTwoGroupsMLConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, MatrixLayout, str]:
    metadata = pd.read_csv(cfg.metadata_path, sep=_infer_sep(cfg.metadata_path))
    raw_matrix = pd.read_csv(cfg.matrix_path, sep=_infer_sep(cfg.matrix_path))

    group_column = _pick_group_column(metadata, cfg.group_column, cfg.groups)
    metadata = metadata.dropna(subset=[group_column]).copy()
    metadata[group_column] = metadata[group_column].astype(str).str.strip()
    metadata = metadata.loc[metadata[group_column] != ""].copy()
    if cfg.groups:
        allowed_groups = {group.casefold() for group in cfg.groups}
        metadata = metadata.loc[
            metadata[group_column].astype(str).str.casefold().isin(allowed_groups)
        ].copy()

    if metadata.empty:
        msg = "No metadata rows remain after applying the group filter."
        raise ValueError(msg)

    layout = _detect_matrix_layout(
        raw_matrix=raw_matrix,
        metadata=metadata,
        preferred_meta_sample_column=cfg.sample_id_column,
        preferred_feature_column=cfg.feature_column,
    )

    if layout.orientation == "feature_rows":
        if layout.feature_column is None:
            msg = "A feature column is required for feature-row matrices."
            raise ValueError(msg)
        matrix = raw_matrix.set_index(layout.feature_column)
    else:
        if layout.sample_row_column is None:
            msg = "A sample row column is required for sample-row matrices."
            raise ValueError(msg)
        matrix = raw_matrix.set_index(layout.sample_row_column).T

    matrix.index = matrix.index.astype(str).str.strip()
    matrix.columns = pd.Index([str(column).strip() for column in matrix.columns])

    alignment = _match_samples(matrix.columns.tolist(), metadata, layout.metadata_sample_column)
    aligned_metadata = metadata.loc[alignment["metadata_index"]].copy().reset_index(drop=True)
    aligned_metadata["matrix_sample_id"] = alignment["matrix_sample_id"].astype(str).tolist()
    aligned_metadata["sample_match_type"] = alignment["match_type"].astype(str).tolist()

    matrix = matrix.loc[:, aligned_metadata["matrix_sample_id"]].copy()
    matrix = matrix.loc[~matrix.index.isna()].copy()
    matrix.index = matrix.index.astype(str).str.strip()
    matrix = matrix.loc[matrix.index != ""].copy()
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    if matrix.index.duplicated().any():
        matrix = matrix.groupby(level=0, sort=False).mean()

    if aligned_metadata[group_column].nunique() < 2:
        msg = "Need at least two groups after alignment for supervised analysis."
        raise ValueError(msg)

    return matrix, aligned_metadata, alignment, layout, group_column


def _filter_and_impute(
    matrix: pd.DataFrame,
    cfg: GlobalTwoGroupsMLConfig,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    missing_fraction = matrix.isna().mean(axis=1)
    filtered = matrix.loc[missing_fraction <= cfg.max_missing_fraction].copy()
    if filtered.empty:
        msg = "All features were removed by the missing-value filter."
        raise ValueError(msg)

    method = cfg.imputation_method.casefold()
    if method == "median":
        feature_medians = filtered.median(axis=1, skipna=True)
        filtered = filtered.T.fillna(feature_medians).T
    elif method == "zero":
        filtered = filtered.fillna(0.0)
    elif method == "none":
        filtered = filtered.dropna(axis=0, how="any")
    else:
        msg = f"Unsupported imputation method: {cfg.imputation_method}"
        raise ValueError(msg)

    if filtered.empty:
        msg = "All features were removed after imputation handling."
        raise ValueError(msg)

    variance = filtered.var(axis=1, ddof=1).fillna(0.0)
    quantile = min(max(cfg.variance_quantile, 0.0), 0.99)
    threshold = float(variance.quantile(quantile)) if len(variance) > 1 else float(variance.iloc[0])
    keep_mask = variance > threshold if quantile > 0 else variance >= threshold

    min_keep = min(len(variance), max(cfg.min_features_after_variance, cfg.top_n_features))
    if int(keep_mask.sum()) < min_keep:
        keep_index = variance.sort_values(ascending=False).head(min_keep).index
        keep_mask = variance.index.isin(keep_index)

    processed = filtered.loc[keep_mask].copy()
    if processed.empty:
        msg = "All features were removed by the variance filter."
        raise ValueError(msg)

    return processed, missing_fraction, variance


def _build_cv(y: pd.Series, requested_folds: int, random_state: int) -> StratifiedKFold | None:
    min_class_size = int(y.value_counts().min())
    folds = min(max(requested_folds, 2), min_class_size)
    if folds < 2:
        return None
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)


def _save_classification_report(
    *,
    name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    classes: list[str],
    tables_dir: Path,
) -> Path:
    report = classification_report(
        y_true,
        y_pred,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T.reset_index().rename(columns={"index": "label"})
    report_path = tables_dir / f"{name}_classification_report.tsv"
    report_df.to_csv(report_path, sep="\t", index=False)
    return report_path


def _save_roc_outputs(
    *,
    name: str,
    y_true: pd.Series,
    y_proba: np.ndarray,
    classes: list[str],
    tables_dir: Path,
    plots_dir: Path,
) -> pd.DataFrame:
    curve_rows: list[pd.DataFrame] = []
    auc_rows: list[dict[str, object]] = []
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    if len(classes) == 2:
        positive_class = classes[1]
        y_binary = (y_true.astype(str) == positive_class).astype(int).to_numpy()
        scores = y_proba[:, 1]
        fpr, tpr, thresholds = roc_curve(y_binary, scores)
        roc_auc = auc(fpr, tpr)
        curve_rows.append(
            pd.DataFrame(
                {
                    "class": positive_class,
                    "fpr": fpr,
                    "tpr": tpr,
                    "threshold": thresholds,
                    "auc": roc_auc,
                }
            )
        )
        auc_rows.append(
            {
                "class": positive_class,
                "auc": roc_auc,
                "n_positive": int(y_binary.sum()),
                "n_negative": int((1 - y_binary).sum()),
            }
        )
        ax.plot(fpr, tpr, linewidth=2, label=f"{positive_class} (AUC={roc_auc:.3f})")
    else:
        y_binary = label_binarize(y_true, classes=classes)
        for class_index, class_name in enumerate(classes):
            if np.unique(y_binary[:, class_index]).size < 2:
                continue
            fpr, tpr, thresholds = roc_curve(y_binary[:, class_index], y_proba[:, class_index])
            roc_auc = auc(fpr, tpr)
            curve_rows.append(
                pd.DataFrame(
                    {
                        "class": class_name,
                        "fpr": fpr,
                        "tpr": tpr,
                        "threshold": thresholds,
                        "auc": roc_auc,
                    }
                )
            )
            auc_rows.append(
                {
                    "class": class_name,
                    "auc": roc_auc,
                    "n_positive": int(y_binary[:, class_index].sum()),
                    "n_negative": int((1 - y_binary[:, class_index]).sum()),
                }
            )
            ax.plot(fpr, tpr, linewidth=2, label=f"{class_name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="#7f7f7f", linewidth=1)
    ax.set_title(f"{name} ROC curve")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(plots_dir / f"{name}_roc_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    curve_table = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    auc_table = pd.DataFrame(auc_rows)
    curve_table.to_csv(tables_dir / f"{name}_roc_curve.tsv", sep="\t", index=False)
    auc_table.to_csv(tables_dir / f"{name}_roc_auc_by_class.tsv", sep="\t", index=False)
    return auc_table


def _evaluate_model(
    *,
    name: str,
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    classes: list[str],
    cfg: GlobalTwoGroupsMLConfig,
    tables_dir: Path,
    plots_dir: Path,
) -> dict[str, object]:
    cv = _build_cv(y, cfg.cv_folds, cfg.random_state)
    metrics: dict[str, object] = {
        "model": name,
        "n_samples": int(len(y)),
        "n_classes": int(len(classes)),
        "cv_folds": int(cv.n_splits) if cv is not None else 0,
        "accuracy": np.nan,
        "f1_macro": np.nan,
        "f1_weighted": np.nan,
        "roc_auc_ovr_macro": np.nan,
        "roc_curve_available": False,
        "evaluation": "cross_validation" if cv is not None else "full_fit_only",
    }

    predictions_path = tables_dir / f"{name}_cv_predictions.tsv"
    confusion_path = tables_dir / f"{name}_confusion_matrix.tsv"
    plot_path = plots_dir / f"{name}_confusion_matrix.png"

    if cv is None:
        pd.DataFrame(
            {
                "sample": X.index.astype(str),
                "group_true": y.astype(str),
                "group_pred": pd.Series(index=X.index, dtype=str),
            }
        ).to_csv(predictions_path, sep="\t", index=False)
        return metrics

    y_pred = cross_val_predict(estimator, X, y, cv=cv, method="predict")
    metrics["accuracy"] = accuracy_score(y, y_pred)
    metrics["f1_macro"] = f1_score(y, y_pred, average="macro")
    metrics["f1_weighted"] = f1_score(y, y_pred, average="weighted")

    prediction_table = pd.DataFrame(
        {
            "sample": X.index.astype(str),
            "group_true": y.astype(str),
            "group_pred": pd.Series(y_pred, index=X.index, dtype=str),
        }
    )
    _save_classification_report(
        name=name,
        y_true=y,
        y_pred=y_pred,
        classes=classes,
        tables_dir=tables_dir,
    )

    try:
        y_proba = cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")
        if len(classes) == 2:
            y_binary = label_binarize(y, classes=classes).ravel()
            metrics["roc_auc_ovr_macro"] = roc_auc_score(y_binary, y_proba[:, 1])
        else:
            y_binary = label_binarize(y, classes=classes)
            metrics["roc_auc_ovr_macro"] = roc_auc_score(
                y_binary,
                y_proba,
                multi_class="ovr",
                average="macro",
            )
        probability_columns = [f"prob_{class_name}" for class_name in classes]
        prediction_table = pd.concat(
            [
                prediction_table,
                pd.DataFrame(y_proba, columns=probability_columns, index=X.index),
            ],
            axis=1,
        )
        roc_auc_table = _save_roc_outputs(
            name=name,
            y_true=y,
            y_proba=y_proba,
            classes=classes,
            tables_dir=tables_dir,
            plots_dir=plots_dir,
        )
        metrics["roc_curve_available"] = not roc_auc_table.empty
    except Exception:
        pass

    prediction_table.to_csv(predictions_path, sep="\t", index=False)

    matrix = confusion_matrix(y, y_pred, labels=classes)
    confusion_df = pd.DataFrame(matrix, index=classes, columns=classes)
    confusion_df.to_csv(confusion_path, sep="\t", index_label="true_group")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        confusion_df,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
    )
    ax.set_title(f"{name} cross-validated confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return metrics


def _plot_scatter(
    frame: pd.DataFrame,
    *,
    x_column: str,
    y_column: str,
    label_column: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    palette: dict[str, tuple[float, float, float]],
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.scatterplot(
        data=frame,
        x=x_column,
        y=y_column,
        hue=label_column,
        palette=palette,
        s=70,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(title=label_column, bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_reference_plots(
    *,
    X: pd.DataFrame,
    X_scaled: pd.DataFrame,
    y: pd.Series,
    top_features: list[str],
    processed_matrix: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    labels = y.astype(str)
    palette = _build_palette(labels)
    plots_dir = output_dir / "plots"
    notes: list[str] = []

    if X_scaled.shape[0] >= 2 and X_scaled.shape[1] >= 2:
        pca = PCA(n_components=2)
        pca_values = pca.fit_transform(X_scaled.to_numpy())
        pca_frame = pd.DataFrame(
            {
                "PC1": pca_values[:, 0],
                "PC2": pca_values[:, 1],
                "group": labels.to_numpy(),
            },
            index=X.index,
        )
        _plot_scatter(
            pca_frame,
            x_column="PC1",
            y_column="PC2",
            label_column="group",
            title="PCA reference plot",
            xlabel=f"PC1 ({pca.explained_variance_ratio_[0] * 100.0:.1f}%)",
            ylabel=f"PC2 ({pca.explained_variance_ratio_[1] * 100.0:.1f}%)",
            output_path=plots_dir / "pca_reference.png",
            palette=palette,
        )
    else:
        notes.append("PCA plot skipped because fewer than 2 samples or 2 features remained.")

    if X_scaled.shape[0] >= 4:
        perplexity = min(30.0, max(2.0, (len(X_scaled) - 1) / 3.0))
        perplexity = min(perplexity, float(len(X_scaled) - 1) - 1e-6)
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=42,
        )
        embedded = tsne.fit_transform(X_scaled.to_numpy())
        tsne_frame = pd.DataFrame(
            {
                "Dim1": embedded[:, 0],
                "Dim2": embedded[:, 1],
                "group": labels.to_numpy(),
            },
            index=X.index,
        )
        _plot_scatter(
            tsne_frame,
            x_column="Dim1",
            y_column="Dim2",
            label_column="group",
            title="t-SNE sample embedding",
            xlabel="t-SNE 1",
            ylabel="t-SNE 2",
            output_path=plots_dir / "tsne_embedding.png",
            palette=palette,
        )
    else:
        notes.append("t-SNE plot skipped because fewer than 4 samples remained.")

    if labels.nunique() >= 2:
        lda_components = min(2, labels.nunique() - 1)
        try:
            lda = LinearDiscriminantAnalysis(n_components=lda_components)
            transformed = lda.fit_transform(X_scaled.to_numpy(), labels.to_numpy())
            if transformed.ndim == 1 or transformed.shape[1] == 1:
                lda_frame = pd.DataFrame(
                    {
                        "LD1": np.ravel(transformed),
                        "group": labels.to_numpy(),
                    },
                    index=X.index,
                )
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.stripplot(
                    data=lda_frame,
                    x="group",
                    y="LD1",
                    hue="group",
                    palette=palette,
                    dodge=False,
                    ax=ax,
                )
                ax.set_title("LDA supervised separation")
                ax.set_xlabel("Group")
                ax.set_ylabel("LD1")
                if ax.legend_ is not None:
                    ax.legend_.remove()
                fig.tight_layout()
                fig.savefig(plots_dir / "lda_separation.png", dpi=200, bbox_inches="tight")
                plt.close(fig)
            else:
                lda_frame = pd.DataFrame(
                    {
                        "LD1": transformed[:, 0],
                        "LD2": transformed[:, 1],
                        "group": labels.to_numpy(),
                    },
                    index=X.index,
                )
                _plot_scatter(
                    lda_frame,
                    x_column="LD1",
                    y_column="LD2",
                    label_column="group",
                    title="LDA supervised separation",
                    xlabel="LD1",
                    ylabel="LD2",
                    output_path=plots_dir / "lda_separation.png",
                    palette=palette,
                )
        except Exception as exc:
            notes.append(f"LDA plot skipped: {exc}")

    if top_features:
        top_frame = processed_matrix.loc[top_features, X.index].copy()
        row_std = top_frame.std(axis=1, ddof=1).replace(0.0, np.nan)
        top_frame = top_frame.sub(top_frame.mean(axis=1), axis=0).div(row_std, axis=0).fillna(0.0)
        ordered_samples = y.sort_values(kind="stable").index.tolist()
        top_frame = top_frame.loc[:, ordered_samples]
        col_colors = y.loc[ordered_samples].map(palette)
        grid = sns.clustermap(
            top_frame,
            cmap="vlag",
            col_cluster=False,
            row_cluster=True,
            xticklabels=False,
            yticklabels=True,
            col_colors=col_colors,
            figsize=(10, max(6, len(top_features) * 0.25 + 2)),
        )
        handles = [
            plt.Line2D([0], [0], marker="s", linestyle="", color=color, label=label, markersize=8)
            for label, color in palette.items()
        ]
        grid.ax_heatmap.legend(
            handles=handles,
            title="Group",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            borderaxespad=0,
        )
        grid.fig.suptitle("Top signature-feature heatmap", y=1.02)
        grid.savefig(plots_dir / "top_signature_heatmap.png", dpi=200, bbox_inches="tight")
        plt.close(grid.fig)

    return notes


def _save_importance_plot(
    combined_ranking: pd.DataFrame,
    output_dir: Path,
    top_n: int,
) -> None:
    top_frame = combined_ranking.head(top_n).copy()
    if top_frame.empty:
        return
    top_frame = top_frame.sort_values("combined_score", ascending=True)

    fig, ax = plt.subplots(figsize=(8, max(5, len(top_frame) * 0.35)))
    sns.barplot(
        data=top_frame,
        x="combined_score",
        y="feature",
        color="#4C72B0",
        ax=ax,
    )
    ax.set_title("Top signature-feature importance")
    ax.set_xlabel("Combined importance score")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(output_dir / "plots" / "top_signature_feature_importance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def _save_qc_artifacts(
    *,
    raw_matrix: pd.DataFrame,
    processed_matrix: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    labels = y.astype(str)
    palette = _build_palette(labels)

    sample_qc = pd.DataFrame(index=raw_matrix.columns.astype(str))
    sample_qc["sample"] = sample_qc.index
    sample_qc["group"] = labels.reindex(sample_qc.index).astype(str)
    sample_qc["missing_fraction_before_imputation"] = raw_matrix.isna().mean(axis=0).reindex(sample_qc.index)
    sample_qc["mean_signal_before_imputation"] = raw_matrix.mean(axis=0, skipna=True).reindex(sample_qc.index)
    sample_qc["median_signal_before_imputation"] = raw_matrix.median(axis=0, skipna=True).reindex(sample_qc.index)
    sample_qc["std_signal_before_imputation"] = raw_matrix.std(axis=0, skipna=True, ddof=1).reindex(sample_qc.index)
    sample_qc["mean_signal_after_processing"] = processed_matrix.mean(axis=0, skipna=True).reindex(sample_qc.index)
    sample_qc["median_signal_after_processing"] = processed_matrix.median(axis=0, skipna=True).reindex(sample_qc.index)
    sample_qc["std_signal_after_processing"] = processed_matrix.std(axis=0, skipna=True, ddof=1).reindex(sample_qc.index)
    sample_qc = sample_qc.reset_index(drop=True)
    sample_qc.to_csv(tables_dir / "sample_qc.tsv", sep="\t", index=False)

    group_signal_summary = (
        sample_qc.groupby("group", dropna=False)
        .agg(
            n_samples=("sample", "size"),
            median_missing_fraction=("missing_fraction_before_imputation", "median"),
            median_signal_before=("median_signal_before_imputation", "median"),
            median_signal_after=("median_signal_after_processing", "median"),
        )
        .reset_index()
    )
    group_signal_summary.to_csv(tables_dir / "group_signal_summary.tsv", sep="\t", index=False)

    sample_corr = processed_matrix.corr(method="pearson")
    sample_corr.to_csv(tables_dir / "sample_correlation.tsv", sep="\t", index_label="sample")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=sample_qc, x="group", hue="group", palette=palette, ax=ax)
    ax.set_title("Sample counts by group")
    ax.set_xlabel("Group")
    ax.set_ylabel("Samples")
    if ax.legend_ is not None:
        ax.legend_.remove()
    fig.tight_layout()
    fig.savefig(plots_dir / "qc_group_size_barplot.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(
        data=sample_qc,
        x="group",
        y="missing_fraction_before_imputation",
        hue="group",
        palette=palette,
        ax=ax,
        dodge=False,
    )
    sns.stripplot(
        data=sample_qc,
        x="group",
        y="missing_fraction_before_imputation",
        hue="group",
        palette=palette,
        dodge=False,
        linewidth=0,
        size=3,
        alpha=0.6,
        ax=ax,
    )
    ax.set_title("Sample missingness before imputation")
    ax.set_xlabel("Group")
    ax.set_ylabel("Missing fraction")
    if ax.legend_ is not None:
        ax.legend_.remove()
    fig.tight_layout()
    fig.savefig(plots_dir / "qc_sample_missingness_by_group.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(
        raw_matrix.isna().mean(axis=1),
        bins=40,
        color="#4C72B0",
        ax=ax,
    )
    ax.set_title("Feature missingness distribution")
    ax.set_xlabel("Missing fraction per feature")
    ax.set_ylabel("Feature count")
    fig.tight_layout()
    fig.savefig(plots_dir / "qc_feature_missingness_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.histplot(processed_matrix.var(axis=1, ddof=1), bins=40, color="#55A868", ax=ax)
    ax.set_title("Feature variance distribution after preprocessing")
    ax.set_xlabel("Variance")
    ax.set_ylabel("Feature count")
    fig.tight_layout()
    fig.savefig(plots_dir / "qc_feature_variance_distribution.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    sns.scatterplot(
        data=sample_qc,
        x="median_signal_before_imputation",
        y="missing_fraction_before_imputation",
        hue="group",
        palette=palette,
        s=50,
        ax=ax,
    )
    ax.set_title("Sample median signal vs missingness")
    ax.set_xlabel("Median signal before imputation")
    ax.set_ylabel("Missing fraction before imputation")
    ax.legend(title="group", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    fig.tight_layout()
    fig.savefig(plots_dir / "qc_sample_signal_vs_missingness.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    ordered_samples = labels.sort_values(kind="stable").index.tolist()
    heatmap_data = sample_corr.loc[ordered_samples, ordered_samples]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        heatmap_data,
        cmap="vlag",
        vmin=max(-1.0, float(heatmap_data.min().min())),
        vmax=1.0,
        xticklabels=False,
        yticklabels=False,
        ax=ax,
    )
    ax.set_title("Sample correlation heatmap")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Samples")
    fig.tight_layout()
    fig.savefig(plots_dir / "qc_sample_correlation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return sample_qc, group_signal_summary


def _write_summary_report(
    *,
    cfg: GlobalTwoGroupsMLConfig,
    group_column: str,
    layout: MatrixLayout,
    alignment: pd.DataFrame,
    qc_summary: pd.DataFrame,
    group_counts: pd.DataFrame,
    group_signal_summary: pd.DataFrame,
    metrics_table: pd.DataFrame,
    top_candidates: pd.DataFrame,
    notes: list[str],
    output_dir: Path,
) -> Path:
    report_path = output_dir / "summary_report.md"
    matched_by_type = alignment["match_type"].value_counts().sort_index().reset_index()
    matched_by_type.columns = ["match_type", "n_samples"]

    lines = [
        "# Supervised Signature Discovery Summary",
        "",
        "## Inputs",
        "",
        f"- Matrix: `{cfg.matrix_path}`",
        f"- Metadata: `{cfg.metadata_path}`",
        f"- Output: `{output_dir}`",
        f"- Group column: `{group_column}`",
        f"- Metadata sample ID column: `{layout.metadata_sample_column}`",
        f"- Detected matrix orientation: `{layout.orientation}`",
        "",
        "## QC",
        "",
        _frame_to_markdown(qc_summary),
        "",
        "### Group counts",
        "",
        _frame_to_markdown(group_counts),
        "",
        "### Sample ID match types",
        "",
        _frame_to_markdown(matched_by_type),
        "",
        "### Group signal summary",
        "",
        _frame_to_markdown(group_signal_summary),
        "",
        "## Model evaluation",
        "",
        _frame_to_markdown(metrics_table),
        "",
        "## Top candidate signature features",
        "",
        _frame_to_markdown(top_candidates, max_rows=min(len(top_candidates), 20)),
        "",
        "## Key outputs",
        "",
        "- `processed/processed_feature_matrix.tsv`",
        "- `processed/aligned_metadata.tsv`",
        "- `tables/feature_ranking_combined.tsv`",
        "- `tables/random_forest_roc_curve.tsv`",
        "- `tables/logistic_l1_roc_curve.tsv`",
        "- `tables/random_forest_classification_report.tsv`",
        "- `tables/logistic_l1_classification_report.tsv`",
        "- `tables/sample_qc.tsv`",
        "- `tables/sample_correlation.tsv`",
        "- `plots/pca_reference.png`",
        "- `plots/tsne_embedding.png`",
        "- `plots/lda_separation.png`",
        "- `plots/random_forest_roc_curve.png`",
        "- `plots/logistic_l1_roc_curve.png`",
        "- `plots/top_signature_feature_importance.png`",
        "- `plots/top_signature_heatmap.png`",
        "- `plots/qc_sample_missingness_by_group.png`",
        "- `plots/qc_feature_missingness_distribution.png`",
        "- `plots/qc_sample_correlation_heatmap.png`",
        "",
    ]

    if notes:
        lines.extend(["## Notes", ""])
        lines.extend(f"- {note}" for note in notes)
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_global_two_groups_ml(config_path: Path) -> Path:
    config = _read_config(config_path)
    cfg = _build_config(config)

    output_dir = cfg.output_dir
    processed_dir = output_dir / "processed"
    tables_dir = output_dir / "tables"
    plots_dir = output_dir / "plots"
    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    matrix, aligned_metadata, alignment, layout, group_column = _prepare_inputs(cfg)
    raw_feature_count = int(matrix.shape[0])
    raw_sample_count = int(matrix.shape[1])
    processed_matrix, missing_fraction, variance = _filter_and_impute(matrix, cfg)

    X = processed_matrix.T.copy()
    X.index = aligned_metadata["matrix_sample_id"].astype(str).tolist()
    y = aligned_metadata.set_index("matrix_sample_id")[group_column].reindex(X.index).astype(str)

    scaler = StandardScaler()
    X_scaled_values = scaler.fit_transform(X.to_numpy()) if cfg.standardize else X.to_numpy()
    X_scaled = pd.DataFrame(X_scaled_values, index=X.index, columns=X.columns)

    feature_missing = missing_fraction.reindex(processed_matrix.index)
    feature_variance = variance.reindex(processed_matrix.index)

    aligned_metadata.to_csv(processed_dir / "aligned_metadata.tsv", sep="\t", index=False)
    alignment.to_csv(processed_dir / "sample_alignment.tsv", sep="\t", index=False)
    processed_matrix.to_csv(processed_dir / "processed_feature_matrix.tsv", sep="\t", index_label="feature")
    X_scaled.T.to_csv(
        processed_dir / "processed_feature_matrix_standardized.tsv",
        sep="\t",
        index_label="feature",
    )

    feature_qc = pd.DataFrame(
        {
            "feature": processed_matrix.index.astype(str),
            "missing_fraction_before_imputation": feature_missing.to_numpy(),
            "variance_after_imputation": feature_variance.to_numpy(),
        }
    ).sort_values("variance_after_imputation", ascending=False)
    feature_qc.to_csv(tables_dir / "feature_qc.tsv", sep="\t", index=False)

    group_counts = y.value_counts().sort_index().rename_axis("group").reset_index(name="n_samples")
    group_counts.to_csv(tables_dir / "group_counts.tsv", sep="\t", index=False)
    sample_qc, group_signal_summary = _save_qc_artifacts(
        raw_matrix=matrix,
        processed_matrix=processed_matrix,
        y=y,
        output_dir=output_dir,
    )

    rf_model = RandomForestClassifier(
        n_estimators=cfg.rf_estimators,
        min_samples_leaf=cfg.rf_min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=cfg.random_state,
        n_jobs=1,
    )
    rf_model.fit(X, y)
    rf_ranking = pd.DataFrame(
        {
            "feature": X.columns.astype(str),
            "random_forest_importance": rf_model.feature_importances_,
        }
    ).sort_values("random_forest_importance", ascending=False, kind="stable")
    rf_ranking["random_forest_rank"] = np.arange(1, len(rf_ranking) + 1)
    rf_ranking.to_csv(tables_dir / "feature_ranking_random_forest.tsv", sep="\t", index=False)

    logistic_model = LogisticRegression(
        penalty="elasticnet",
        l1_ratio=1.0,
        solver="saga",
        C=cfg.logistic_c,
        class_weight="balanced",
        max_iter=5000,
        random_state=cfg.random_state,
    )
    logistic_model.fit(X_scaled, y)
    logistic_importance = np.abs(logistic_model.coef_).mean(axis=0)
    logistic_ranking = pd.DataFrame(
        {
            "feature": X.columns.astype(str),
            "logistic_l1_abs_coef": logistic_importance,
        }
    ).sort_values("logistic_l1_abs_coef", ascending=False, kind="stable")
    logistic_ranking["logistic_l1_rank"] = np.arange(1, len(logistic_ranking) + 1)
    logistic_ranking.to_csv(tables_dir / "feature_ranking_logistic_l1.tsv", sep="\t", index=False)

    combined_ranking = rf_ranking.merge(logistic_ranking, on="feature", how="outer")
    combined_ranking["random_forest_importance"] = combined_ranking["random_forest_importance"].fillna(0.0)
    combined_ranking["logistic_l1_abs_coef"] = combined_ranking["logistic_l1_abs_coef"].fillna(0.0)
    combined_ranking["random_forest_score_norm"] = np.where(
        combined_ranking["random_forest_importance"].max() > 0,
        combined_ranking["random_forest_importance"] / combined_ranking["random_forest_importance"].max(),
        0.0,
    )
    combined_ranking["logistic_l1_score_norm"] = np.where(
        combined_ranking["logistic_l1_abs_coef"].max() > 0,
        combined_ranking["logistic_l1_abs_coef"] / combined_ranking["logistic_l1_abs_coef"].max(),
        0.0,
    )
    combined_ranking["combined_score"] = (
        combined_ranking["random_forest_score_norm"] + combined_ranking["logistic_l1_score_norm"]
    )
    combined_ranking = combined_ranking.sort_values(
        ["combined_score", "random_forest_importance", "logistic_l1_abs_coef", "feature"],
        ascending=[False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    combined_ranking["combined_rank"] = np.arange(1, len(combined_ranking) + 1)
    combined_ranking.to_csv(tables_dir / "feature_ranking_combined.tsv", sep="\t", index=False)

    top_candidates = combined_ranking.head(cfg.top_n_features).copy()
    top_candidates.to_csv(tables_dir / "top_signature_features.tsv", sep="\t", index=False)

    classes = sorted(y.unique().tolist())
    rf_eval = _evaluate_model(
        name="random_forest",
        estimator=rf_model,
        X=X,
        y=y,
        classes=classes,
        cfg=cfg,
        tables_dir=tables_dir,
        plots_dir=plots_dir,
    )
    logistic_estimator = Pipeline(
        [
            ("scaler", StandardScaler() if cfg.standardize else "passthrough"),
            (
                "classifier",
                LogisticRegression(
                    penalty="elasticnet",
                    l1_ratio=1.0,
                    solver="saga",
                    C=cfg.logistic_c,
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=cfg.random_state,
                ),
            ),
        ]
    )
    logistic_eval = _evaluate_model(
        name="logistic_l1",
        estimator=logistic_estimator,
        X=X,
        y=y,
        classes=classes,
        cfg=cfg,
        tables_dir=tables_dir,
        plots_dir=plots_dir,
    )

    metrics_table = pd.DataFrame([rf_eval, logistic_eval])
    metrics_table.to_csv(tables_dir / "model_metrics.tsv", sep="\t", index=False)

    top_features = top_candidates["feature"].head(min(cfg.top_n_features, 30)).astype(str).tolist()
    notes = _save_reference_plots(
        X=X,
        X_scaled=X_scaled,
        y=y,
        top_features=top_features,
        processed_matrix=processed_matrix,
        output_dir=output_dir,
    )
    _save_importance_plot(combined_ranking, output_dir, top_n=min(cfg.top_n_features, 20))

    qc_summary = pd.DataFrame(
        [
            {"metric": "raw_feature_count", "value": raw_feature_count},
            {"metric": "raw_sample_count", "value": raw_sample_count},
            {"metric": "matched_sample_count", "value": int(len(aligned_metadata))},
            {"metric": "matched_exact", "value": int((alignment["match_type"] == "exact").sum())},
            {"metric": "matched_canonical", "value": int((alignment["match_type"] == "canonical").sum())},
            {
                "metric": "feature_count_after_missing_filter",
                "value": int((missing_fraction <= cfg.max_missing_fraction).sum()),
            },
            {"metric": "feature_count_after_variance_filter", "value": int(processed_matrix.shape[0])},
            {"metric": "n_groups", "value": int(y.nunique())},
            {"metric": "missing_fraction_cutoff", "value": cfg.max_missing_fraction},
            {"metric": "imputation_method", "value": cfg.imputation_method},
            {"metric": "variance_quantile", "value": cfg.variance_quantile},
            {"metric": "standardized_for_modeling", "value": cfg.standardize},
            {"metric": "matrix_orientation", "value": layout.orientation},
            {"metric": "metadata_sample_column", "value": layout.metadata_sample_column},
            {"metric": "median_sample_missing_fraction", "value": sample_qc["missing_fraction_before_imputation"].median()},
            {"metric": "median_sample_signal_before_imputation", "value": sample_qc["median_signal_before_imputation"].median()},
        ]
    )
    qc_summary.to_csv(tables_dir / "qc_summary.tsv", sep="\t", index=False)

    report_path = _write_summary_report(
        cfg=cfg,
        group_column=group_column,
        layout=layout,
        alignment=alignment,
        qc_summary=qc_summary,
        group_counts=group_counts,
        group_signal_summary=group_signal_summary,
        metrics_table=metrics_table,
        top_candidates=top_candidates,
        notes=notes,
        output_dir=output_dir,
    )

    print(f"Processed matrix written to {processed_dir / 'processed_feature_matrix.tsv'}")
    print(f"Combined feature ranking written to {tables_dir / 'feature_ranking_combined.tsv'}")
    print(f"Metrics written to {tables_dir / 'model_metrics.tsv'}")
    print(f"Summary report written to {report_path}")
    return report_path
