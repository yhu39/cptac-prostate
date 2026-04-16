from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

from cptac_prostate.global_diff import _pick_sample_column, _strip_quotes


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


def _get_optional_path(
    config: configparser.ConfigParser,
    section: str,
    option: str,
) -> Path | None:
    if not config.has_section(section) or not config.has_option(section, option):
        return None
    value = _strip_quotes(config.get(section, option))
    return Path(value) if value else None


def _get_optional_list(
    config: configparser.ConfigParser,
    section: str,
    option: str,
) -> list[str]:
    if not config.has_section(section) or not config.has_option(section, option):
        return []
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


def _resolve_data_path(base_dir: Path, relative_or_abs: Path) -> Path:
    return relative_or_abs if relative_or_abs.is_absolute() else base_dir / relative_or_abs


def _normalize_grade(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return str(int(float(text)))
    except ValueError:
        return None


def _safe_fdrcorrect(pvalues: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=pvalues.index, dtype=float)
    valid = pvalues.notna()
    if valid.any():
        result.loc[valid] = fdrcorrection(pvalues.loc[valid])[1]
    return result


def _safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    frame = pd.concat([x, y], axis=1).dropna()
    if len(frame) < 3:
        return np.nan, np.nan, len(frame)
    if frame.iloc[:, 0].nunique() < 2 or frame.iloc[:, 1].nunique() < 2:
        return np.nan, np.nan, len(frame)
    rho, pvalue = stats.spearmanr(frame.iloc[:, 0], frame.iloc[:, 1])
    return float(rho), float(pvalue), len(frame)


def _safe_linear_fit(formula: str, data: pd.DataFrame, coefficient: str) -> dict[str, float]:
    frame = data.dropna().copy()
    if len(frame) < 4:
        return {"beta": np.nan, "pvalue": np.nan, "adj_r2": np.nan, "n": len(frame)}
    try:
        fit = smf.ols(formula, data=frame).fit()
    except Exception:
        return {"beta": np.nan, "pvalue": np.nan, "adj_r2": np.nan, "n": len(frame)}
    if coefficient not in fit.params.index:
        return {"beta": np.nan, "pvalue": np.nan, "adj_r2": float(fit.rsquared_adj), "n": len(frame)}
    return {
        "beta": float(fit.params[coefficient]),
        "pvalue": float(fit.pvalues[coefficient]),
        "adj_r2": float(fit.rsquared_adj),
        "n": len(frame),
    }


@dataclass
class PurityConfig:
    input_dir: Path
    global_path: Path
    meta_dir: Path
    meta_path: Path
    output_dir: Path
    feature_column: str = "geneSymbol"
    sample_id_column: str | None = None
    group_column: str = "Tissuetype"
    tumor_label: str = "tumor"
    normal_label: str = "normal"
    purity_column: str = "Purity"
    grade_column: str = "BCR_Gleason_Grade"
    batch_column: str | None = None
    stromal_score_column: str | None = None
    immune_score_column: str | None = None
    age_column: str | None = "age"
    candidate_table_path: Path | None = None
    candidate_gene_column: str = "gene"
    candidate_genes: list[str] | None = None
    high_purity_threshold: float = 0.6
    strong_purity_rho_threshold: float = 0.3
    strong_purity_fdr_threshold: float = 0.05
    major_attenuation_threshold: float = 50.0
    modest_attenuation_threshold: float = 20.0
    top_n_per_class: int = 3

    @property
    def data_path(self) -> Path:
        return _resolve_data_path(self.input_dir, self.global_path)

    @property
    def metadata_path(self) -> Path:
        return _resolve_data_path(self.meta_dir, self.meta_path)


def _load_config(config_path: Path) -> PurityConfig:
    config = _read_config(config_path)
    candidate_genes = _get_optional_list(config, "settings", "candidate_genes")
    return PurityConfig(
        input_dir=_get_required_path(config, "input", "input_dir"),
        global_path=_get_required_path(config, "input", "global_path"),
        meta_dir=_get_required_path(config, "input", "meta_dir"),
        meta_path=_get_required_path(config, "input", "meta_path"),
        output_dir=_get_required_path(config, "output", "output_dir"),
        feature_column=_get_optional_value(config, "settings", "feature_column", "geneSymbol"),
        sample_id_column=_get_optional_value(config, "settings", "sample_id_column", "") or None,
        group_column=_get_optional_value(config, "settings", "group_column", "Tissuetype"),
        tumor_label=_get_optional_value(config, "settings", "tumor_label", "tumor"),
        normal_label=_get_optional_value(config, "settings", "normal_label", "normal"),
        purity_column=_get_optional_value(config, "settings", "purity_column", "Purity"),
        grade_column=_get_optional_value(config, "settings", "grade_column", "BCR_Gleason_Grade"),
        batch_column=_get_optional_value(config, "settings", "batch_column", "") or None,
        stromal_score_column=_get_optional_value(config, "settings", "stromal_score_column", "") or None,
        immune_score_column=_get_optional_value(config, "settings", "immune_score_column", "") or None,
        age_column=_get_optional_value(config, "settings", "age_column", "age") or None,
        candidate_table_path=_get_optional_path(config, "settings", "candidate_table_path"),
        candidate_gene_column=_get_optional_value(config, "settings", "candidate_gene_column", "gene"),
        candidate_genes=candidate_genes or None,
        high_purity_threshold=_get_optional_float(config, "settings", "high_purity_threshold", 0.6),
        strong_purity_rho_threshold=_get_optional_float(
            config, "settings", "strong_purity_rho_threshold", 0.3
        ),
        strong_purity_fdr_threshold=_get_optional_float(
            config, "settings", "strong_purity_fdr_threshold", 0.05
        ),
        major_attenuation_threshold=_get_optional_float(
            config, "settings", "major_attenuation_threshold", 50.0
        ),
        modest_attenuation_threshold=_get_optional_float(
            config, "settings", "modest_attenuation_threshold", 20.0
        ),
        top_n_per_class=_get_optional_int(config, "settings", "top_n_per_class", 3),
    )


def _load_candidate_genes(cfg: PurityConfig) -> list[str]:
    if cfg.candidate_genes:
        return sorted(dict.fromkeys(cfg.candidate_genes))
    if cfg.candidate_table_path is None:
        msg = "No candidate genes were provided. Set [settings] candidate_table_path or candidate_genes."
        raise ValueError(msg)
    table = _read_table_auto(cfg.candidate_table_path)
    if cfg.candidate_gene_column not in table.columns:
        msg = (
            f"Candidate gene column '{cfg.candidate_gene_column}' was not found "
            f"in {cfg.candidate_table_path}."
        )
        raise ValueError(msg)
    genes = table[cfg.candidate_gene_column].dropna().astype(str).str.strip()
    return sorted(dict.fromkeys(gene for gene in genes if gene))


def _prepare_matrix_and_metadata(cfg: PurityConfig) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    matrix_raw = _read_table_auto(cfg.data_path)
    meta_raw = _read_table_auto(cfg.metadata_path)

    if cfg.feature_column not in matrix_raw.columns:
        msg = f"Feature column '{cfg.feature_column}' was not found in {cfg.data_path}."
        raise ValueError(msg)
    if cfg.group_column not in meta_raw.columns:
        msg = f"Group column '{cfg.group_column}' was not found in {cfg.metadata_path}."
        raise ValueError(msg)
    if cfg.purity_column not in meta_raw.columns:
        msg = f"Purity column '{cfg.purity_column}' was not found in {cfg.metadata_path}."
        raise ValueError(msg)

    duplicated_features = int(matrix_raw[cfg.feature_column].astype(str).duplicated().sum())
    matrix = matrix_raw.copy()
    matrix[cfg.feature_column] = matrix[cfg.feature_column].astype(str)
    if duplicated_features > 0:
        matrix = matrix.groupby(cfg.feature_column, as_index=False).mean(numeric_only=True)

    sample_columns = [column for column in matrix.columns if column != cfg.feature_column]
    duplicated_samples = int(pd.Index(sample_columns).duplicated().sum())
    matrix = matrix.set_index(cfg.feature_column)
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    if duplicated_samples > 0:
        matrix = matrix.T.groupby(level=0).mean().T

    sample_id_column = _pick_sample_column(meta_raw, matrix.columns.tolist(), cfg.sample_id_column)
    meta = meta_raw.copy()
    meta[sample_id_column] = meta[sample_id_column].astype(str)
    duplicated_metadata_samples = int(meta[sample_id_column].duplicated().sum())
    if duplicated_metadata_samples > 0:
        meta = meta.drop_duplicates(subset=[sample_id_column], keep="first")

    matched_samples = [sample for sample in matrix.columns if sample in set(meta[sample_id_column])]
    matrix = matrix.loc[:, matched_samples]
    meta = meta.loc[meta[sample_id_column].isin(matched_samples)].copy()
    meta = meta.set_index(sample_id_column).loc[matched_samples].reset_index()

    sample_like_columns = [column for column in matrix.columns if column not in set(meta[sample_id_column])]
    if sample_like_columns:
        matrix = matrix.drop(columns=sample_like_columns, errors="ignore")

    info = {
        "sample_id_column": sample_id_column,
        "duplicated_features": duplicated_features,
        "duplicated_samples": duplicated_samples,
        "duplicated_metadata_samples": duplicated_metadata_samples,
        "dropped_nonmatched_matrix_columns": sample_like_columns,
    }
    return matrix, meta, info


def _build_analysis_frame(
    cfg: PurityConfig,
    matrix: pd.DataFrame,
    meta: pd.DataFrame,
    genes: list[str],
    sample_id_column: str,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    matched_genes = [gene for gene in genes if gene in matrix.index]
    missing_genes = [gene for gene in genes if gene not in matrix.index]
    matrix_subset = matrix.loc[matched_genes].T.reset_index().rename(columns={"index": sample_id_column})
    long_frame = matrix_subset.melt(
        id_vars=[sample_id_column],
        var_name="protein",
        value_name="abundance",
    )
    analysis = long_frame.merge(meta, on=sample_id_column, how="left")
    analysis["group"] = analysis[cfg.group_column].astype(str).str.casefold()
    analysis["is_tumor"] = (analysis["group"] == cfg.tumor_label.casefold()).astype(int)
    analysis["is_normal"] = (analysis["group"] == cfg.normal_label.casefold()).astype(int)
    analysis["purity"] = pd.to_numeric(analysis[cfg.purity_column], errors="coerce")
    analysis["purity_filled_for_group_model"] = analysis["purity"].where(analysis["is_tumor"] == 1, 0.0)
    analysis["grade_numeric"] = (
        analysis[cfg.grade_column].apply(_normalize_grade).pipe(pd.to_numeric, errors="coerce")
        if cfg.grade_column in analysis.columns
        else np.nan
    )
    return analysis, matched_genes, missing_genes


def _compute_covariate_summary(cfg: PurityConfig, meta: pd.DataFrame, sample_id_column: str) -> dict[str, Any]:
    group_series = meta[cfg.group_column].astype(str).str.casefold()
    purity = pd.to_numeric(meta[cfg.purity_column], errors="coerce")
    summary = {
        "n_samples_metadata_matched": int(len(meta)),
        "n_tumor": int((group_series == cfg.tumor_label.casefold()).sum()),
        "n_normal": int((group_series == cfg.normal_label.casefold()).sum()),
        "n_purity_nonmissing": int(purity.notna().sum()),
        "n_tumor_purity_nonmissing": int(
            ((group_series == cfg.tumor_label.casefold()) & purity.notna()).sum()
        ),
        "n_normal_purity_nonmissing": int(
            ((group_series == cfg.normal_label.casefold()) & purity.notna()).sum()
        ),
        "n_normal_purity_zero_like": int(
            (
                (group_series == cfg.normal_label.casefold())
                & purity.fillna(-999).between(-1e-8, 1e-8)
            ).sum()
        ),
        "sample_id_column": sample_id_column,
    }
    return summary


def _analyze_proteins(cfg: PurityConfig, analysis: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    grouped = analysis.groupby("protein", sort=True)
    available_covariates = {
        "has_batch": cfg.batch_column is not None and cfg.batch_column in analysis.columns,
        "has_stromal": cfg.stromal_score_column is not None and cfg.stromal_score_column in analysis.columns,
        "has_immune": cfg.immune_score_column is not None and cfg.immune_score_column in analysis.columns,
        "has_grade": cfg.grade_column in analysis.columns and analysis["grade_numeric"].notna().any(),
    }

    for protein, frame in grouped:
        tumor_frame = frame.loc[frame["is_tumor"] == 1].copy()
        tumor_purity_frame = tumor_frame.loc[tumor_frame["purity"].notna()].copy()
        combined_frame = frame.loc[
            frame["group"].isin([cfg.tumor_label.casefold(), cfg.normal_label.casefold()])
        ].copy()
        combined_model_frame = combined_frame.copy()
        combined_model_frame["group_binary"] = combined_model_frame["is_tumor"]
        combined_model_frame["purity_model"] = combined_model_frame["purity_filled_for_group_model"]
        high_purity_frame = combined_model_frame.loc[
            (combined_model_frame["is_normal"] == 1)
            | (
                (combined_model_frame["is_tumor"] == 1)
                & (combined_model_frame["purity"] >= cfg.high_purity_threshold)
            )
        ].copy()

        rho, purity_p, n_corr = _safe_spearman(
            tumor_purity_frame["abundance"], tumor_purity_frame["purity"]
        )
        purity_lm = _safe_linear_fit(
            "abundance ~ purity", tumor_purity_frame[["abundance", "purity"]], "purity"
        )
        model1 = _safe_linear_fit(
            "abundance ~ group_binary",
            combined_model_frame[["abundance", "group_binary"]],
            "group_binary",
        )
        model2 = _safe_linear_fit(
            "abundance ~ group_binary + purity_model",
            combined_model_frame[["abundance", "group_binary", "purity_model"]],
            "group_binary",
        )

        model3 = {"beta": np.nan, "pvalue": np.nan, "adj_r2": np.nan, "n": np.nan}
        if available_covariates["has_batch"]:
            batch_frame = combined_model_frame[
                ["abundance", "group_binary", "purity_model", cfg.batch_column]
            ].rename(columns={cfg.batch_column: "batch"})
            model3 = _safe_linear_fit(
                "abundance ~ group_binary + purity_model + C(batch)",
                batch_frame,
                "group_binary",
            )

        model4 = {"beta": np.nan, "pvalue": np.nan, "adj_r2": np.nan, "n": np.nan}
        collinearity_note = ""
        extra_covariates: list[str] = []
        if available_covariates["has_stromal"]:
            extra_covariates.append(cfg.stromal_score_column)
        if available_covariates["has_immune"]:
            extra_covariates.append(cfg.immune_score_column)
        if extra_covariates:
            extra_frame = combined_model_frame[
                ["abundance", "group_binary", "purity_model"] + extra_covariates
            ].copy()
            numeric_covariates = extra_frame[["purity_model"] + extra_covariates].apply(
                pd.to_numeric, errors="coerce"
            )
            cov_corr = numeric_covariates.corr().abs()
            if cov_corr.where(~np.eye(len(cov_corr), dtype=bool)).max().max() < 0.8:
                renamed = extra_frame.rename(
                    columns={
                        cfg.stromal_score_column: "stromal_score",
                        cfg.immune_score_column: "immune_score",
                    }
                )
                terms = ["group_binary", "purity_model"]
                if available_covariates["has_stromal"]:
                    terms.append("stromal_score")
                if available_covariates["has_immune"]:
                    terms.append("immune_score")
                model4 = _safe_linear_fit(
                    "abundance ~ " + " + ".join(terms),
                    renamed[["abundance"] + terms],
                    "group_binary",
                )
            else:
                collinearity_note = "stromal/immune skipped due to high collinearity"

        high_purity_model = _safe_linear_fit(
            "abundance ~ group_binary",
            high_purity_frame[["abundance", "group_binary"]],
            "group_binary",
        )

        grade_model = {"beta": np.nan, "pvalue": np.nan, "adj_r2": np.nan, "n": np.nan}
        if available_covariates["has_grade"]:
            tumor_grade_frame = tumor_purity_frame[["abundance", "purity", "grade_numeric"]].dropna().copy()
            if tumor_grade_frame["grade_numeric"].nunique() >= 2:
                tumor_grade_frame["grade_group"] = tumor_grade_frame["grade_numeric"].astype(int).astype(str)
                grade_model = _safe_linear_fit(
                    "abundance ~ purity + C(grade_group)",
                    tumor_grade_frame[["abundance", "purity", "grade_group"]],
                    "purity",
                )

        beta_unadjusted = model1["beta"]
        beta_adjusted = model2["beta"]
        percent_attenuation = np.nan
        if pd.notna(beta_unadjusted) and abs(beta_unadjusted) > 0:
            percent_attenuation = ((abs(beta_unadjusted) - abs(beta_adjusted)) / abs(beta_unadjusted)) * 100
        direction_changed = (
            pd.notna(beta_unadjusted)
            and pd.notna(beta_adjusted)
            and np.sign(beta_unadjusted) != np.sign(beta_adjusted)
            and abs(beta_adjusted) > 0
        )

        records.append(
            {
                "protein": protein,
                "n_tumor": int(tumor_frame["abundance"].notna().sum()),
                "n_normal": int(frame.loc[frame["is_normal"] == 1, "abundance"].notna().sum()),
                "n_tumor_purity": n_corr,
                "logFC_or_group_beta_unadjusted": beta_unadjusted,
                "p_unadjusted": model1["pvalue"],
                "purity_spearman_rho": rho,
                "purity_spearman_p": purity_p,
                "purity_lm_beta": purity_lm["beta"],
                "purity_lm_p": purity_lm["pvalue"],
                "purity_lm_adj_r2": purity_lm["adj_r2"],
                "group_beta_adjusted": beta_adjusted,
                "p_adjusted": model2["pvalue"],
                "group_beta_batch_adjusted": model3["beta"],
                "p_batch_adjusted": model3["pvalue"],
                "group_beta_covariate_adjusted": model4["beta"],
                "p_covariate_adjusted": model4["pvalue"],
                "group_beta_high_purity": high_purity_model["beta"],
                "p_high_purity": high_purity_model["pvalue"],
                "purity_beta_grade_adjusted": grade_model["beta"],
                "purity_p_grade_adjusted": grade_model["pvalue"],
                "percent_attenuation": percent_attenuation,
                "direction_changed": bool(direction_changed),
                "attenuation_major": bool(
                    pd.notna(percent_attenuation) and percent_attenuation > cfg.major_attenuation_threshold
                ),
                "attenuation_modest": bool(
                    pd.notna(percent_attenuation)
                    and cfg.modest_attenuation_threshold <= percent_attenuation <= cfg.major_attenuation_threshold
                ),
                "model_note": collinearity_note,
            }
        )

    results = pd.DataFrame.from_records(records).sort_values("protein").reset_index(drop=True)
    results["fdr_unadjusted"] = _safe_fdrcorrect(results["p_unadjusted"])
    results["purity_spearman_fdr"] = _safe_fdrcorrect(results["purity_spearman_p"])
    results["fdr_adjusted"] = _safe_fdrcorrect(results["p_adjusted"])
    results["fdr_batch_adjusted"] = _safe_fdrcorrect(results["p_batch_adjusted"])
    results["fdr_covariate_adjusted"] = _safe_fdrcorrect(results["p_covariate_adjusted"])
    results["fdr_high_purity"] = _safe_fdrcorrect(results["p_high_purity"])
    results["purity_fdr_grade_adjusted"] = _safe_fdrcorrect(results["purity_p_grade_adjusted"])

    def classify(row: pd.Series) -> str:
        strong_purity = (
            pd.notna(row["purity_spearman_rho"])
            and abs(row["purity_spearman_rho"]) >= cfg.strong_purity_rho_threshold
            and pd.notna(row["purity_spearman_fdr"])
            and row["purity_spearman_fdr"] < cfg.strong_purity_fdr_threshold
        )
        robust_group = (
            pd.notna(row["fdr_adjusted"])
            and row["fdr_adjusted"] < 0.05
            and not bool(row["direction_changed"])
        )
        attenuation = row["percent_attenuation"]
        if robust_group and (pd.isna(attenuation) or attenuation < cfg.modest_attenuation_threshold) and not strong_purity:
            return "tumor_intrinsic"
        if strong_purity and pd.notna(attenuation) and attenuation > cfg.major_attenuation_threshold:
            return "purity_or_microenvironment_associated"
        return "mixed"

    results["classification"] = results.apply(classify, axis=1)

    def build_notes(row: pd.Series) -> str:
        notes: list[str] = []
        if row["classification"] == "tumor_intrinsic":
            notes.append("group effect is robust after purity adjustment")
        if row["classification"] == "purity_or_microenvironment_associated":
            notes.append("group effect attenuates strongly after purity adjustment")
        if row["classification"] == "mixed":
            notes.append("both tumor status and purity contribute")
        if bool(row["direction_changed"]):
            notes.append("effect direction changed after purity adjustment")
        if pd.notna(row["fdr_high_purity"]) and row["fdr_high_purity"] < 0.05:
            notes.append("high-purity tumor sensitivity analysis remains significant")
        if pd.notna(row["purity_fdr_grade_adjusted"]) and row["purity_fdr_grade_adjusted"] < 0.05:
            notes.append("purity association persists after grade adjustment")
        if row["model_note"]:
            notes.append(str(row["model_note"]))
        return "; ".join(notes)

    results["notes"] = results.apply(build_notes, axis=1)
    return results


def _write_tables(cfg: PurityConfig, results: pd.DataFrame) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(cfg.output_dir / "purity_test_results.tsv", sep="\t", index=False)
    for label in ["tumor_intrinsic", "purity_or_microenvironment_associated", "mixed"]:
        subset = results.loc[results["classification"] == label].copy()
        sort_columns = {
            "tumor_intrinsic": ["fdr_adjusted", "percent_attenuation"],
            "purity_or_microenvironment_associated": ["purity_spearman_fdr", "percent_attenuation"],
            "mixed": ["fdr_adjusted", "purity_spearman_fdr"],
        }[label]
        subset = subset.sort_values(sort_columns, na_position="last")
        subset.head(10).to_csv(cfg.output_dir / f"top_{label}.tsv", sep="\t", index=False)


def _plot_purity_histogram(cfg: PurityConfig, meta: pd.DataFrame) -> None:
    tumor_purity = pd.to_numeric(
        meta.loc[meta["group"] == cfg.tumor_label.casefold(), cfg.purity_column],
        errors="coerce",
    ).dropna()
    plt.figure(figsize=(6, 4))
    plt.hist(tumor_purity, bins=20, color="#4C72B0", edgecolor="white")
    plt.xlabel("Tumor purity")
    plt.ylabel("Tumor samples")
    plt.tight_layout()
    plt.savefig(cfg.output_dir / "tumor_purity_histogram.png", dpi=200)
    plt.close()


def _plot_covariate_heatmap(cfg: PurityConfig, meta: pd.DataFrame) -> None:
    covariates: dict[str, pd.Series] = {"tumor_purity": pd.to_numeric(meta[cfg.purity_column], errors="coerce")}
    if cfg.age_column and cfg.age_column in meta.columns:
        covariates["age"] = pd.to_numeric(meta[cfg.age_column], errors="coerce")
    if cfg.grade_column in meta.columns:
        covariates["grade"] = pd.to_numeric(meta[cfg.grade_column].apply(_normalize_grade), errors="coerce")
    if cfg.stromal_score_column and cfg.stromal_score_column in meta.columns:
        covariates["stromal_score"] = pd.to_numeric(meta[cfg.stromal_score_column], errors="coerce")
    if cfg.immune_score_column and cfg.immune_score_column in meta.columns:
        covariates["immune_score"] = pd.to_numeric(meta[cfg.immune_score_column], errors="coerce")
    covariate_frame = pd.DataFrame(covariates).dropna(axis=1, how="all")
    if covariate_frame.shape[1] < 2:
        return
    corr = covariate_frame.corr(method="spearman")
    plt.figure(figsize=(5, 4))
    sns.heatmap(corr, cmap="vlag", center=0, annot=True, fmt=".2f", square=True)
    plt.tight_layout()
    plt.savefig(cfg.output_dir / "covariate_correlation_heatmap.png", dpi=200)
    plt.close()


def _plot_ranked_attenuation(cfg: PurityConfig, results: pd.DataFrame) -> None:
    plot_df = results.sort_values("percent_attenuation", ascending=False, na_position="last").copy()
    plt.figure(figsize=(8, 4))
    sns.barplot(data=plot_df, x="protein", y="percent_attenuation", hue="classification", dodge=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Percent attenuation")
    plt.xlabel("")
    plt.legend(title="classification", frameon=False)
    plt.tight_layout()
    plt.savefig(cfg.output_dir / "ranked_attenuation.png", dpi=200)
    plt.close()


def _plot_effect_vs_purity(cfg: PurityConfig, results: pd.DataFrame) -> None:
    plt.figure(figsize=(5, 4))
    sns.scatterplot(
        data=results,
        x="purity_spearman_rho",
        y="logFC_or_group_beta_unadjusted",
        hue="classification",
        s=80,
    )
    plt.axhline(0, color="grey", lw=1)
    plt.axvline(0, color="grey", lw=1)
    plt.xlabel("Tumor-only purity Spearman rho")
    plt.ylabel("Tumor vs normal beta")
    plt.tight_layout()
    plt.savefig(cfg.output_dir / "effect_vs_purity_scatter.png", dpi=200)
    plt.close()


def _plot_classification_counts(cfg: PurityConfig, results: pd.DataFrame) -> None:
    counts = results["classification"].value_counts().rename_axis("classification").reset_index(name="n")
    plt.figure(figsize=(5, 4))
    sns.barplot(data=counts, x="classification", y="n")
    plt.xticks(rotation=20, ha="right")
    plt.xlabel("")
    plt.ylabel("Proteins")
    plt.tight_layout()
    plt.savefig(cfg.output_dir / "classification_counts.png", dpi=200)
    plt.close()


def _plot_beta_comparison(cfg: PurityConfig, results: pd.DataFrame) -> None:
    plt.figure(figsize=(5, 4))
    sns.scatterplot(
        data=results,
        x="logFC_or_group_beta_unadjusted",
        y="group_beta_adjusted",
        hue="classification",
        s=80,
    )
    lims = [
        np.nanmin(
            [results["logFC_or_group_beta_unadjusted"].min(), results["group_beta_adjusted"].min()]
        ),
        np.nanmax(
            [results["logFC_or_group_beta_unadjusted"].max(), results["group_beta_adjusted"].max()]
        ),
    ]
    if np.isfinite(lims).all():
        plt.plot(lims, lims, color="grey", lw=1, linestyle="--")
    plt.xlabel("Unadjusted tumor vs normal beta")
    plt.ylabel("Purity-adjusted beta")
    plt.tight_layout()
    plt.savefig(cfg.output_dir / "beta_before_after_adjustment.png", dpi=200)
    plt.close()


def _plot_beta_dumbbell(cfg: PurityConfig, results: pd.DataFrame) -> None:
    plot_df = results[
        ["protein", "logFC_or_group_beta_unadjusted", "group_beta_adjusted", "classification"]
    ].dropna().copy()
    if plot_df.empty:
        return
    plot_df = plot_df.sort_values("logFC_or_group_beta_unadjusted", ascending=True).reset_index(drop=True)
    y = np.arange(len(plot_df))

    fig_height = max(4, 0.45 * len(plot_df))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    for idx, row in plot_df.iterrows():
        ax.plot(
            [row["group_beta_adjusted"], row["logFC_or_group_beta_unadjusted"]],
            [idx, idx],
            color="#bdbdbd",
            linewidth=1.5,
            zorder=1,
        )
    ax.scatter(
        plot_df["logFC_or_group_beta_unadjusted"],
        y,
        color="#3182bd",
        s=55,
        label="Unadjusted tumor-normal beta",
        zorder=2,
    )
    ax.scatter(
        plot_df["group_beta_adjusted"],
        y,
        color="#e6550d",
        s=55,
        label="Purity-adjusted beta",
        zorder=3,
    )
    ax.axvline(0, color="grey", linewidth=1, linestyle="--")
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["protein"])
    ax.set_xlabel("Tumor vs normal beta")
    ax.set_ylabel("")
    ax.legend(frameon=False, loc="best")
    plt.tight_layout()
    plt.savefig(cfg.output_dir / "beta_before_after_dumbbell.png", dpi=220)
    plt.close(fig)


def _plot_selected_proteins(cfg: PurityConfig, analysis: pd.DataFrame, results: pd.DataFrame) -> None:
    plot_dir = cfg.output_dir / "protein_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    selected: list[str] = []
    for label, sort_columns in [
        ("tumor_intrinsic", ["fdr_adjusted", "percent_attenuation"]),
        ("purity_or_microenvironment_associated", ["purity_spearman_fdr", "percent_attenuation"]),
        ("mixed", ["fdr_adjusted", "purity_spearman_fdr"]),
    ]:
        subset = results.loc[results["classification"] == label].sort_values(sort_columns, na_position="last")
        selected.extend(subset["protein"].head(cfg.top_n_per_class).tolist())
    selected = list(dict.fromkeys(selected))
    summary_lookup = results.set_index("protein")

    for protein in selected:
        protein_frame = analysis.loc[analysis["protein"] == protein].copy()
        tumor_frame = protein_frame.loc[
            (protein_frame["is_tumor"] == 1) & protein_frame["purity"].notna()
        ].copy()
        box_frame = protein_frame.loc[
            protein_frame["group"].isin([cfg.tumor_label.casefold(), cfg.normal_label.casefold()])
        ].copy()
        box_frame["group_label"] = box_frame["group"].map(
            {cfg.normal_label.casefold(): "Normal", cfg.tumor_label.casefold(): "Tumor"}
        )
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
        sns.regplot(data=tumor_frame, x="purity", y="abundance", scatter_kws={"s": 25}, ax=axes[0])
        row = summary_lookup.loc[protein]
        axes[0].set_title(f"{protein} vs purity")
        axes[0].text(
            0.05,
            0.95,
            f"rho={row['purity_spearman_rho']:.2f}\nFDR={row['purity_spearman_fdr']:.3g}",
            transform=axes[0].transAxes,
            va="top",
        )
        sns.boxplot(
            data=box_frame,
            x="group_label",
            y="abundance",
            ax=axes[1],
            hue="group_label",
            dodge=False,
            legend=False,
            palette=["#9ecae1", "#fb6a4a"],
        )
        sns.stripplot(
            data=box_frame,
            x="group_label",
            y="abundance",
            ax=axes[1],
            color="black",
            size=3,
            alpha=0.6,
        )
        axes[1].set_title(f"{protein} tumor vs normal")
        axes[1].text(
            0.05,
            0.95,
            (
                f"beta={row['logFC_or_group_beta_unadjusted']:.2f}\n"
                f"adj_beta={row['group_beta_adjusted']:.2f}\n"
                f"atten={row['percent_attenuation']:.1f}%"
            ),
            transform=axes[1].transAxes,
            va="top",
        )
        plt.tight_layout()
        plt.savefig(plot_dir / f"{protein}_purity_panel.png", dpi=200)
        plt.close(fig)


def _write_summary_tables(
    cfg: PurityConfig,
    summary: dict[str, Any],
    matched_genes: list[str],
    missing_genes: list[str],
    info: dict[str, Any],
) -> None:
    pd.DataFrame([summary]).to_csv(cfg.output_dir / "purity_test_data_summary.tsv", sep="\t", index=False)
    pd.DataFrame(
        {"matched_candidate_genes": pd.Series(matched_genes), "missing_candidate_genes": pd.Series(missing_genes)}
    ).to_csv(cfg.output_dir / "purity_test_candidate_genes.tsv", sep="\t", index=False)
    pd.DataFrame(
        {"item": list(info.keys()), "value": [str(value) for value in info.values()]}
    ).to_csv(cfg.output_dir / "purity_test_processing_notes.tsv", sep="\t", index=False)


def _build_assumptions(cfg: PurityConfig, summary: dict[str, Any], info: dict[str, Any]) -> list[str]:
    assumptions = []
    assumptions.append(
        f"Sample IDs were aligned using metadata column `{summary['sample_id_column']}` because it matched the matrix columns."
    )
    assumptions.append(
        f"Detected {int(info['duplicated_features'])} duplicated proteins, {int(info['duplicated_samples'])} duplicated matrix sample columns, and {int(info['duplicated_metadata_samples'])} duplicated metadata sample IDs."
    )
    if info["dropped_nonmatched_matrix_columns"]:
        assumptions.append(
            "Dropped non-sample matrix columns that did not match metadata sample IDs: "
            + ", ".join(info["dropped_nonmatched_matrix_columns"])
            + "."
        )
    assumptions.append(
        f"Purity was taken from metadata column `{cfg.purity_column}`. Normal samples have no purity values in this dataset."
    )
    assumptions.append("Tumor-only purity association analyses were fit only on tumor samples with non-missing purity.")
    assumptions.append(
        "For combined tumor-vs-normal purity-adjusted models, normal samples were assigned purity=0 as a sensitivity analysis. Those coefficients are interpretable only with caution because group and purity are partially collinear."
    )
    assumptions.append(
        "Batch, stromal score, and immune score models were only fit if those covariates existed and passed a basic collinearity screen."
    )
    assumptions.append(
        f"Rule-based classification thresholds: strong purity association = |rho| >= {cfg.strong_purity_rho_threshold} and FDR < {cfg.strong_purity_fdr_threshold}; major attenuation > {cfg.major_attenuation_threshold}%; modest attenuation {cfg.modest_attenuation_threshold}% to {cfg.major_attenuation_threshold}%."
    )
    return assumptions


def _write_markdown_report(
    cfg: PurityConfig,
    results: pd.DataFrame,
    summary: dict[str, Any],
    matched_genes: list[str],
    missing_genes: list[str],
    assumptions: list[str],
) -> None:
    intrinsic = results.loc[results["classification"] == "tumor_intrinsic"].sort_values(
        ["fdr_adjusted", "percent_attenuation"], na_position="last"
    )
    purity_driven = results.loc[
        results["classification"] == "purity_or_microenvironment_associated"
    ].sort_values(["purity_spearman_fdr", "percent_attenuation"], na_position="last")
    mixed = results.loc[results["classification"] == "mixed"].sort_values(
        ["fdr_adjusted", "purity_spearman_fdr"], na_position="last"
    )
    recommended = intrinsic.head(5)["protein"].tolist()
    recommendation_note = "Prioritize proteins that remained robust after purity adjustment."
    if not recommended:
        recommended = mixed.head(5)["protein"].tolist()
        recommendation_note = (
            "No proteins met the tumor_intrinsic rule. The fallback list below contains the strongest mixed proteins, which still retain an adjusted tumor-vs-normal signal but show some purity dependence."
        )
    lines = [
        "# Purity Test Summary",
        "",
        "## Data used",
        f"- Protein matrix: `{cfg.data_path}`",
        f"- Metadata: `{cfg.metadata_path}`",
        f"- Candidate gene source: `{cfg.candidate_table_path}`" if cfg.candidate_table_path else "- Candidate gene source: `[settings] candidate_genes`",
        f"- Candidate genes tested: {len(matched_genes)}",
        f"- Missing candidate genes from the matrix: {len(missing_genes)}",
        "",
        "## Data structure summary",
        f"- Proteins in matrix: {summary['n_proteins']}",
        f"- Samples in aligned analysis set: {summary['n_samples_metadata_matched']}",
        f"- Tumor samples: {summary['n_tumor']}",
        f"- Normal samples: {summary['n_normal']}",
        f"- Samples with non-missing tumor purity: {summary['n_purity_nonmissing']}",
        f"- Tumor samples with non-missing purity: {summary['n_tumor_purity_nonmissing']}",
        f"- Normal samples with non-missing purity: {summary['n_normal_purity_nonmissing']}",
        f"- High-purity threshold used for sensitivity analysis: {cfg.high_purity_threshold}",
        "",
        "## Assumptions and statistical cautions",
    ]
    lines.extend(f"- {item}" for item in assumptions)
    lines.extend(
        [
            "",
            "## Results summary",
            f"- tumor_intrinsic: {(results['classification'] == 'tumor_intrinsic').sum()}",
            f"- purity_or_microenvironment_associated: {(results['classification'] == 'purity_or_microenvironment_associated').sum()}",
            f"- mixed: {(results['classification'] == 'mixed').sum()}",
            "",
            "## Top tumor-intrinsic proteins",
        ]
    )
    lines.extend(
        f"- {row.protein}: adjusted beta={row.group_beta_adjusted:.3f}, attenuation={row.percent_attenuation:.1f}%"
        for row in intrinsic.head(5).itertuples()
    )
    lines.extend(["", "## Top purity-associated proteins"])
    lines.extend(
        f"- {row.protein}: rho={row.purity_spearman_rho:.3f}, attenuation={row.percent_attenuation:.1f}%"
        for row in purity_driven.head(5).itertuples()
    )
    lines.extend(["", "## Top mixed proteins"])
    lines.extend(
        f"- {row.protein}: adjusted beta={row.group_beta_adjusted:.3f}, rho={row.purity_spearman_rho:.3f}"
        for row in mixed.head(5).itertuples()
    )
    lines.extend(["", "## Recommended proteins for downstream DIA validation"])
    lines.append(f"- {recommendation_note}")
    if recommended:
        lines.extend(f"- {protein}" for protein in recommended)
    else:
        lines.append("- No proteins met the tumor_intrinsic rule-based criteria.")
    lines.extend(
        [
            "",
            "## How to run",
            f"- `C:\\Users\\yhu39\\AppData\\Local\\anaconda3\\envs\\prostate\\python.exe -m cptac_prostate.cli --config {cfg.output_dir / 'config.ini'}`",
            "",
            "## Main outputs",
            "- `purity_test_results.tsv`",
            "- `top_tumor_intrinsic.tsv`",
            "- `top_purity_or_microenvironment_associated.tsv`",
            "- `top_mixed.tsv`",
            "- summary figures and per-protein panels in the output directory",
        ]
    )
    (cfg.output_dir / "purity_test_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run_purity_test(config_path: Path) -> None:
    cfg = _load_config(config_path)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    candidate_genes = _load_candidate_genes(cfg)
    matrix, meta, info = _prepare_matrix_and_metadata(cfg)
    analysis, matched_genes, missing_genes = _build_analysis_frame(
        cfg, matrix, meta, candidate_genes, info["sample_id_column"]
    )
    meta_summary = _compute_covariate_summary(cfg, meta, info["sample_id_column"])
    meta_summary["n_proteins"] = int(matrix.shape[0])
    meta_summary["n_matrix_samples"] = int(matrix.shape[1])
    meta_summary["n_candidate_genes_requested"] = int(len(candidate_genes))
    meta_summary["n_candidate_genes_matched"] = int(len(matched_genes))
    meta_summary["n_candidate_genes_missing"] = int(len(missing_genes))
    analysis["group"] = analysis["group"].astype(str)
    meta["group"] = meta[cfg.group_column].astype(str).str.casefold()

    results = _analyze_proteins(cfg, analysis)
    _write_tables(cfg, results)
    _write_summary_tables(cfg, meta_summary, matched_genes, missing_genes, info)
    _plot_purity_histogram(cfg, meta)
    _plot_covariate_heatmap(cfg, meta)
    _plot_ranked_attenuation(cfg, results)
    _plot_effect_vs_purity(cfg, results)
    _plot_classification_counts(cfg, results)
    _plot_beta_comparison(cfg, results)
    _plot_beta_dumbbell(cfg, results)
    _plot_selected_proteins(cfg, analysis, results)
    assumptions = _build_assumptions(cfg, meta_summary, info)
    _write_markdown_report(cfg, results, meta_summary, matched_genes, missing_genes, assumptions)

    print(pd.DataFrame([meta_summary]).to_string(index=False))
    print(f"purity_test output written to {cfg.output_dir}")
