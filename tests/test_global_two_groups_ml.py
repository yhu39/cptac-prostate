from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from cptac_prostate.global_two_groups_ml import run_global_two_groups_ml


def _write_config(config_path: Path, matrix_path: Path, metadata_path: Path, output_dir: Path) -> None:
    config_path.write_text(
        "\n".join(
            [
                "[input]",
                f'matrix_path = "{matrix_path}"',
                f'metadata_path = "{metadata_path}"',
                'group_column = "Group"',
                'sample_id_column = "SampleID"',
                'feature_column = "geneSymbol"',
                "",
                "[output]",
                f'output_dir = "{output_dir}"',
                "",
                "[task]",
                'name = "global_two_groups_ml"',
                "",
                "[analysis]",
                "max_missing_fraction = 0.4",
                'imputation_method = "median"',
                "variance_quantile = 0.1",
                "top_n_features = 12",
                "cv_folds = 3",
                "random_state = 7",
            ]
        ),
        encoding="utf-8",
    )


class GlobalTwoGroupsMLTest(unittest.TestCase):
    def test_run_global_two_groups_ml_smoke(self) -> None:
        tmp_path = Path.cwd() / "tmp_test_global_two_groups_ml"
        tmp_path.mkdir(exist_ok=True)

        rng = np.random.default_rng(7)
        n_groups = 3
        samples_per_group = 5
        n_features = 60
        labels = ["A", "B", "C"]

        matrix_values = rng.normal(0.0, 1.0, size=(n_features, n_groups * samples_per_group))
        signal_features = {
            "A": [0, 1, 2],
            "B": [3, 4, 5],
            "C": [6, 7, 8],
        }

        matrix_columns: list[str] = []
        metadata_rows: list[dict[str, str]] = []
        for group_index, group_label in enumerate(labels):
            for sample_index in range(samples_per_group):
                sample_number = group_index * samples_per_group + sample_index
                matrix_sample = f"S{sample_number:02d}.{group_label}.T"
                metadata_sample = f"S{sample_number:02d}-{group_label}_T"
                matrix_columns.append(matrix_sample)
                metadata_rows.append({"SampleID": metadata_sample, "Group": group_label})
                for feature_index in signal_features[group_label]:
                    matrix_values[feature_index, sample_number] += 4.0

        matrix_values[10, 0] = np.nan
        matrix_values[11, 3] = np.nan
        matrix_values[12, 7] = np.nan

        matrix = pd.DataFrame(matrix_values, columns=matrix_columns)
        matrix.insert(0, "Index", [f"ENSG{i:06d}" for i in range(n_features)])
        matrix.insert(0, "geneSymbol", [f"gene_{i}" for i in range(n_features)])

        metadata = pd.DataFrame(metadata_rows)

        matrix_path = tmp_path / "matrix.tsv"
        metadata_path = tmp_path / "metadata.tsv"
        output_dir = tmp_path / "results"
        config_path = tmp_path / "config.ini"

        matrix.to_csv(matrix_path, sep="\t", index=False)
        metadata.to_csv(metadata_path, sep="\t", index=False)
        _write_config(config_path, matrix_path, metadata_path, output_dir)

        report_path = run_global_two_groups_ml(config_path)

        self.assertTrue(report_path.exists())
        self.assertTrue((output_dir / "processed" / "processed_feature_matrix.tsv").exists())
        self.assertTrue((output_dir / "tables" / "feature_ranking_combined.tsv").exists())
        self.assertTrue((output_dir / "tables" / "model_metrics.tsv").exists())
        self.assertTrue((output_dir / "tables" / "random_forest_roc_curve.tsv").exists())
        self.assertTrue((output_dir / "tables" / "logistic_l1_roc_curve.tsv").exists())
        self.assertTrue((output_dir / "tables" / "sample_qc.tsv").exists())
        self.assertTrue((output_dir / "plots" / "pca_reference.png").exists())
        self.assertTrue((output_dir / "plots" / "tsne_embedding.png").exists())
        self.assertTrue((output_dir / "plots" / "lda_separation.png").exists())
        self.assertTrue((output_dir / "plots" / "random_forest_roc_curve.png").exists())
        self.assertTrue((output_dir / "plots" / "logistic_l1_roc_curve.png").exists())
        self.assertTrue((output_dir / "plots" / "qc_sample_correlation_heatmap.png").exists())

        metrics = pd.read_csv(output_dir / "tables" / "model_metrics.tsv", sep="\t")
        self.assertEqual(set(metrics["model"]), {"random_forest", "logistic_l1"})
        self.assertTrue(metrics["accuracy"].notna().all())
        self.assertTrue(metrics["roc_curve_available"].all())

        top_features = pd.read_csv(output_dir / "tables" / "top_signature_features.tsv", sep="\t")
        self.assertTrue(
            any(feature in set(top_features["feature"].head(10)) for feature in {"gene_0", "gene_3", "gene_6"})
        )
