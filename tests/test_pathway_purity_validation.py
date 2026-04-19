from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from cptac_prostate.pathway_purity_validation import run_pathway_purity_validation


def _write_text(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class PathwayPurityValidationTest(unittest.TestCase):
    def test_run_pathway_purity_validation_smoke(self) -> None:
        self._run_smoke("mean_zscore", "results_mean_zscore")

    def test_run_pathway_purity_validation_ssgsea_smoke(self) -> None:
        self._run_smoke("ssgsea", "results_ssgsea")

    def _run_smoke(self, score_method: str, output_name: str) -> None:
        tmp_path = Path.cwd() / "tmp_pathway_purity_validation_test"
        tmp_path.mkdir(exist_ok=True)

        matrix = pd.DataFrame(
            {
                "geneSymbol": ["MYC1", "MYC2", "EMT1", "EMT2", "MIX1", "MIX2"],
                "N1": [-1.2, -1.1, 1.4, 1.3, 0.2, 0.1],
                "N2": [-1.0, -0.9, 1.3, 1.2, 0.1, 0.0],
                "N3": [-1.1, -1.0, 1.5, 1.4, 0.3, 0.2],
                "N4": [-0.9, -0.8, 1.2, 1.1, 0.2, 0.1],
                "T1": [-0.2, -0.1, 0.8, 0.7, 0.3, 0.2],
                "T2": [0.1, 0.2, 0.6, 0.5, 0.4, 0.3],
                "T3": [0.7, 0.8, 0.1, 0.0, 0.7, 0.6],
                "T4": [1.0, 1.1, -0.2, -0.1, 0.8, 0.7],
                "T5": [1.2, 1.3, -0.3, -0.2, 0.9, 0.8],
                "T6": [1.4, 1.5, -0.5, -0.4, 1.0, 0.9],
            }
        )
        metadata = pd.DataFrame(
            {
                "SampleID": ["N1", "N2", "N3", "N4", "T1", "T2", "T3", "T4", "T5", "T6"],
                "Tissuetype": ["normal", "normal", "normal", "normal", "tumor", "tumor", "tumor", "tumor", "tumor", "tumor"],
                "Purity": [None, None, None, None, 0.45, 0.55, 0.68, 0.80, 0.88, 0.93],
                "FirstCategory": [None, None, None, None, "Low Purity", "Sufficient Purity", "Sufficient Purity", "Sufficient Purity", "Sufficient Purity", "Sufficient Purity"],
            }
        )

        matrix_path = tmp_path / "matrix.tsv"
        metadata_path = tmp_path / "metadata.tsv"
        dummy_rank_path = tmp_path / "dummy_rank.tsv"
        hallmark_gmt_path = tmp_path / "hallmark.gmt"
        original_gsea_results_path = tmp_path / "gsea_full_results.tsv"
        original_gsea_config_path = tmp_path / "config_prerank_gsea.ini"
        output_dir = tmp_path / output_name
        config_path = tmp_path / f"config_{score_method}.ini"

        matrix.to_csv(matrix_path, sep="\t", index=False)
        metadata.to_csv(metadata_path, sep="\t", index=False)
        pd.DataFrame({"Gene": ["MYC1", "EMT1"], "t_value": [2.0, -2.0]}).to_csv(dummy_rank_path, sep="\t", index=False)
        pd.DataFrame(
            {
                "Term": ["Myc Targets V1", "Epithelial Mesenchymal Transition"],
                "NES": [1.9, -2.1],
                "FDR q-val": [0.01, 0.02],
                "Lead_genes": ["MYC1;MYC2", "EMT1;EMT2"],
            }
        ).to_csv(original_gsea_results_path, sep="\t", index=False)
        _write_text(
            hallmark_gmt_path,
            [
                "Myc Targets V1\t\tMYC1\tMYC2",
                "Epithelial Mesenchymal Transition\t\tEMT1\tEMT2",
                "Complement\t\tMIX1\tMIX2",
            ],
        )
        _write_text(
            original_gsea_config_path,
            [
                "[input]",
                f'input_table = "{dummy_rank_path}"',
                "",
                "[output]",
                f'output_dir = "{tmp_path / "original_gsea"}"',
                "",
                "[settings]",
                'gene_column = "Gene"',
                'rank_column = "t_value"',
                f'gene_sets = "{hallmark_gmt_path}"',
                "permutation_num = 10",
                "min_size = 2",
                "max_size = 50",
                "seed = 7",
            ],
        )
        _write_text(
            config_path,
            [
                "[task]",
                'name = "pathway_purity_validation"',
                "",
                "[input]",
                f'gsea_config_path = "{original_gsea_config_path}"',
                f'gsea_results_path = "{original_gsea_results_path}"',
                f'matrix_path = "{matrix_path}"',
                f'metadata_path = "{metadata_path}"',
                f'hallmark_gmt_path = "{hallmark_gmt_path}"',
                "",
                "[output]",
                f'output_dir = "{output_dir}"',
                "",
                "[settings]",
                'feature_column = "geneSymbol"',
                'sample_id_column = "SampleID"',
                'group_column = "Tissuetype"',
                'tumor_label = "tumor"',
                'normal_label = "normal"',
                'purity_column = "Purity"',
                'tumor_filter_column = "FirstCategory"',
                'tumor_filter_value = "Sufficient Purity"',
                "high_purity_quantile = 0.5",
                "min_group_size = 2",
                "gsea_permutation_num = 10",
                "gsea_min_size = 2",
                "gsea_max_size = 50",
                "gsea_seed = 7",
                "pathway_significance_fdr = 0.05",
                "top_plot_n = 2",
                f'pathway_score_method = "{score_method}"',
            ],
        )

        report_path = run_pathway_purity_validation(config_path)

        self.assertTrue(report_path.exists())
        self.assertTrue((output_dir / "summary.md").exists())
        self.assertTrue((output_dir / "cleaned_metadata.csv").exists())
        self.assertTrue((output_dir / "pathway_sample_scores.csv").exists())
        self.assertTrue((output_dir / "pathway_definitions.csv").exists())
        self.assertTrue((output_dir / "original_significant_pathways.csv").exists())
        self.assertTrue((output_dir / "original_significant_positive_NES_pathways.csv").exists())
        self.assertTrue((output_dir / "original_significant_negative_NES_pathways.csv").exists())
        self.assertTrue((output_dir / "pathway_purity_evaluation.csv").exists())
        self.assertTrue((output_dir / "pathway_gsea_comparison_original_vs_adjusted.csv").exists())
        self.assertTrue((output_dir / "plots" / "pathway_significance_before_vs_after_purity_adjustment.png").exists())
        self.assertTrue((output_dir / "plots" / "pathway_significance_before_vs_after_purity_adjustment_positive_NES.png").exists())
        self.assertTrue((output_dir / "plots" / "pathway_significance_before_vs_after_purity_adjustment_negative_NES.png").exists())
        self.assertTrue((output_dir / "plots" / "pathway_significance_scatter_positive_NES.png").exists())
        self.assertTrue((output_dir / "plots" / "pathway_significance_scatter_negative_NES.png").exists())
        self.assertTrue((output_dir / "plots" / "pathway_score_vs_purity" / "Myc_Targets_V1.png").exists())
        self.assertTrue(
            (output_dir / "plots" / "pathway_score_vs_purity" / "Epithelial_Mesenchymal_Transition.png").exists()
        )
        self.assertTrue((output_dir / "plots" / "pathway_score_vs_purity_positive_NES" / "Myc_Targets_V1.png").exists())
        self.assertTrue(
            (output_dir / "plots" / "pathway_score_vs_purity_negative_NES" / "Epithelial_Mesenchymal_Transition.png").exists()
        )

        positive = pd.read_csv(output_dir / "original_significant_positive_NES_pathways.csv")
        negative = pd.read_csv(output_dir / "original_significant_negative_NES_pathways.csv")
        evaluation = pd.read_csv(output_dir / "pathway_purity_evaluation.csv")
        definitions = pd.read_csv(output_dir / "pathway_definitions.csv")

        self.assertEqual(set(positive["Term"]), {"Myc Targets V1"})
        self.assertEqual(set(negative["Term"]), {"Epithelial Mesenchymal Transition"})
        self.assertIn("positive_NES_pathway_is_purity_sensitive", set(evaluation["purity_sensitive_interpretation"]))
        self.assertIn("negative_NES_pathway_is_purity_sensitive", set(evaluation["purity_sensitive_interpretation"]))
        self.assertEqual(set(definitions["pathway_score_method"]), {score_method})
