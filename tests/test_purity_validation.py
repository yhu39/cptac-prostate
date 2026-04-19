from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from cptac_prostate.purity_validation import run_purity_validation


def _write_text(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class PurityValidationTest(unittest.TestCase):
    def test_run_purity_validation_smoke(self) -> None:
        tmp_path = Path.cwd() / "tmp_purity_validation_test"
        tmp_path.mkdir(exist_ok=True)

        matrix = pd.DataFrame(
            {
                "geneSymbol": [
                    "POSTN",
                    "COMP",
                    "THBS4",
                    "SFRP4",
                    "DCN",
                    "EPCAM",
                    "AMACR",
                    "ENTPD5",
                    "GOLM1",
                    "TAGLN",
                    "ACTA2",
                    "KRT8",
                ],
                "N1": [2.0, 1.8, 1.6, 1.5, 1.7, -1.2, -1.1, -1.0, -1.0, 1.8, 1.6, -0.9],
                "N2": [1.9, 1.7, 1.5, 1.4, 1.6, -1.0, -0.9, -0.8, -0.8, 1.7, 1.5, -0.8],
                "N3": [2.1, 1.9, 1.7, 1.6, 1.8, -1.1, -1.0, -0.9, -0.9, 1.9, 1.7, -1.0],
                "N4": [1.8, 1.6, 1.4, 1.3, 1.5, -1.0, -0.8, -0.7, -0.7, 1.6, 1.4, -0.7],
                "T1": [1.5, 1.4, 1.2, 1.0, 1.2, -0.2, 0.0, 0.1, 0.0, 1.2, 1.0, -0.1],
                "T2": [1.2, 1.1, 0.9, 0.8, 0.9, 0.2, 0.3, 0.4, 0.3, 0.9, 0.8, 0.2],
                "T3": [0.8, 0.7, 0.5, 0.4, 0.5, 0.8, 0.9, 1.0, 0.9, 0.4, 0.3, 0.8],
                "T4": [0.6, 0.5, 0.3, 0.2, 0.3, 1.0, 1.1, 1.2, 1.1, 0.2, 0.1, 1.0],
                "T5": [0.5, 0.4, 0.2, 0.1, 0.2, 1.1, 1.2, 1.3, 1.2, 0.1, 0.0, 1.1],
                "T6": [0.4, 0.3, 0.1, 0.0, 0.1, 1.2, 1.3, 1.4, 1.3, 0.0, -0.1, 1.2],
            }
        )
        metadata = pd.DataFrame(
            {
                "SampleID": ["N1", "N2", "N3", "N4", "T1", "T2", "T3", "T4", "T5", "T6"],
                "Tissuetype": ["normal", "normal", "normal", "normal", "tumor", "tumor", "tumor", "tumor", "tumor", "tumor"],
                "Purity": [None, None, None, None, 0.45, 0.55, 0.70, 0.80, 0.88, 0.92],
                "FirstCategory": [None, None, None, None, "Low Purity", "Sufficient Purity", "Sufficient Purity", "Sufficient Purity", "Sufficient Purity", "Sufficient Purity"],
            }
        )

        matrix_path = tmp_path / "matrix.tsv"
        metadata_path = tmp_path / "metadata.tsv"
        hallmark_terms_path = tmp_path / "hallmark_terms.tsv"
        kegg_results_path = tmp_path / "kegg_results.tsv"
        hallmark_gmt_path = tmp_path / "hallmark.gmt"
        kegg_gmt_path = tmp_path / "kegg.gmt"
        original_protein_results_path = tmp_path / "original_proteins.tsv"
        original_pathway_results_path = tmp_path / "original_pathways.tsv"
        output_dir = tmp_path / "results"
        config_path = tmp_path / "config.ini"

        matrix.to_csv(matrix_path, sep="\t", index=False)
        metadata.to_csv(metadata_path, sep="\t", index=False)
        pd.DataFrame(
            {
                "Term": ["Myogenesis", "Coagulation", "Epithelial Mesenchymal Transition"],
                "Genes": ["TAGLN;ACTA2;COMP", "POSTN;COMP;THBS4", "POSTN;COMP;SFRP4"],
            }
        ).to_csv(hallmark_terms_path, sep="\t", index=False)
        pd.DataFrame(
            {
                "Term": ["Focal adhesion", "ECM-receptor interaction"],
                "Lead_genes": ["POSTN;COMP;THBS4", "POSTN;COMP;DCN"],
            }
        ).to_csv(kegg_results_path, sep="\t", index=False)
        pd.DataFrame(
            {
                "Gene": ["POSTN", "COMP", "EPCAM"],
                "FDR": [0.001, 0.01, 0.2],
                "Significance": ["S-UP", "S-UP", "NS"],
                "Log2FC(mean)": [1.1, 0.9, -0.3],
            }
        ).to_csv(original_protein_results_path, sep="\t", index=False)
        pd.DataFrame(
            {
                "Term": ["Myogenesis", "Coagulation", "Focal adhesion"],
                "NES": [-2.0, -1.8, -2.1],
                "FDR q-val": [0.01, 0.02, 0.03],
                "Lead_genes": ["TAGLN;ACTA2;COMP", "POSTN;COMP;THBS4", "POSTN;COMP;THBS4"],
            }
        ).to_csv(original_pathway_results_path, sep="\t", index=False)
        _write_text(
            hallmark_gmt_path,
            [
                "Myogenesis\t\tTAGLN\tACTA2\tCOMP",
                "Coagulation\t\tPOSTN\tCOMP\tTHBS4",
                "Epithelial Mesenchymal Transition\t\tPOSTN\tCOMP\tSFRP4",
                "MYC Targets V1\t\tEPCAM\tAMACR\tENTPD5",
                "MYC Targets V2\t\tGOLM1\tEPCAM\tKRT8",
                "E2F Targets\t\tEPCAM\tAMACR\tKRT8",
                "Oxidative Phosphorylation\t\tKRT8\tENTPD5\tAMACR",
            ],
        )
        _write_text(
            kegg_gmt_path,
            [
                "Focal adhesion\t\tPOSTN\tCOMP\tTHBS4",
                "ECM-receptor interaction\t\tPOSTN\tCOMP\tDCN",
                "Complement and coagulation cascades\t\tPOSTN\tCOMP\tTHBS4",
            ],
        )
        _write_text(
            config_path,
            [
                "[task]",
                'name = "purity_validation"',
                "",
                "[input]",
                f'matrix_path = "{matrix_path}"',
                f'metadata_path = "{metadata_path}"',
                f'hallmark_terms_path = "{hallmark_terms_path}"',
                f'kegg_gsea_results_path = "{kegg_results_path}"',
                f'original_protein_results_path = "{original_protein_results_path}"',
                f'original_pathway_results_path = "{original_pathway_results_path}"',
                f'hallmark_gmt_path = "{hallmark_gmt_path}"',
                f'kegg_gmt_path = "{kegg_gmt_path}"',
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
            ],
        )

        report_path = run_purity_validation(config_path)

        self.assertTrue(report_path.exists())
        self.assertTrue((output_dir / "cleaned_metadata.csv").exists())
        self.assertTrue((output_dir / "sample_signature_scores.csv").exists())
        self.assertTrue((output_dir / "protein_purity_correlations.csv").exists())
        self.assertTrue((output_dir / "signature_purity_correlations.csv").exists())
        self.assertTrue((output_dir / "diff_unadjusted.csv").exists())
        self.assertTrue((output_dir / "diff_purity_adjusted.csv").exists())
        self.assertTrue((output_dir / "diff_comparison_adjusted_vs_unadjusted.csv").exists())
        self.assertTrue((output_dir / "gsea_focus_pathways.csv").exists())
        self.assertTrue((output_dir / "proteins_original_sig_lost_after_purity_adjustment.csv").exists())
        self.assertTrue((output_dir / "pathways_original_sig_lost_after_purity_adjustment.csv").exists())
        self.assertTrue((output_dir / "plots" / "signature_by_purity_group.png").exists())

        signatures = pd.read_csv(output_dir / "signature_purity_correlations.csv")
        self.assertIn("stromal_ecm_score", set(signatures["signature"]))
