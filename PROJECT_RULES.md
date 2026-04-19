# CPTAC Prostate Project Rules

## Environment

- Python environment: `C:\Users\yhu39\AppData\Local\anaconda3\envs\prostate`

## Core analysis rules

### Tumor and normal definitions

Current purity-validation analyses use:

- tumor: `Tissuetype == "tumor"` and `FirstCategory == "Sufficient Purity"`
- normal: `Tissuetype == "normal"`

### Original protein significance

For the original protein differential results in:

- `E:\lab\cptac-prostate\runs\20260407_global_diff_msfragger\tumor_vs_normal_diff.tsv`

the authoritative original significance definition is the `Significance` label from the source table:

- `S-U`: significantly upregulated in tumor
- `S-D`: significantly downregulated in tumor, equivalently higher in normal/NAT

These labels come from the original `global_diff` workflow and are defined using:

- `FDR <= 0.01`
- `Log2FC(median) >= log2(1.5)` for `S-U`
- `Log2FC(median) <= -log2(1.5)` for `S-D`

Important rule:

- Do not treat `S-U` as equivalent to any generic `FDR < 0.05`.
- `S-U` specifically means `FDR <= 0.01` and `Log2FC(median) >= 0.58`.
- `S-D` specifically means `FDR <= 0.01` and `Log2FC(median) <= -0.58`.

When downstream purity-validation refers to "original significant" proteins, it should align to these original `S-U` / `S-D` calls rather than redefining significance with a looser custom rule.

### Original direction labels

For the source `Significance` labels:

- `S-U` and `U` should both be interpreted as tumor-up
- `S-D` and `D` should both be interpreted as tumor-down

Direction note:

- `S-U` and `S-D` are significant calls
- `U` and `D` are directional but not significant under the stricter original significance threshold

For original protein direction, the source `Significance` label should be treated as the primary direction annotation.

### Original protein effect size

- The canonical original fold-change quantity is `Log2FC(median)`.
- `Log2FC(mean)` may appear in tables, but original significance labeling and canonical original fold-change interpretation should be tied to `Log2FC(median)`.

### Pathway significance

For the original pathway GSEA results in:

- `E:\lab\cptac-prostate\runs\20260416_GSEA_KEGG\gsea_full_results.tsv`

current downstream purity-validation comparisons use:

- original pathway significance: `GSEA FDR q-val < 0.05`
- adjusted pathway significance: `GSEA FDR q-val < 0.05`

## Purity-validation candidate protein modes

The `purity_validation` workflow supports four candidate-protein source modes:

1. `original_significant`
   - proteins with original `Significance` equal to `S-U` or `S-D`
2. `original_significant_leading_edge`
   - proteins with original `Significance` equal to `S-U` or `S-D` and overlapping leading-edge genes from originally significant pathways
3. `preset_markers`
   - fixed 17-protein prior marker panel
4. `custom_list`
   - proteins supplied from a file or config list

### Preset 17 markers

Stromal / ECM / CAF-like:

- `POSTN, COMP, THBS4, SFRP4, COL1A1, COL1A2, DCN, LUM, ACTA2, TAGLN`

Tumor / epithelial-like:

- `EPCAM, AMACR, ENTPD5, GOLM1, KRT8, KRT18, FOLH1`

Rule:

- `preset_markers` is a fixed prior marker panel, not an original-significance filter.
- These markers can still be annotated by original `S-U` / `S-D` labels for tumor-up vs NAT-up split plots.

## Current organized output layout

Current purity-validation runs were organized under:

- `E:\lab\cptac-prostate\runs\20260416_GSEA_purity\01_original_significant`
- `E:\lab\cptac-prostate\runs\20260416_GSEA_purity\02_original_significant_leading_edge`
- `E:\lab\cptac-prostate\runs\20260416_GSEA_purity\03_preset_markers`

Typical candidate-level outputs include:

- `candidate_protein_selection.csv`
- `plots/candidate_beta_before_after_adjustment.png`
- `candidate_original_SU_tumor_up.csv`
- `candidate_original_SD_nat_up.csv`
- `plots/candidate_beta_original_SU_tumor_up.png`
- `plots/candidate_beta_original_SD_nat_up.png`

## Pathway purity workflow

The repository now also includes a dedicated CLI task:

- `pathway_purity_validation`

Primary intended use:

- start from an existing preranked GSEA run and its config
- identify originally significant positive-NES and negative-NES pathways
- rerun pathway testing after tumor-purity adjustment
- quantify whether pathway-level signal tracks tumor purity

Current reference input pair:

- original GSEA results: `E:\lab\cptac-prostate\runs\20260416_GSEA\gsea_full_results.tsv`
- original GSEA config: `E:\lab\cptac-prostate\runs\20260416_GSEA\config_prerank_gsea.ini`

Current organized output directory:

- `E:\lab\cptac-prostate\runs\20260416_GSEA_purity\pathways`

Key outputs include:

- `original_significant_pathways.csv`
- `original_significant_positive_NES_pathways.csv`
- `original_significant_negative_NES_pathways.csv`
- `pathway_sample_scores.csv`
- `pathway_purity_evaluation.csv`
- `pathway_gsea_comparison_original_vs_adjusted.csv`
- `pathway_purity_summary.md`

Pathway interpretation rules:

- positive NES means the pathway is enriched on the tumor-up side of the original ranked list
- negative NES means the pathway is enriched on the normal/NAT-up side of the original ranked list
- original pathway significance is defined by the source GSEA table using `FDR q-val < 0.05`
- adjusted pathway significance is evaluated with the same `FDR q-val < 0.05` threshold after rerunning preranked GSEA from the purity-adjusted model

Purity-sensitivity logic:

- a positive-NES pathway is purity-sensitive when its pathway score correlates positively with tumor purity-sensitive tumor signal and it loses significance after purity adjustment
- a negative-NES pathway is purity-sensitive when its pathway score correlates negatively with purity or stromal admixture and it loses significance after purity adjustment
- pathways retained after purity adjustment should not be dismissed as simple purity artifacts
