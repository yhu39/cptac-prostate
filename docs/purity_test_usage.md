# Purity Test Usage

Run the purity sensitivity workflow with:

```powershell
C:\Users\yhu39\AppData\Local\anaconda3\envs\prostate\python.exe -m cptac_prostate.cli --config E:\lab\cptac-prostate\runs\20260415_purity_test\config.ini
```

Expected config fields:

- `[input] input_dir`, `global_path`, `meta_dir`, `meta_path`
- `[output] output_dir`
- `[task] name = "purity_test"`
- optional `[settings]`
  - `feature_column`
  - `sample_id_column`
  - `group_column`
  - `tumor_label`
  - `normal_label`
  - `purity_column`
  - `grade_column`
  - `candidate_table_path`
  - `candidate_gene_column`
  - `candidate_genes`
  - `high_purity_threshold`
  - `strong_purity_rho_threshold`
  - `strong_purity_fdr_threshold`
  - `major_attenuation_threshold`
  - `modest_attenuation_threshold`
  - `top_n_per_class`

Outputs:

- `purity_test_results.tsv`
- `top_tumor_intrinsic.tsv`
- `top_purity_or_microenvironment_associated.tsv`
- `top_mixed.tsv`
- `purity_test_summary.md`
- `purity_test_data_summary.tsv`
- `purity_test_candidate_genes.tsv`
- `purity_test_processing_notes.tsv`
- summary plots and per-protein panels in the output directory
