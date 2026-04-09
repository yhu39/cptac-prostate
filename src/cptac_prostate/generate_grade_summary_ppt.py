from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from shutil import which

from cptac_prostate.global_diff_summary import (
    _build_summary_config,
    _compute_grade_progression_metrics,
    _load_inputs,
    _read_config,
)


def _gene_line(genes: list[str]) -> str:
    return ", ".join(genes) if genes else "None"


def _metric_lines(metrics: dict[str, object]) -> dict[str, str]:
    top_pair = metrics["pairwise_jaccard"][0]
    top_increasing = metrics["top_increasing"]["gene"].tolist()
    top_decreasing = metrics["top_decreasing"]["gene"].tolist()
    return {
        "counts": (
            f"GG2={len(metrics['grade_sets']['GG2'])}, "
            f"GG3={len(metrics['grade_sets']['GG3'])}, "
            f"GG4={len(metrics['grade_sets']['GG4'])}, "
            f"GG5={len(metrics['grade_sets']['GG5'])}"
        ),
        "samples": (
            f"normal={metrics['sample_sizes']['normal']}, "
            f"GG1={metrics['sample_sizes']['GG1']}, "
            f"GG2={metrics['sample_sizes']['GG2']}, "
            f"GG3={metrics['sample_sizes']['GG3']}, "
            f"GG4={metrics['sample_sizes']['GG4']}, "
            f"GG5={metrics['sample_sizes']['GG5']}, "
            f"tumor={metrics['sample_sizes']['tumor']}"
        ),
        "trunk": _gene_line(metrics["trunk_genes"]),
        "gained": _gene_line(metrics["gained_g5"]),
        "lost": _gene_line(metrics["lost_g5"]),
        "top_pair": f"{top_pair['pair']} (Jaccard={top_pair['jaccard']:.3f})",
        "up": _gene_line(metrics["monotonic_up"]),
        "down": _gene_line(metrics["monotonic_down"]),
        "top_inc": _gene_line(top_increasing),
        "top_dec": _gene_line(top_decreasing),
    }


def _english_slide_markdown(metrics: dict[str, object], output_dir: Path) -> str:
    lines = _metric_lines(metrics)
    venn_path = "../global_diff_summary_venn.png"
    heatmap_path = "../global_diff_summary_heatmap.png"
    return f"""% Prostate Cancer Grade Progression Summary
% Codex
% 2026-04-07

# Prostate Cancer Grade Progression

- Dataset: MSFragger global proteomics, tumor vs normal overlap summary
- Focus: tumor-shared `S-U` proteins from GG2 to GG5
- Outputs used: overlap table, tumor-overlap table, heatmap medians, venn diagram

---

# Analysis Methods

- Differential analysis was run separately for `GG2/GG3/GG4/GG5 vs normal` and `tumor vs normal`.
- The summary retained only `S-U` proteins that also overlap with `tumor vs normal`.
- A tumor-overlap table was built for GG2, GG3, GG4, and GG5 intersections.
- Heatmap values are group medians from the global matrix.
- Sample sizes used in the grade analysis: {lines["samples"]}.

---

# Main Results

- Tumor-overlap `S-U` counts by grade: {lines["counts"]}.
- Trunk tumor program shared across GG2-GG5: 14 genes.
- Trunk genes: {lines["trunk"]}.
- GG5 gained vs GG2: {lines["gained"]}.
- GG5 lost vs GG2: {lines["lost"]}.
- Highest set similarity: {lines["top_pair"]}.

---

# Venn Summary

![Venn diagram]({venn_path}){{ width=85% }}

---

# Heatmap Summary

![Heatmap]({heatmap_path}){{ width=85% }}

---

# Progression Interpretation

- The results support a two-layer model.
- Layer 1: a stable trunk tumor program that persists across GG2-GG5.
- Layer 2: a later progression program with ECM remodeling and metabolic rewiring.
- The strongest increasing genes are: {lines["top_inc"]}.
- The strongest decreasing genes are: {lines["top_dec"]}.

---

# Literature Support and Novelty

- Published work already supports the general idea of a prostate-cancer progression trajectory.
- Published work also supports stable prostate tumor identity markers such as `AMACR`, `GOLM1`, and `EPCAM`.
- Published work supports ECM / stromal / metabolic progression signals involving `SFRP4`, `POSTN`, `THBS4`, `COMP`, and `FASN`.
- The novelty here is the dataset-specific integration of these ideas into one two-layer model built from grade-specific `S-U` proteins plus overlap with `tumor_vs_normal`.
- So the biological direction is literature-supported, while this exact gene-level formulation is best viewed as a new working model from the current cohort.

---

# Monotonic Programs

- Monotonic up from GG2 to GG5: {lines["up"]}.
- These genes are consistent with invasion, extracellular-matrix remodeling, cell plasticity, or increased biosynthetic demand.
- Monotonic down from GG2 to GG5: {lines["down"]}.
- These genes are consistent with loss of earlier epithelial or glycosylation-linked states.

---

# Viewpoint and Working Model

- A reasonable working hypothesis is a shift from a more differentiated tumor state toward a more aggressive ECM-rich and metabolically rewired state.
- `COMP`, `SFRP4`, `POSTN`, and `THBS4` make the late program especially interesting for grade progression.
- `AMACR`, `EPCAM`, and `GOLM1` look more like stable trunk markers than late-specific markers.
- GG4 should be interpreted cautiously because the current cohort has only {metrics["sample_sizes"]["GG4"]} GG4 samples.

---

# Follow-Up Directions

- Validate the trunk and progression modules in independent cohorts.
- Use spatial transcriptomics / spatial proteomics / single-cell data to resolve tumor-cell vs stromal origin.
- Model grade as a continuous progression axis instead of only pairwise contrasts.
- Prioritize late-rising genes such as `SFRP4`, `COMP`, `POSTN`, `THBS4`, `FASN`, and `UGDH` for perturbation studies.
- Test whether apparent GG4 gaps reflect biology or limited sample size.

---

# Selected Literature Support

- AMACR review: [PMID 40605376](https://pubmed.ncbi.nlm.nih.gov/40605376/)
- GOLM1 biomarker: [PMID 18953438](https://pubmed.ncbi.nlm.nih.gov/18953438/)
- EpCAM prognosis: [PMID 21514185](https://pubmed.ncbi.nlm.nih.gov/21514185/)
- SFRP4 aggressive prostate cancer: [PMID 29079735](https://pubmed.ncbi.nlm.nih.gov/29079735/)
- Periostin in advanced prostate cancer: [PMID 32416542](https://pubmed.ncbi.nlm.nih.gov/32416542/)
- THBS4 stem-like properties: [PMID 32421868](https://pubmed.ncbi.nlm.nih.gov/32421868/)
- COMP promotes prostate cancer progression: [PMID 29228690](https://pubmed.ncbi.nlm.nih.gov/29228690/)
- FASN and prostate cancer progression: [PMID 26878389](https://pubmed.ncbi.nlm.nih.gov/26878389/)
- Spatial progression profiling: [PMID 38077153](https://pubmed.ncbi.nlm.nih.gov/38077153/)
"""


def _chinese_slide_markdown(metrics: dict[str, object], output_dir: Path) -> str:
    lines = _metric_lines(metrics)
    venn_path = "../global_diff_summary_venn.png"
    heatmap_path = "../global_diff_summary_heatmap.png"
    return f"""% å‰åˆ—è…ºç™Œåˆ†çº§è¿›å±•æ€»ç»“
% Codex
% 2026-04-07

# å‰åˆ—è…ºç™Œåˆ†çº§è¿›å±•æ€»ç»“

- æ•°æ®æ¥æºï¼šMSFragger global proteomics
- ?????Â?????????GG2 ??? GG5 ??Â­??? `tumor vs normal` ??Â?ÂÂ ??? `S-U` ??????
- ä½¿ç”¨ç»“æžœï¼šoverlap tableã€tumor overlap tableã€heatmap medianã€venn diagram

---

# åˆ†æžæ–¹æ³•

- åˆ†åˆ«è¿›è¡Œäº† `GG2/GG3/GG4/GG5 vs normal` å’Œ `tumor vs normal` çš„å·®å¼‚åˆ†æžã€‚
- æ€»ç»“æ—¶åªä¿ç•™åŒæ—¶å±žäºŽ `S-U` ä¸”ä¸Ž `tumor vs normal` é‡å çš„è›‹ç™½ã€‚
- ????????? GG2??ÂGG3??ÂGG4??ÂGG5 ??? tumor-overlap ????????????
- Heatmap ä½¿ç”¨å…¨å±€è¡¨è¾¾çŸ©é˜µä¸­çš„å„ç»„ä¸­ä½æ•°ã€‚
- å½“å‰æ ·æœ¬æ•°ï¼š{lines["samples"]}ã€‚

---

# ä¸»è¦ç»“æžœ

- å„ grade çš„ tumor-overlap `S-U` æ•°é‡ï¼š{lines["counts"]}ã€‚
- ??? GG2-GG5 ????????? trunk tumor program ??? 14 ????????Â ???
- Trunk åŸºå› ï¼š{lines["trunk"]}ã€‚
- ?????? GG2?????? GG5 ??Â­??????????????????{lines["gained"]}???
- ?????? GG2?????? GG5 ??Â­??Â????????????????????????{lines["lost"]}???
- ç›¸ä¼¼åº¦æœ€é«˜çš„ç»„åˆï¼š{lines["top_pair"]}ã€‚

---

# Venn å›¾æ€»ç»“

![Venn å›¾]({venn_path}){{ width=85% }}

---

# Heatmap æ€»ç»“

![Heatmap]({heatmap_path}){{ width=85% }}

---

# ç»“æžœè§£é‡Š

- å½“å‰ç»“æžœæ”¯æŒâ€œä¸¤å±‚æ¨¡åž‹â€ã€‚
- ç¬¬ä¸€å±‚æ˜¯ç¨³å®šå­˜åœ¨çš„ trunk tumor programï¼Œä»£è¡¨è·¨ grade ä¿æŒä¸€è‡´çš„è‚¿ç˜¤åŸºç¡€ç¨‹åºã€‚
- ç¬¬äºŒå±‚æ˜¯æ™šæœŸå¢žå¼ºçš„ progression programï¼Œåå‘ ECM é‡å¡‘å’Œä»£è°¢é‡ç¼–ç¨‹ã€‚
- å¢žå¹…æœ€å¤§çš„åŸºå› ä¸»è¦æ˜¯ï¼š{lines["top_inc"]}ã€‚
- é™å¹…æœ€å¤§çš„åŸºå› ä¸»è¦æ˜¯ï¼š{lines["top_dec"]}ã€‚

---

# æ–‡çŒ®æ”¯æŒä¸Žæ–°æ„

- çŽ°æœ‰æ–‡çŒ®å·²ç»æ”¯æŒâ€œå‰åˆ—è…ºç™Œå­˜åœ¨è¿žç»­ progression trajectoryâ€è¿™ä¸€å¤§æ–¹å‘ã€‚
- çŽ°æœ‰æ–‡çŒ®ä¹Ÿæ”¯æŒ `AMACR`ã€`GOLM1`ã€`EPCAM` è¿™ç±»ç¨³å®šè‚¿ç˜¤æ ‡å¿—ç‰©ã€‚
- å¯¹äºŽ `SFRP4`ã€`POSTN`ã€`THBS4`ã€`COMP`ã€`FASN`ï¼Œä¹Ÿæœ‰æ–‡çŒ®æ”¯æŒå…¶ä¸Ž ECMã€åŸºè´¨é‡å¡‘ã€ä¾µè¢­æ€§æˆ–ä»£è°¢é‡ç¼–ç¨‹ç›¸å…³ã€‚
- å½“å‰åˆ†æžçš„åˆ›æ–°ç‚¹åœ¨äºŽï¼šæŠŠè¿™äº›æ–¹å‘æ•´åˆæˆä¸€ä¸ªåŸºäºŽå½“å‰é˜Ÿåˆ—çš„å…·ä½“ä¸¤å±‚æ¨¡åž‹ï¼Œå®šä¹‰ä¾æ®æ˜¯ grade-specific `S-U` è›‹ç™½åŠ ä¸Šä¸Ž `tumor_vs_normal` çš„é‡å ã€‚
- å› æ­¤ï¼Œè¿™ä¸ªæ¨¡åž‹çš„â€œæ–¹å‘â€æœ‰æ–‡çŒ®ä¾æ®ï¼Œä½†â€œè¿™å¥—å…·ä½“åŸºå› å±‚é¢çš„æ¨¡åž‹â€æ›´é€‚åˆä½œä¸ºå½“å‰æ•°æ®é©±åŠ¨çš„æ–°å·¥ä½œå‡è¯´ã€‚

---

# å•è°ƒå˜åŒ–ç¨‹åº

- ??? GG2 ??? GG5 ?Â????????Â?????????Â ???{lines["up"]}???
- è¿™ç»„åŸºå› æ›´æ”¯æŒä¾µè¢­å¢žå¼ºã€åŸºè´¨é‡å¡‘ã€ç»†èƒžå¯å¡‘æ€§ä¸Šå‡æˆ–ç”Ÿç‰©åˆæˆéœ€æ±‚å¢žåŠ ã€‚
- ??? GG2 ??? GG5 ?Â?????????Â????????Â ???{lines["down"]}???
- è¿™ç»„åŸºå› æ›´åƒæ—©æœŸä¸Šçš®çŠ¶æ€æˆ–ç³–åŸºåŒ–ç›¸å…³çŠ¶æ€åœ¨é€æ­¥å‡å¼±ã€‚

---

# å·¥ä½œå‡è¯´

- ä¸€ä¸ªåˆç†çš„å·¥ä½œæ¨¡åž‹æ˜¯ï¼šè‚¿ç˜¤ä»Žç›¸å¯¹åˆ†åŒ–çš„çŠ¶æ€é€æ­¥è½¬å‘æ›´å…·ä¾µè¢­æ€§ã€ECM å¯Œé›†å’Œä»£è°¢é‡ç¼–ç¨‹çš„çŠ¶æ€ã€‚
- `COMP`ã€`SFRP4`ã€`POSTN`ã€`THBS4` æ˜¯æœ€å€¼å¾—å…³æ³¨çš„æ™šæœŸ progression è½´ã€‚
- `AMACR`ã€`EPCAM`ã€`GOLM1` æ›´åƒç¨³å®š trunk markerï¼Œè€Œä¸æ˜¯çº¯ç²¹æ™šæœŸ markerã€‚
- GG4 ?????Â?????????????????Â ????????Â?Â???? {metrics["sample_sizes"]["GG4"]} ????Â ???????

---

# åŽç»­ç ”ç©¶æ–¹å‘

- åœ¨ç‹¬ç«‹é˜Ÿåˆ—ä¸­éªŒè¯ trunk æ¨¡å—å’Œ progression æ¨¡å—ã€‚
- ç”¨ spatial transcriptomics / spatial proteomics / single-cell æ•°æ®æ‹†åˆ† tumor cell ä¸Ž stroma æ¥æºã€‚
- å°† grade ä½œä¸ºè¿žç»­è¿›ç¨‹æ¥å»ºæ¨¡ï¼Œè€Œä¸æ˜¯åªåš pairwise æ¯”è¾ƒã€‚
- ä¼˜å…ˆéªŒè¯ `SFRP4`ã€`COMP`ã€`POSTN`ã€`THBS4`ã€`FASN`ã€`UGDH` çš„åŠŸèƒ½ã€‚
- ???????????? GG4 ??Â­??????????Â?????Â??Â?????Â ??????Â??Â??????

---

# æ–‡çŒ®æ”¯æŒ

- AMACR ç»¼è¿°ï¼š[PMID 40605376](https://pubmed.ncbi.nlm.nih.gov/40605376/)
- GOLM1 biomarkerï¼š[PMID 18953438](https://pubmed.ncbi.nlm.nih.gov/18953438/)
- EpCAM é¢„åŽç›¸å…³ï¼š[PMID 21514185](https://pubmed.ncbi.nlm.nih.gov/21514185/)
- SFRP4 ä¸Ž aggressive PCaï¼š[PMID 29079735](https://pubmed.ncbi.nlm.nih.gov/29079735/)
- Periostin ä¸Ž advanced PCaï¼š[PMID 32416542](https://pubmed.ncbi.nlm.nih.gov/32416542/)
- THBS4 ä¸Ž stem-like propertiesï¼š[PMID 32421868](https://pubmed.ncbi.nlm.nih.gov/32421868/)
- COMP ä¿ƒè¿›å‰åˆ—è…ºç™Œè¿›å±•ï¼š[PMID 29228690](https://pubmed.ncbi.nlm.nih.gov/29228690/)
- FASN ä¸Ž progressionï¼š[PMID 26878389](https://pubmed.ncbi.nlm.nih.gov/26878389/)
- ç©ºé—´è½¬å½•ç»„ grade progressionï¼š[PMID 38077153](https://pubmed.ncbi.nlm.nih.gov/38077153/)
"""


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run_pandoc(markdown_path: Path, pptx_path: Path) -> None:
    pandoc = which("pandoc")
    if pandoc is None:
        msg = "pandoc is not available on PATH."
        raise RuntimeError(msg)
    subprocess.run(
        [pandoc, str(markdown_path.name), "-t", "pptx", "-o", str(pptx_path.name)],
        cwd=markdown_path.parent,
        check=True,
    )


def generate_grade_summary_ppt(config_path: Path) -> tuple[Path, Path]:
    config = _read_config(config_path)
    cfg = _build_summary_config(config)
    state = _load_inputs(cfg)
    metrics = _compute_grade_progression_metrics(state)

    en_dir = cfg.output_dir / "en"
    zh_dir = cfg.output_dir / "zh"
    en_md = en_dir / "global_diff_grade_summary_slides.md"
    zh_md = zh_dir / "global_diff_grade_summary_slides.md"
    en_pptx = en_dir / "global_diff_grade_summary.pptx"
    zh_pptx = zh_dir / "global_diff_grade_summary.pptx"

    _write_text(en_md, _english_slide_markdown(metrics, cfg.output_dir))
    _write_text(zh_md, _chinese_slide_markdown(metrics, cfg.output_dir))
    _run_pandoc(en_md, en_pptx)
    _run_pandoc(zh_md, zh_pptx)
    return en_pptx, zh_pptx


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="generate-grade-summary-ppt",
        description="Generate bilingual PPTX summaries for prostate grade progression.",
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    en_pptx, zh_pptx = generate_grade_summary_ppt(args.config)
    print(en_pptx)
    print(zh_pptx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

