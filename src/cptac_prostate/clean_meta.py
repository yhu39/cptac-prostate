from __future__ import annotations

import configparser
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd


_STEP_OUTPUT_DIR: Path | None = None


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


def _get_step_names(config: configparser.ConfigParser) -> list[str]:
    if not config.has_section("steps"):
        return []

    step_names: list[str] = []
    for option in sorted(config.options("steps")):
        step_names.append(_strip_quotes(config.get("steps", option)))
    return step_names


def read_input_file1(input_path: Path) -> pd.DataFrame:
    return pd.read_csv(input_path)


def _normalize_gleason_grade(value: object) -> int | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    try:
        grade = int(float(text))
    except ValueError:
        return None

    if grade in {1, 2, 3, 4, 5}:
        return grade
    return None


def _format_pie_label(values: list[int]):
    total = sum(values)

    def _formatter(pct: float) -> str:
        count = int(round(pct * total / 100.0))
        return f"{pct:.1f}%\n(n={count})"

    return _formatter


def fix_sample_name(input_frame: pd.DataFrame) -> pd.DataFrame:
    """Placeholder for clean_meta step: fix_sample_name."""
    # C3L.05292.T -> C3L-05292_T
    input_frame["SampleID"] = input_frame["common_ID"].apply(
        lambda x: (lambda p: f"{p[0]}-{p[1]}_{p[2]}")(x.split("."))
    )
    return input_frame.copy()


def find_tumors(input_frame: pd.DataFrame) -> pd.DataFrame:
    """Placeholder for clean_meta step: find_tumors."""
    if _STEP_OUTPUT_DIR is None:
        msg = "Output directory is not configured for find_tumors."
        raise ValueError(msg)

    tumor_frame = input_frame[
        (input_frame["FirstCategory"] == "Sufficient Purity")
        & (input_frame["Tissuetype"] == "tumor")
    ].copy()

    _STEP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tumor_output_path = _STEP_OUTPUT_DIR / "samples_tumor_sufficient_purity.tsv"
    tumor_log_path = _STEP_OUTPUT_DIR / "samples_tumor_sufficient_purity.log"

    tumor_frame.to_csv(tumor_output_path, sep="\t", index=False)
    tumor_log_path.write_text(f"tumors size: {len(tumor_frame)}\n", encoding="utf-8")

    return input_frame.copy()


def find_normals(input_frame: pd.DataFrame) -> pd.DataFrame:
    """Placeholder for clean_meta step: find_normals."""
    if _STEP_OUTPUT_DIR is None:
        msg = "Output directory is not configured for find_normals."
        raise ValueError(msg)

    normal_frame = input_frame[input_frame["Tissuetype"] == "normal"].copy()

    _STEP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    normal_output_path = _STEP_OUTPUT_DIR / "samples_normal_.tsv"
    normal_log_path = _STEP_OUTPUT_DIR / "samples_normal.log"

    normal_frame.to_csv(normal_output_path, sep="\t", index=False)
    normal_log_path.write_text(f"normals size: {len(normal_frame)}\n", encoding="utf-8")

    return input_frame.copy()


def find_grades(input_frame: pd.DataFrame) -> pd.DataFrame:
    """Placeholder for clean_meta step: find_grades."""
    if _STEP_OUTPUT_DIR is None:
        msg = "Output directory is not configured for find_grades."
        raise ValueError(msg)

    tumor_frame = input_frame[
        (input_frame["FirstCategory"] == "Sufficient Purity")
        & (input_frame["Tissuetype"] == "tumor")
    ].copy()

    sample_column = "SampleID" if "SampleID" in tumor_frame.columns else "common_ID"
    grade_dict: dict[str, int] = {}

    for _, row in tumor_frame.iterrows():
        grade = _normalize_gleason_grade(row["BCR_Gleason_Grade"])
        if grade is None:
            continue
        grade_dict[str(row[sample_column])] = grade

    _STEP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grade_output_path = _STEP_OUTPUT_DIR / "BCR_Gleason_Grade.joblib"
    grade_table_output_path = _STEP_OUTPUT_DIR / "BCR_Gleason_Grade.tsv"
    piechart_output_path = _STEP_OUTPUT_DIR / "BCR_Gleason_Grade_PieChart.png"

    joblib.dump(grade_dict, grade_output_path)
    pd.DataFrame(
        {"sample": list(grade_dict.keys()), "bcr_gleason_grade": list(grade_dict.values())}
    ).to_csv(grade_table_output_path, sep="\t", index=False)

    grade_counts = pd.Series(grade_dict).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 6))
    if grade_counts.empty:
        ax.text(0.5, 0.5, "No valid Gleason grades", ha="center", va="center")
        ax.axis("off")
    else:
        ax.pie(
            grade_counts.values,
            labels=[f"Gleason {grade}" for grade in grade_counts.index],
            autopct=_format_pie_label(grade_counts.values.tolist()),
            startangle=90,
        )
        ax.set_title("BCR Gleason Grade Distribution")
    fig.savefig(piechart_output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return input_frame.copy()


STEP_FUNCTIONS = {
    "fix_sample_name": fix_sample_name,
    "find_tumors": find_tumors,
    "find_normals": find_normals,
    "find_grades": find_grades,
}


def write_output_file1(output_frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_frame.to_csv(output_path, sep="\t", index=False)


def run_clean_meta(config_path: Path) -> Path:
    global _STEP_OUTPUT_DIR

    config = _read_config(config_path)
    input_dir = _get_required_path(config, "input", "input_dir")
    input_file1 = _get_required_path(config, "input", "input_file1")
    output_dir = _get_required_path(config, "output", "output_dir")
    output_file1 = _get_required_path(config, "output", "output_file1")

    input_path = input_dir / input_file1
    output_path = output_dir / output_file1

    current_frame = read_input_file1(input_path)
    _STEP_OUTPUT_DIR = output_dir

    for step_name in _get_step_names(config):
        print(step_name)
        if step_name not in STEP_FUNCTIONS:
            msg = f"Unsupported clean_meta step: {step_name}"
            raise ValueError(msg)
        current_frame = STEP_FUNCTIONS[step_name](current_frame)

    write_output_file1(current_frame, output_path)
    print(f"clean_meta output written to {output_path}")
    return output_path
