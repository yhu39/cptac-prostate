from __future__ import annotations

import configparser
from pathlib import Path

import pandas as pd


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
    return pd.read_csv(input_path, sep="\t")


def fix_sample_name(input_frame: pd.DataFrame) -> pd.DataFrame:
    """Placeholder for clean_data step: fix_sample_name."""
    return input_frame.copy()


def find_tumors(input_frame: pd.DataFrame) -> pd.DataFrame:
    """Placeholder for clean_data step: find_tumors."""
    return input_frame.copy()


def find_normals(input_frame: pd.DataFrame) -> pd.DataFrame:
    """Placeholder for clean_data step: find_normals."""
    return input_frame.copy()


def find_grades(input_frame: pd.DataFrame) -> pd.DataFrame:
    """Placeholder for clean_data step: find_grades."""
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


def run_clean_data(config_path: Path) -> Path:
    config = _read_config(config_path)
    input_dir = _get_required_path(config, "input", "input_dir")
    input_file1 = _get_required_path(config, "input", "input_file1")
    output_dir = _get_required_path(config, "output", "output_dir")
    output_file1 = _get_required_path(config, "output", "output_file1")

    input_path = input_dir / input_file1
    output_path = output_dir / output_file1

    current_frame = read_input_file1(input_path)

    for step_name in _get_step_names(config):
        print(step_name)
        if step_name not in STEP_FUNCTIONS:
            msg = f"Unsupported clean_data step: {step_name}"
            raise ValueError(msg)
        current_frame = STEP_FUNCTIONS[step_name](current_frame)

    write_output_file1(current_frame, output_path)
    print(f"clean_data output written to {output_path}")
    return output_path
