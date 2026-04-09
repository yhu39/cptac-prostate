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


def _get_optional_path(
    config: configparser.ConfigParser,
    section: str,
    option: str,
) -> Path | None:
    if not config.has_section(section) or not config.has_option(section, option):
        return None
    return Path(_strip_quotes(config.get(section, option)))


def _normalize_key(value: object) -> str:
    return str(value).strip()


def _load_sample_map(sample_match_path: Path) -> dict[str, str]:
    sample_df = pd.read_excel(sample_match_path)
    required_columns = {"sample number", "sample ID"}
    missing_columns = required_columns.difference(sample_df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        msg = f"Sample match file is missing required columns: {missing}"
        raise ValueError(msg)

    sample_df = sample_df.loc[:, ["sample number", "sample ID"]].dropna()
    sample_df = sample_df.astype(str)

    sample_map: dict[str, str] = {}
    for _, row in sample_df.iterrows():
        sample_number = _normalize_key(row["sample number"])
        sample_id = _normalize_key(row["sample ID"])
        if not sample_number or not sample_id:
            continue
        sample_map[sample_number] = sample_id

    if not sample_map:
        msg = "Sample match file did not contain any usable sample mappings."
        raise ValueError(msg)

    return sample_map


def _build_renamed_columns(columns: pd.Index, sample_map: dict[str, str]) -> list[str]:
    renamed_columns = [sample_map.get(_normalize_key(column), str(column)) for column in columns]
    duplicated_columns = pd.Index(renamed_columns)[pd.Index(renamed_columns).duplicated()].unique().tolist()
    if duplicated_columns:
        duplicates = ", ".join(duplicated_columns)
        msg = f"Renaming would create duplicate columns: {duplicates}"
        raise ValueError(msg)
    return renamed_columns


def _resolve_output_path(
    config: configparser.ConfigParser,
    output_dir: Path,
    global_path: Path,
) -> Path:
    output_file = _get_optional_path(config, "output", "output_file")
    if output_file is None:
        output_file = _get_optional_path(config, "output", "output_file1")
    if output_file is not None:
        return output_dir / output_file

    stem = global_path.stem
    suffix = global_path.suffix or ".tsv"
    return output_dir / f"{stem}-sample_id_matched{suffix}"


def run_global_sample_match(config_path: Path) -> Path:
    config = _read_config(config_path)
    input_dir = _get_required_path(config, "input", "input_dir")
    global_path = _get_required_path(config, "input", "global_path")
    sample_match_dir = _get_required_path(config, "input", "sample_match_dir")
    sample_match_path = _get_required_path(config, "input", "sample_match_path")
    output_dir = _get_required_path(config, "output", "output_dir")

    data_path = input_dir / global_path
    mapping_path = sample_match_dir / sample_match_path
    output_path = _resolve_output_path(config, output_dir, global_path)

    data = pd.read_csv(data_path, sep="\t")
    sample_map = _load_sample_map(mapping_path)
    renamed_columns = _build_renamed_columns(data.columns, sample_map)

    matched_count = sum(
        1 for original_column, renamed_column in zip(data.columns, renamed_columns) if str(original_column) != renamed_column
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    renamed_data = data.copy()
    renamed_data.columns = renamed_columns
    renamed_data.to_csv(output_path, sep="\t", index=False)

    print(f"Matched {matched_count} columns using sample number -> sample ID.")
    print(f"global_sample_match output written to {output_path}")
    return output_path
