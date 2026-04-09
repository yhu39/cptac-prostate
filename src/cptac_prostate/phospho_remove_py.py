from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


SITE_PATTERN = re.compile(r"(^|;)\s*[^;]+_([STY])(\d+)(?=;|$)")


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


def _get_optional_value(
    config: configparser.ConfigParser,
    section: str,
    option: str,
    default: str,
) -> str:
    if not config.has_section(section) or not config.has_option(section, option):
        return default
    return _strip_quotes(config.get(section, option))


def _get_step_names(config: configparser.ConfigParser) -> list[str]:
    if not config.has_section("steps"):
        return []

    step_names: list[str] = []
    for option in sorted(config.options("steps")):
        step_names.append(_strip_quotes(config.get("steps", option)))
    return step_names


def _infer_sep(path: Path) -> str:
    return "\t" if path.suffix.lower() in {".tsv", ".txt"} else ","


def _detect_site_column(columns: pd.Index, preferred: str | None) -> str:
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(["gene_sites", "gene_site"])

    for column in dict.fromkeys(candidates):
        if column in columns:
            return column

    msg = "Could not find a phosphosite column. Expected one of: gene_sites, gene_site."
    raise ValueError(msg)


def _classify_row(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    matches = list(SITE_PATTERN.finditer(text))
    if not matches:
        msg = f"Unable to parse phosphosite information from value: {text!r}"
        raise ValueError(msg)

    amino_acids = {match.group(2) for match in matches}
    return "Y" if "Y" in amino_acids else "ST"


def _build_output_path(output_dir: Path, input_path: Path, suffix: str) -> Path:
    ext = input_path.suffix or ".tsv"
    return output_dir / f"{input_path.stem}_{suffix}{ext}"


@dataclass
class PhosphoRemovePYConfig:
    input_dir: Path
    phospho_path: Path
    output_dir: Path
    site_column: str | None = None
    st_output_name: str | None = None
    y_output_name: str | None = None
    summary_name: str = "phospho_remove_pY_summary.txt"

    @property
    def input_path(self) -> Path:
        return self.input_dir / self.phospho_path


@dataclass
class PhosphoRemovePYState:
    cfg: PhosphoRemovePYConfig
    data: pd.DataFrame | None = None
    site_column: str | None = None
    st_data: pd.DataFrame | None = None
    y_data: pd.DataFrame | None = None
    summary_path: Path | None = None


def read_phospho(state: PhosphoRemovePYState) -> PhosphoRemovePYState:
    data = pd.read_csv(state.cfg.input_path, sep=_infer_sep(state.cfg.input_path))
    site_column = _detect_site_column(data.columns, state.cfg.site_column)
    state.data = data
    state.site_column = site_column
    return state


def remove_pY(state: PhosphoRemovePYState) -> PhosphoRemovePYState:
    if state.data is None or state.site_column is None:
        msg = "Phospho table has not been loaded."
        raise ValueError(msg)

    labels = state.data[state.site_column].map(_classify_row)
    state.st_data = state.data.loc[labels == "ST"].copy()
    state.y_data = state.data.loc[labels == "Y"].copy()
    return state


def write_phospho(state: PhosphoRemovePYState) -> PhosphoRemovePYState:
    if state.st_data is None or state.y_data is None:
        msg = "Phospho rows have not been split."
        raise ValueError(msg)

    state.cfg.output_dir.mkdir(parents=True, exist_ok=True)
    st_output_path = (
        state.cfg.output_dir / state.cfg.st_output_name
        if state.cfg.st_output_name
        else _build_output_path(state.cfg.output_dir, state.cfg.phospho_path, "ST_only")
    )
    y_output_path = (
        state.cfg.output_dir / state.cfg.y_output_name
        if state.cfg.y_output_name
        else _build_output_path(state.cfg.output_dir, state.cfg.phospho_path, "Y_only")
    )

    state.st_data.to_csv(st_output_path, sep=_infer_sep(st_output_path), index=False)
    state.y_data.to_csv(y_output_path, sep=_infer_sep(y_output_path), index=False)
    return state


def get_summary(state: PhosphoRemovePYState) -> PhosphoRemovePYState:
    if state.data is None or state.st_data is None or state.y_data is None or state.site_column is None:
        msg = "Split results are not available."
        raise ValueError(msg)

    st_output_path = (
        state.cfg.output_dir / state.cfg.st_output_name
        if state.cfg.st_output_name
        else _build_output_path(state.cfg.output_dir, state.cfg.phospho_path, "ST_only")
    )
    y_output_path = (
        state.cfg.output_dir / state.cfg.y_output_name
        if state.cfg.y_output_name
        else _build_output_path(state.cfg.output_dir, state.cfg.phospho_path, "Y_only")
    )
    summary_path = state.cfg.output_dir / state.cfg.summary_name

    summary_lines = [
        f"input_path={state.cfg.input_path}",
        f"site_column={state.site_column}",
        f"total_rows={len(state.data)}",
        f"st_rows={len(state.st_data)}",
        f"y_rows={len(state.y_data)}",
        f"st_output={st_output_path}",
        f"y_output={y_output_path}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    state.summary_path = summary_path
    return state


STEP_FUNCTIONS = {
    "read_phospho": read_phospho,
    "remove_pY": remove_pY,
    "write_phospho": write_phospho,
    "get_summary": get_summary,
}


def _build_phospho_remove_py_config(config: configparser.ConfigParser) -> PhosphoRemovePYConfig:
    return PhosphoRemovePYConfig(
        input_dir=_get_required_path(config, "input", "input_dir"),
        phospho_path=_get_required_path(config, "input", "phospho_path"),
        output_dir=_get_required_path(config, "output", "output_dir"),
        site_column=_get_optional_value(config, "input", "site_column", "") or None,
        st_output_name=_get_optional_value(config, "output", "st_output_name", "") or None,
        y_output_name=_get_optional_value(config, "output", "y_output_name", "") or None,
        summary_name=_get_optional_value(
            config,
            "output",
            "summary_name",
            "phospho_remove_pY_summary.txt",
        ),
    )


def run_phospho_remove_py(config_path: Path) -> tuple[Path, Path]:
    config = _read_config(config_path)
    cfg = _build_phospho_remove_py_config(config)
    state = PhosphoRemovePYState(cfg=cfg)

    for step_name in _get_step_names(config):
        print(step_name)
        if step_name not in STEP_FUNCTIONS:
            msg = f"Unsupported phospho_remove_pY step: {step_name}"
            raise ValueError(msg)
        state = STEP_FUNCTIONS[step_name](state)

    if state.st_data is None or state.y_data is None:
        msg = "The phospho_remove_pY workflow completed without producing split phospho tables."
        raise ValueError(msg)

    st_output_path = (
        cfg.output_dir / cfg.st_output_name
        if cfg.st_output_name
        else _build_output_path(cfg.output_dir, cfg.phospho_path, "ST_only")
    )
    y_output_path = (
        cfg.output_dir / cfg.y_output_name
        if cfg.y_output_name
        else _build_output_path(cfg.output_dir, cfg.phospho_path, "Y_only")
    )
    print(f"phospho_remove_pY ST output written to {st_output_path}")
    print(f"phospho_remove_pY Y output written to {y_output_path}")
    return st_output_path, y_output_path
