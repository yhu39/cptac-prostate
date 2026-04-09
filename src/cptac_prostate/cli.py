from __future__ import annotations

import argparse
import configparser
from pathlib import Path

from cptac_prostate.clean_data import run_clean_data
from cptac_prostate.clean_meta import run_clean_meta
from cptac_prostate.global_diff import run_global_diff
from cptac_prostate.global_diff_pairwise import run_global_diff_pairwise
from cptac_prostate.global_diff_summary import run_global_diff_summary
from cptac_prostate.global_sample_match import run_global_sample_match
from cptac_prostate.global_sample_cv import run_global_sample_cv
from cptac_prostate.phospho_combine_py import run_phospho_combine_py
from cptac_prostate.phospho_diff import run_phospho_diff
from cptac_prostate.phospho_gene_summary import run_phospho_gene_summary
from cptac_prostate.phospho_diff_summary import run_phospho_diff_summary
from cptac_prostate.phospho_remove_py import run_phospho_remove_py


DEFAULT_CONFIG_PATH = Path(r"E:\lab\cptac-prostate\runs\20260406_clean_meta\config.ini")


def _strip_quotes(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]
    return value


def load_config(config_path: Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser()
    parser.read(config_path)
    return parser


def get_task_name(config: configparser.ConfigParser) -> str:
    if not config.has_section("task") or not config.has_option("task", "name"):
        msg = "Config file is missing [task] name."
        raise ValueError(msg)
    return _strip_quotes(config.get("task", "name"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prostate",
        description="Run CPTAC prostate workflows from a config.ini file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to config.ini (default: {DEFAULT_CONFIG_PATH}).",
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Print the resolved task name from the config and exit.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config)
    task_name = get_task_name(config)

    if args.show_config:
        print(f"config={args.config}")
        print(f"task={task_name}")
        return 0

    if task_name == "clean_meta":
        run_clean_meta(args.config)
        return 0
    elif task_name == "clean_data":
        run_clean_data(args.config)
        return 0
    elif task_name == "global_diff":
        run_global_diff(args.config)
        return 0
    elif task_name == "global_diff_pairwise":
        run_global_diff_pairwise(args.config)
        return 0
    elif task_name == "phospho_diff":
        run_phospho_diff(args.config)
        return 0
    elif task_name == "phospho_diff_summary":
        run_phospho_diff_summary(args.config)
        return 0
    elif task_name == "phospho_gene_summary":
        run_phospho_gene_summary(args.config)
        return 0
    elif task_name == "phospho_remove_pY":
        run_phospho_remove_py(args.config)
        return 0
    elif task_name == "phospho_combine_pY":
        run_phospho_combine_py(args.config)
        return 0
    elif task_name == "global_diff_summary":
        run_global_diff_summary(args.config)
        return 0
    elif task_name == "global_sample_match":
        run_global_sample_match(args.config)
        return 0
    elif task_name == "global_sample_cv":
        run_global_sample_cv(args.config)
        return 0

    msg = f"Unsupported task in config: {task_name}"
    raise ValueError(msg)


if __name__ == "__main__":
    raise SystemExit(main())
