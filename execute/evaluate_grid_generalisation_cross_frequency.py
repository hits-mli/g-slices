import argparse
import csv
import re
import subprocess
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from gslice.dataset import (
    get_dataset_family_from_params,
    get_dataset_frequency_variants,
    get_dataset_name_from_params,
)
from gslice.utils.variables import get_relative_time_step


@dataclass(frozen=True)
class RunSpec:
    model_type: str
    run_dir: Path
    checkpoint_path: Path
    config_path: Path
    base_dataset: str
    dataset_name: str
    dataset_family: str
    freq: str
    label: str


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r") as fp:
        data = yaml.safe_load(fp)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}.")
    return data


def _resolve_run_config_path(run_dir: Path) -> Path | None:
    for candidate in (
        run_dir / "config.yaml",
        run_dir / "resolved_config.yaml",
        run_dir / ".hydra" / "config.yaml",
    ):
        if candidate.exists():
            return candidate
    return None


def _build_family_lookup() -> dict[str, str]:
    variants = get_dataset_frequency_variants()
    graph: dict[str, set[str]] = defaultdict(set)
    for base, related in variants.items():
        for other in related:
            graph[base].add(other)
            graph[other].add(base)

    family_lookup: dict[str, str] = {}
    visited: set[str] = set()
    for node in sorted(graph):
        if node in visited:
            continue
        component: list[str] = []
        queue = deque([node])
        visited.add(node)
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in sorted(graph[current]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        family_key = sorted(component)[0]
        for item in component:
            family_lookup[item] = family_key
    return family_lookup


def _candidate_model_roots(results_root: Path) -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []
    known_dirs = {
        "tsflow": "tsflow",
        "slice": "slice",
        "tsflow_run": "tsflow",
        "lcde": "lcde",
        # DSPD (tsdiff) baselines: one family dir per noise kernel — discovery
        # dedupes on (model_type, dataset, seed), so merging them under a single
        # dir would silently collapse the kernels.
        "tsdiff_gp": "tsdiff_gp",
        "tsdiff_ou": "tsdiff_ou",
        "tsdiff_gauss": "tsdiff_gauss",
    }
    for dirname, model_type in known_dirs.items():
        candidate = results_root / dirname
        if candidate.is_dir():
            candidates.append((candidate, model_type))
    for nested_root_name in ["grid_generalisation", "subsample_generalisation"]:
        nested_root = results_root / nested_root_name
        for dirname, model_type in known_dirs.items():
            candidate = nested_root / dirname
            if candidate.is_dir():
                candidates.append((candidate, model_type))
    if results_root.name in known_dirs and results_root.is_dir():
        candidates.append((results_root, known_dirs[results_root.name]))
    if results_root.name in {"grid_generalisation", "subsample_generalisation"} and results_root.is_dir():
        for dirname, model_type in known_dirs.items():
            candidate = results_root / dirname
            if candidate.is_dir():
                candidates.append((candidate, model_type))

    unique: list[tuple[Path, str]] = []
    seen: set[tuple[str, str]] = set()
    for path, model_type in candidates:
        key = (str(path.resolve()), model_type)
        if key in seen:
            continue
        seen.add(key)
        unique.append((path, model_type))
    return unique


def _discover_runs(results_root: Path) -> list[RunSpec]:
    runs: list[RunSpec] = []
    for model_root, model_type in _candidate_model_roots(results_root):
        for checkpoint_path in sorted(model_root.glob("*/best_checkpoint.ckpt")):
            run_dir = checkpoint_path.parent
            config_path = _resolve_run_config_path(run_dir)
            if config_path is None:
                continue
            config = _read_yaml(config_path)
            dataset_params = dict(config.get("dataset_params", {}))
            base_dataset = str(dataset_params.get("base_dataset", "")).strip()
            dataset_name = (
                get_dataset_name_from_params(dataset_params)
                if dataset_params
                else base_dataset or run_dir.name
            )
            dataset_family = (
                get_dataset_family_from_params(dataset_params)
                if dataset_params
                else base_dataset or dataset_name
            )
            freq = str(config.get("model_params", {}).get("freq", ""))
            label = f"{model_type}_{run_dir.name}"
            runs.append(
                RunSpec(
                    model_type=model_type,
                    run_dir=run_dir,
                    checkpoint_path=checkpoint_path,
                    config_path=config_path,
                    base_dataset=base_dataset or dataset_family,
                    dataset_name=dataset_name,
                    dataset_family=dataset_family,
                    freq=freq,
                    label=label,
                )
            )
    return runs


def _group_runs_by_family(runs: list[RunSpec]) -> dict[tuple[str, str], list[RunSpec]]:
    grouped: dict[tuple[str, str], list[RunSpec]] = defaultdict(list)
    for run in runs:
        family_key = run.base_dataset or run.dataset_family
        grouped[(run.model_type, family_key)].append(run)
    return grouped


def _sort_runs_for_eval(runs: list[RunSpec]) -> list[RunSpec]:
    def _sort_key(run: RunSpec) -> tuple[float, str]:
        try:
            step_hours = float(get_relative_time_step(run.freq))
        except Exception:
            step_hours = float("inf")
        return (step_hours, run.dataset_name)

    return sorted(runs, key=_sort_key)


def _py_list(items: list[str]) -> str:
    return repr(items)


def _display_dataset_name(family_key: str) -> str:
    name = str(family_key)
    name = re.sub(r"_pl\d+$", "", name)
    parts = [part for part in name.split("_") if part]
    return "".join(part[:1].upper() + part[1:] for part in parts)


def _display_model_name(model_type: str) -> str:
    return {
        "tsflow": "TSFlow",
        "slice": "SLiCE",
        "lcde": "LCDE",
        "tsdiff_gp": "DSPD-GP",
        "tsdiff_ou": "DSPD-OU",
        "tsdiff_gauss": "DSPD-Gauss",
    }.get(model_type, str(model_type))


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as fp:
        return list(csv.DictReader(fp))


def _build_markdown_table(rows: list[dict[str, str]]) -> str:
    train_freqs = sorted(
        {row["train_freq"] for row in rows},
        key=lambda value: (float(get_relative_time_step(value)), value),
    )
    eval_freqs = sorted(
        {row["eval_freq"] for row in rows},
        key=lambda value: (float(get_relative_time_step(value)), value),
    )
    values = {
        (row["eval_freq"], row["train_freq"]): float(row["CRPS"])
        for row in rows
    }

    lines = [
        "| test \\ train | " + " | ".join(train_freqs) + " |",
        "|" + "---|" * (len(train_freqs) + 1),
    ]
    for eval_freq in eval_freqs:
        row_values = [
            f"{values[(eval_freq, train_freq)]:.6f}" if (eval_freq, train_freq) in values else ""
            for train_freq in train_freqs
        ]
        lines.append(f"| {eval_freq} | " + " | ".join(row_values) + " |")
    return "\n".join(lines)


def _write_values_markdown(output_dir: Path) -> Path | None:
    grouped_rows: list[tuple[str, str, list[dict[str, str]]]] = []
    for model_dir in sorted(path for path in output_dir.iterdir() if path.is_dir()):
        for csv_path in sorted(model_dir.glob("*.csv")):
            rows = _read_csv_rows(csv_path)
            if not rows:
                continue
            grouped_rows.append((model_dir.name, csv_path.stem, rows))

    if not grouped_rows:
        return None

    lines = [
        "# Cross-Frequency Summary",
        "",
        f"Source: `{output_dir.relative_to(output_dir.parents[1])}`" if len(output_dir.parents) >= 2 else f"Source: `{output_dir}`",
        "",
    ]
    for model_type, family_key, rows in grouped_rows:
        lines.append(f"## {_display_dataset_name(family_key)} - {_display_model_name(model_type)}")
        lines.append("")
        lines.append(_build_markdown_table(rows))
        lines.append("")

    values_path = output_dir / "values.md"
    values_path.write_text("\n".join(lines).rstrip() + "\n")
    return values_path


def _build_eval_command(
    *,
    repo_root: Path,
    family_key: str,
    group_runs: list[RunSpec],
    device: str,
    num_samples: int,
    seed: int | None,
    output_dir: Path,
    fine_to_coarse_eval_adapter_override: Any,
    only_coarser_eval_inputs: bool,
    only_not_finer_eval_inputs: bool,
) -> tuple[list[str], Path]:
    model_type = group_runs[0].model_type
    def _canonical_dataset_name(dataset_name: str) -> str:
        normalized = re.sub(r"_pl\d+$", "", dataset_name)
        return normalized

    def _preferred_dataset_name(names: list[str]) -> str:
        pl_names = sorted(name for name in names if re.search(r"_pl\d+$", name))
        if pl_names:
            return pl_names[0]
        return sorted(names)[0]

    checkpoints = [str(run.checkpoint_path.relative_to(repo_root)) for run in group_runs]
    configs = [str(run.config_path.relative_to(repo_root)) for run in group_runs]
    labels = [run.label for run in group_runs]

    dataset_aliases: dict[str, list[RunSpec]] = defaultdict(list)
    dataset_sort_step: dict[str, float] = {}
    for run in group_runs:
        canonical = _canonical_dataset_name(run.dataset_name)
        dataset_aliases[canonical].append(run)
        dataset_sort_step[canonical] = min(
            dataset_sort_step.get(canonical, float("inf")),
            float(get_relative_time_step(run.freq)),
        )

    eval_dataset_configs = [
        str(_preferred_dataset_name([run.config_path.relative_to(repo_root).as_posix() for run in dataset_aliases[canonical]]))
        for canonical in sorted(dataset_aliases, key=lambda name: (dataset_sort_step[name], name))
    ]
    # slice and the tsdiff families share TSFlowCond's whole eval surface; the
    # checkpoint evaluator only understands {tsflow, lcde}, and the actual class
    # is re-selected from diffusion_params inside _create_tsflow_model.
    eval_model_type = (
        "tsflow" if model_type in {"slice", "tsdiff_gp", "tsdiff_ou", "tsdiff_gauss"} else model_type
    )
    model_output_dir = output_dir / model_type
    output_json = model_output_dir / f"{family_key}.json"
    output_csv = model_output_dir / f"{family_key}.csv"
    plot_dir = model_output_dir / f"{family_key}_plots"

    cmd = [
        sys.executable,
        "execute/evaluate_checkpoints_cross_frequency.py",
        "--checkpoint_paths",
        _py_list(checkpoints),
        "--config_paths",
        _py_list(configs),
        "--model_types",
        _py_list([eval_model_type] * len(group_runs)),
        "--labels",
        _py_list(labels),
        "--eval_dataset_configs",
        _py_list(eval_dataset_configs),
        "--device",
        device,
        "--num_samples",
        str(num_samples),
        "--plot_dir",
        str(plot_dir.relative_to(repo_root)),
        "--save_json",
        str(output_json.relative_to(repo_root)),
        "--save_csv",
        str(output_csv.relative_to(repo_root)),
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if fine_to_coarse_eval_adapter_override is not None:
        cmd.extend(["--fine_to_coarse_eval_adapter_override", str(fine_to_coarse_eval_adapter_override)])
    if only_coarser_eval_inputs:
        cmd.append("--only_coarser_eval_inputs")
    if only_not_finer_eval_inputs:
        cmd.append("--only_not_finer_eval_inputs")
    return cmd, output_json


def _group_runs_by_dataset_family(runs: list[RunSpec]) -> dict[str, list[RunSpec]]:
    grouped: dict[str, list[RunSpec]] = defaultdict(list)
    for run in runs:
        family_key = run.base_dataset or run.dataset_family
        grouped[family_key].append(run)
    return grouped


def _build_comparison_command(
    *,
    repo_root: Path,
    family_key: str,
    family_runs: list[RunSpec],
    device: str,
    num_samples: int,
    seed: int | None,
    output_dir: Path,
    fine_to_coarse_eval_adapter_override: Any,
    only_coarser_eval_inputs: bool,
    only_not_finer_eval_inputs: bool,
    overlay_comparison_models_symlog: bool,
) -> tuple[list[str], Path] | None:
    by_model_freq: dict[tuple[str, str], RunSpec] = {}
    for run in _sort_runs_for_eval(family_runs):
        if run.model_type not in {"slice", "tsflow"}:
            continue
        by_model_freq[(run.model_type, run.freq)] = run

    common_freqs = sorted(
        {freq for model_type, freq in by_model_freq if model_type == "slice"}
        & {freq for model_type, freq in by_model_freq if model_type == "tsflow"},
        key=lambda value: (float(get_relative_time_step(value)), value),
    )
    if not common_freqs:
        return None

    comparison_runs: list[RunSpec] = []
    for freq in common_freqs:
        comparison_runs.append(by_model_freq[("slice", freq)])
        comparison_runs.append(by_model_freq[("tsflow", freq)])

    def _canonical_dataset_name(dataset_name: str) -> str:
        normalized = re.sub(r"_pl\d+$", "", dataset_name)
        return normalized

    def _preferred_dataset_name(names: list[str]) -> str:
        pl_names = sorted(name for name in names if re.search(r"_pl\d+$", name))
        if pl_names:
            return pl_names[0]
        return sorted(names)[0]

    dataset_aliases: dict[str, list[RunSpec]] = defaultdict(list)
    dataset_sort_step: dict[str, float] = {}
    for run in family_runs:
        canonical = _canonical_dataset_name(run.dataset_name)
        dataset_aliases[canonical].append(run)
        dataset_sort_step[canonical] = min(
            dataset_sort_step.get(canonical, float("inf")),
            float(get_relative_time_step(run.freq)),
        )
    eval_dataset_configs = [
        str(_preferred_dataset_name([run.config_path.relative_to(repo_root).as_posix() for run in dataset_aliases[canonical]]))
        for canonical in sorted(dataset_aliases, key=lambda name: (dataset_sort_step[name], name))
    ]

    comparison_dir = output_dir / "comparison" / family_key
    cmd = [
        sys.executable,
        "execute/evaluate_checkpoints_cross_frequency.py",
        "--checkpoint_paths",
        _py_list([str(run.checkpoint_path.relative_to(repo_root)) for run in comparison_runs]),
        "--config_paths",
        _py_list([str(run.config_path.relative_to(repo_root)) for run in comparison_runs]),
        "--model_types",
        _py_list(["tsflow" if run.model_type == "slice" else run.model_type for run in comparison_runs]),
        "--labels",
        _py_list([run.label for run in comparison_runs]),
        "--eval_dataset_configs",
        _py_list(eval_dataset_configs),
        "--device",
        device,
        "--num_samples",
        str(num_samples),
        "--save_comparison_plots_dir",
        str(comparison_dir.relative_to(repo_root)),
        "--skip_individual_summary_plots",
    ]
    if seed is not None:
        cmd.extend(["--seed", str(seed)])
    if fine_to_coarse_eval_adapter_override is not None:
        cmd.extend(["--fine_to_coarse_eval_adapter_override", str(fine_to_coarse_eval_adapter_override)])
    if only_coarser_eval_inputs:
        cmd.append("--only_coarser_eval_inputs")
    if only_not_finer_eval_inputs:
        cmd.append("--only_not_finer_eval_inputs")
    if overlay_comparison_models_symlog:
        cmd.append("--overlay_comparison_models_symlog")
    return cmd, comparison_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Auto-detect TSFlow/LCDE checkpoints and run same-dataset cross-frequency evaluation."
    )
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--output_dir", type=str, default="results/cross_frequency_auto")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=6432)
    parser.add_argument(
        "--allowed_model_types",
        type=str,
        default="tsflow,slice,lcde,tsdiff_gp,tsdiff_ou,tsdiff_gauss",
        help="Comma-separated subset of model roots to include, e.g. tsflow,slice",
    )
    parser.add_argument(
        "--fine_to_coarse_eval_adapter_override",
        type=str,
        default=None,
        help="Optional override for dataset_params.fine_to_coarse_eval_adapter passed to checkpoint evaluation.",
    )
    parser.add_argument(
        "--only_coarser_eval_inputs",
        action="store_true",
        help="Only evaluate each checkpoint on datasets with strictly coarser input grids.",
    )
    parser.add_argument(
        "--only_not_finer_eval_inputs",
        action="store_true",
        help="Only evaluate each checkpoint on datasets with equal or coarser input grids.",
    )
    parser.add_argument(
        "--overlay_comparison_models_symlog",
        action="store_true",
        help="Use aligned symmetric symlog y-axes for the TSFlow/G-SLiCE comparison plots.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    results_root = (repo_root / args.results_root).resolve()
    output_dir = (repo_root / args.output_dir).resolve()

    allowed_model_types = {part.strip() for part in str(args.allowed_model_types).split(",") if part.strip()}
    runs = [run for run in _discover_runs(results_root) if run.model_type in allowed_model_types]
    if not runs:
        raise ValueError(f"No checkpoints with configs found under {results_root}.")

    grouped = _group_runs_by_family(runs)
    family_grouped = _group_runs_by_dataset_family(runs)
    commands: list[tuple[list[str], Path, str]] = []
    for (_, family_key), group_runs in sorted(grouped.items()):
        group_runs = _sort_runs_for_eval(group_runs)
        unique_freqs = {run.freq for run in group_runs}
        if len(unique_freqs) < 2:
            continue
        cmd, output_json = _build_eval_command(
            repo_root=repo_root,
            family_key=family_key,
            group_runs=group_runs,
            device=args.device,
            num_samples=args.num_samples,
            seed=args.seed,
            output_dir=output_dir,
            fine_to_coarse_eval_adapter_override=args.fine_to_coarse_eval_adapter_override,
            only_coarser_eval_inputs=bool(args.only_coarser_eval_inputs),
            only_not_finer_eval_inputs=bool(args.only_not_finer_eval_inputs),
        )
        commands.append((cmd, output_json, "results"))

    for family_key, family_runs in sorted(family_grouped.items()):
        comparison_command = _build_comparison_command(
            repo_root=repo_root,
            family_key=family_key,
            family_runs=family_runs,
            device=args.device,
            num_samples=args.num_samples,
            seed=args.seed,
            output_dir=output_dir,
            fine_to_coarse_eval_adapter_override=args.fine_to_coarse_eval_adapter_override,
            only_coarser_eval_inputs=bool(args.only_coarser_eval_inputs),
            only_not_finer_eval_inputs=bool(args.only_not_finer_eval_inputs),
            overlay_comparison_models_symlog=bool(args.overlay_comparison_models_symlog),
        )
        if comparison_command is None:
            continue
        cmd, comparison_dir = comparison_command
        commands.append((cmd, comparison_dir, "comparison"))

    if not commands:
        raise ValueError(f"No same-dataset multi-frequency groups found under {results_root}.")

    output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[tuple[Path, int]] = []
    for cmd, output_path, kind in commands:
        print(" ".join(cmd), flush=True)
        if not args.dry_run:
            try:
                subprocess.run(cmd, check=True, cwd=repo_root)
                if kind == "comparison":
                    print(f"Saved comparison plots to {output_path}", flush=True)
                else:
                    print(f"Saved results to {output_path}", flush=True)
            except subprocess.CalledProcessError as exc:
                failures.append((output_path, int(exc.returncode)))
                print(
                    f"Skipping failed group {output_path.stem} "
                    f"(exit code {exc.returncode}).",
                    flush=True,
                )
                if args.fail_fast:
                    raise

    if failures:
        print("Failed groups:", flush=True)
        for output_json, returncode in failures:
            print(f"  {output_json} (exit code {returncode})", flush=True)

    if not args.dry_run:
        values_path = _write_values_markdown(output_dir)
        if values_path is not None:
            print(f"Saved Markdown summary to {values_path}", flush=True)


if __name__ == "__main__":
    main()
