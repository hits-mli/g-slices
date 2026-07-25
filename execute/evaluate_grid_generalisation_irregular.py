import argparse
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

import yaml

from gslice.dataset import get_dataset_family_from_params, get_dataset_name_from_params
from gslice.irregular import get_irregular_grid_spec, has_irregular_grid


@dataclass(frozen=True)
class RunSpec:
    model_type: str
    run_dir: Path
    checkpoint_path: Path
    config_path: Path
    dataset_name: str
    dataset_family: str
    gamma_k: float
    train_seed: int | None
    label: str


RAW_RESULTS_SUBDIR = "seed_runs"


def _resolve_checkpoint_path(run_dir: Path, variant: str) -> Path:
    normalized = str(variant or "best").strip().lower()
    if normalized == "best":
        checkpoint_path = run_dir / "best_checkpoint.ckpt"
    elif normalized == "last":
        checkpoint_path = run_dir / "csv_logs" / "version_0" / "checkpoints" / "last.ckpt"
    else:
        raise ValueError("checkpoint_variant must be one of {'best', 'last'}.")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


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


_KNOWN_MODEL_DIRS = {
    "tsflow": "tsflow",
    "slice": "slice",
    "tsflow_run": "tsflow",
    # DSPD (tsdiff) baselines: one family dir per noise kernel — discovery
    # dedupes on (model_type, dataset, seed), so merging them under a single
    # dir would silently collapse the kernels.
    "tsdiff_gp": "tsdiff_gp",
    "tsdiff_ou": "tsdiff_ou",
    "tsdiff_gauss": "tsdiff_gauss",
}


def _candidate_model_roots(results_root: Path) -> list[tuple[Path, str]]:
    candidates: list[tuple[Path, str]] = []
    for dirname, model_type in _KNOWN_MODEL_DIRS.items():
        candidate = results_root / dirname
        if candidate.is_dir():
            candidates.append((candidate, model_type))
    if results_root.name in _KNOWN_MODEL_DIRS and results_root.is_dir():
        candidates.append((results_root, _KNOWN_MODEL_DIRS[results_root.name]))
    return candidates


def _run_sort_key(run_dir: Path) -> tuple[int, str]:
    try:
        return int(run_dir.name), run_dir.name
    except ValueError:
        return -1, run_dir.name


def _discover_runs(results_root: Path, *, checkpoint_variant: str) -> list[RunSpec]:
    latest_by_dataset: dict[tuple[str, str, int | None], RunSpec] = {}
    for model_root, model_type in _candidate_model_roots(results_root):
        run_dirs = sorted((path for path in model_root.iterdir() if path.is_dir()), key=_run_sort_key)
        for run_dir in run_dirs:
            config_path = _resolve_run_config_path(run_dir)
            if config_path is None:
                continue
            try:
                resolved_checkpoint_path = _resolve_checkpoint_path(run_dir, checkpoint_variant)
            except FileNotFoundError:
                continue
            config = _read_yaml(config_path)
            dataset_params = dict(config.get("dataset_params", {}) or {})
            if not has_irregular_grid(dataset_params):
                continue
            spec = get_irregular_grid_spec(dataset_params, fallback_freq=str(config.get("model_params", {}).get("freq", "")))
            dataset_name = get_dataset_name_from_params(dataset_params)
            dataset_family = get_dataset_family_from_params(dataset_params)
            train_seed = config.get("seed")
            if train_seed is not None:
                train_seed = int(train_seed)
            run_spec = RunSpec(
                model_type=model_type,
                run_dir=run_dir,
                checkpoint_path=resolved_checkpoint_path,
                config_path=config_path,
                dataset_name=dataset_name,
                dataset_family=dataset_family,
                gamma_k=float(spec.gamma_k),
                train_seed=train_seed,
                label=(
                    f"{model_type}_{run_dir.name}_{checkpoint_variant}"
                    f"{'' if train_seed is None else f'_s{int(train_seed)}'}"
                ),
            )
            dataset_key = (model_type, dataset_name, train_seed)
            current = latest_by_dataset.get(dataset_key)
            if current is None or _run_sort_key(run_dir) > _run_sort_key(current.run_dir):
                latest_by_dataset[dataset_key] = run_spec
    return sorted(
        latest_by_dataset.values(),
        key=lambda run: (
            run.model_type,
            run.dataset_family,
            -1 if run.train_seed is None else int(run.train_seed),
            run.gamma_k,
            _run_sort_key(run.run_dir),
        ),
    )


def _group_runs(runs: list[RunSpec]) -> dict[tuple[str, str, int | None], list[RunSpec]]:
    grouped: dict[tuple[str, str, int | None], list[RunSpec]] = defaultdict(list)
    for run in runs:
        grouped[(run.model_type, run.dataset_family, run.train_seed)].append(run)
    return grouped


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _extract_gamma_k(dataset_name: str) -> float:
    match = re.search(r"_k([0-9mp]+)$", str(dataset_name))
    if match is None:
        raise ValueError(f"Could not extract gamma-k from dataset name {dataset_name!r}.")
    return float(match.group(1).replace("p", ".").replace("m", "-"))


def _make_regular_eval_dataset_params(dataset_params: dict[str, Any]) -> dict[str, Any]:
    regular_dataset_params = dict(dataset_params)
    irregular_grid_params = dict(regular_dataset_params.get("irregular_grid_params", {}) or {})
    if not irregular_grid_params:
        raise ValueError("Regular-grid generalisation requires dataset_params.irregular_grid_params.")
    irregular_grid_params["eval_context_sampling"] = "regular"
    irregular_grid_params["eval_future_sampling"] = "regular"
    regular_dataset_params["irregular_grid_params"] = irregular_grid_params
    return regular_dataset_params


def _materialize_regular_eval_config(*, output_dir: Path, run: RunSpec) -> Path:
    config = _read_yaml(run.config_path)
    dataset_params = dict(config.get("dataset_params", {}) or {})
    regular_config = {
        "dataset_params": _make_regular_eval_dataset_params(dataset_params),
    }
    target_path = output_dir / "_eval_dataset_configs" / f"{_slugify(run.dataset_family)}__regular.yaml"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    # The filename depends only on the dataset family, so every seed-group (and,
    # under --num_shards, every concurrent shard) writes the same path. Write via
    # a unique temp file + atomic rename so a reader never sees a partial file.
    tmp_path = target_path.with_name(f"{target_path.name}.tmp{os.getpid()}")
    with tmp_path.open("w") as fp:
        yaml.safe_dump(regular_config, fp, sort_keys=False)
    tmp_path.replace(target_path)
    return target_path


def _seed_slug(seed: int | None) -> str:
    return "noseed" if seed is None else f"s{int(seed)}"


def _raw_eval_json_path(*, output_dir: Path, eval_mode: str, run: RunSpec) -> Path:
    return (
        output_dir
        / eval_mode
        / run.model_type
        / RAW_RESULTS_SUBDIR
        / f"{_slugify(run.dataset_family)}__train_k_{run.gamma_k:g}__{_seed_slug(run.train_seed)}.json"
    )


def _run_eval(
    *,
    repo_root: Path,
    run: RunSpec,
    eval_mode: str,
    eval_dataset_configs: list[Path],
    output_dir: Path,
    save_json: Path,
    device: str,
    num_samples: int,
    seed: int | None,
    adapter_mode: str | None,
    dry_run: bool,
) -> int:
    save_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(repo_root / "execute" / "evaluate_checkpoints_cross_frequency.py"),
        "--checkpoint_paths",
        repr([str(run.checkpoint_path)]),
        "--config_paths",
        repr([str(run.config_path)]),
        "--model_types",
        # slice and the tsdiff families ride TSFlowCond's eval surface; the class
        # is re-selected from diffusion_params inside _create_tsflow_model.
        repr(["tsflow" if run.model_type in {"slice", "tsdiff_gp", "tsdiff_ou", "tsdiff_gauss"} else run.model_type]),
        "--labels",
        repr([run.label]),
        "--eval_dataset_configs",
        repr([str(path) for path in eval_dataset_configs]),
        "--device",
        str(device),
        "--num_samples",
        str(int(num_samples)),
        "--save_json",
        str(save_json),
    ]
    if seed is not None:
        cmd.extend(["--seed", str(int(seed))])
    if adapter_mode is not None:
        cmd.extend(["--fine_to_coarse_eval_adapter_override", str(adapter_mode)])
    if dry_run:
        print(" ".join(cmd))
        return 0
    completed = subprocess.run(cmd, cwd=repo_root, check=False)
    return int(completed.returncode)


def _read_json_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r") as fp:
        data = json.load(fp)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of evaluation rows in {path}.")
    return [dict(row) for row in data]


def _aggregate_metric(values: list[float]) -> tuple[float, float]:
    if not values:
        raise ValueError("Cannot aggregate an empty metric list.")
    if len(values) == 1:
        return float(values[0]), 0.0
    return float(fmean(values)), float(stdev(values))


def _aggregate_seed_results(output_dir: Path) -> None:
    for eval_mode_dir in sorted(path for path in output_dir.iterdir() if path.is_dir()):
        eval_mode = eval_mode_dir.name
        for model_dir in sorted(path for path in eval_mode_dir.iterdir() if path.is_dir()):
            if model_dir.name == RAW_RESULTS_SUBDIR:
                continue
            raw_dir = model_dir / RAW_RESULTS_SUBDIR
            if not raw_dir.is_dir():
                continue

            rows_by_train_dataset: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
            for json_path in sorted(raw_dir.glob("*.json")):
                seed_slug = json_path.stem.rsplit("__", 1)[-1]
                for row in _read_json_rows(json_path):
                    rows_by_train_dataset[str(row["train_dataset"])].append((seed_slug, row))

            for train_dataset, seed_rows in rows_by_train_dataset.items():
                grouped_by_eval_dataset: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
                for seed_slug, row in seed_rows:
                    grouped_by_eval_dataset[str(row["eval_dataset"])].append((seed_slug, row))

                aggregated_rows: list[dict[str, Any]] = []
                for eval_dataset, grouped_rows in sorted(grouped_by_eval_dataset.items()):
                    representative = dict(grouped_rows[0][1])
                    seed_slugs = sorted(seed_slug for seed_slug, _ in grouped_rows)
                    aggregated_row = dict(representative)
                    aggregated_row["label"] = f"{representative['model_type']}_seed_average"
                    aggregated_row["checkpoint_path"] = None
                    aggregated_row["config_path"] = None
                    aggregated_row["train_seed"] = None
                    aggregated_row["seed_values"] = seed_slugs
                    aggregated_row["num_seeds"] = len(seed_slugs)
                    for metric_name in ("CRPS", "ND", "NRMSE"):
                        metric_values = [float(row[metric_name]) for _, row in grouped_rows]
                        metric_mean, metric_std = _aggregate_metric(metric_values)
                        aggregated_row[metric_name] = metric_mean
                        aggregated_row[f"{metric_name}_std"] = metric_std
                    aggregated_rows.append(aggregated_row)

                if not aggregated_rows:
                    continue

                train_k = _extract_gamma_k(train_dataset)
                dataset_family_slug = _slugify(str(train_dataset).split("__")[0])
                target_path = model_dir / f"{dataset_family_slug}__train_k_{train_k:g}.json"
                with target_path.open("w") as fp:
                    json.dump(aggregated_rows, fp, indent=2)
                    fp.write("\n")


def _write_values_markdown(output_dir: Path) -> None:
    rows_by_eval_mode: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for eval_mode_dir in sorted(path for path in output_dir.iterdir() if path.is_dir()):
        for model_dir in sorted(path for path in eval_mode_dir.iterdir() if path.is_dir()):
            if model_dir.name == RAW_RESULTS_SUBDIR:
                continue
            for json_path in sorted(model_dir.glob("*.json")):
                rows_by_eval_mode[(eval_mode_dir.name, model_dir.name)].extend(_read_json_rows(json_path))

    for (eval_mode, model_type), rows in rows_by_eval_mode.items():
        if not rows:
            continue
        target_dir = output_dir / eval_mode / model_type
        if eval_mode == "cross_irregular":
            train_values = sorted({_extract_gamma_k(str(row["train_dataset"])) for row in rows})
            eval_values = sorted({_extract_gamma_k(str(row["eval_dataset"])) for row in rows})
            row_map = {
                (
                    _extract_gamma_k(str(row["eval_dataset"])),
                    _extract_gamma_k(str(row["train_dataset"])),
                ): row
                for row in rows
            }
            lines = [
                f"# Irregular Generalisation - {model_type} ({eval_mode})",
                "",
                "Values are mean ± std over seeds.",
                "",
                "| test k \\\\ train k | " + " | ".join(f"{value:g}" for value in train_values) + " |",
                "|" + "---|" * (len(train_values) + 1),
            ]
            for eval_value in eval_values:
                lines.append(
                    f"| {eval_value:g} | "
                    + " | ".join(
                        f"{float(row_map[(eval_value, train_value)]['CRPS']):.6f} ± {float(row_map[(eval_value, train_value)].get('CRPS_std', 0.0)):.6f}"
                        for train_value in train_values
                    )
                    + " |"
                )
        else:
            ordered_rows = sorted(rows, key=lambda row: _extract_gamma_k(str(row["train_dataset"])))
            lines = [
                f"# Irregular Generalisation - {model_type} ({eval_mode})",
                "",
                "Values are mean ± std over seeds.",
                "",
                "| train k | eval dataset | CRPS | ND | NRMSE |",
                "|---|---|---|---|---|",
            ]
            for row in ordered_rows:
                lines.append(
                    f"| {_extract_gamma_k(str(row['train_dataset'])):g} | {row['eval_dataset']} | "
                    f"{float(row['CRPS']):.6f} ± {float(row.get('CRPS_std', 0.0)):.6f} | "
                    f"{float(row['ND']):.6f} ± {float(row.get('ND_std', 0.0)):.6f} | "
                    f"{float(row['NRMSE']):.6f} ± {float(row.get('NRMSE_std', 0.0)):.6f} |"
                )
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "values.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate irregular-grid generalisation across training/eval gamma-k levels.")
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--adapter_mode", default="none")
    parser.add_argument("--checkpoint_variant", choices=("best", "last"), default="best")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Split the per-checkpoint evaluations across this many independent processes/GPUs.",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Which shard this process handles (0 <= shard_index < num_shards).",
    )
    parser.add_argument(
        "--aggregate_only",
        action="store_true",
        help="Skip evaluation and only aggregate/emit tables from raw JSONs already on disk.",
    )
    args = parser.parse_args()

    num_shards = max(1, int(args.num_shards))
    shard_index = int(args.shard_index)
    if not 0 <= shard_index < num_shards:
        raise SystemExit(f"--shard_index must satisfy 0 <= index < {num_shards}, got {shard_index}.")

    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        # Sharded mode: every shard writes only its own raw JSONs, so aggregation
        # runs once afterwards over the complete set.
        _aggregate_seed_results(output_dir)
        _write_values_markdown(output_dir)
        return

    runs = _discover_runs(results_root, checkpoint_variant=str(args.checkpoint_variant))
    grouped = _group_runs(runs)
    repo_root = Path(__file__).resolve().parents[1]

    # Flatten to one work item per checkpoint. The eval-dataset list stays derived
    # from the *full* group, so every checkpoint is still evaluated against every
    # grid regardless of how the work is sharded.
    work: list[tuple[Any, list[Path], Path]] = []
    for (_, _, _), group_runs in sorted(grouped.items()):
        ordered_group_runs = sorted(group_runs, key=lambda item: item.gamma_k)
        cross_irregular_configs = [run.config_path for run in ordered_group_runs]
        regular_eval_config = _materialize_regular_eval_config(output_dir=output_dir, run=ordered_group_runs[0])
        for run in ordered_group_runs:
            work.append((run, cross_irregular_configs, regular_eval_config))

    shard_work = work[shard_index::num_shards]
    if num_shards > 1:
        print(
            f"[shard {shard_index}/{num_shards}] handling {len(shard_work)} of {len(work)} checkpoints",
            flush=True,
        )

    failures = 0
    for run, cross_irregular_configs, regular_eval_config in shard_work:
        failures += _run_eval(
            repo_root=repo_root,
            run=run,
            eval_mode="cross_irregular",
            eval_dataset_configs=cross_irregular_configs,
            output_dir=output_dir,
            save_json=_raw_eval_json_path(output_dir=output_dir, eval_mode="cross_irregular", run=run),
            device=args.device,
            num_samples=int(args.num_samples),
            seed=args.seed,
            adapter_mode=None if args.adapter_mode == "none" else args.adapter_mode,
            dry_run=bool(args.dry_run),
        )
        failures += _run_eval(
            repo_root=repo_root,
            run=run,
            eval_mode="regular_grid",
            eval_dataset_configs=[regular_eval_config],
            output_dir=output_dir,
            save_json=_raw_eval_json_path(output_dir=output_dir, eval_mode="regular_grid", run=run),
            device=args.device,
            num_samples=int(args.num_samples),
            seed=args.seed,
            adapter_mode=None if args.adapter_mode == "none" else args.adapter_mode,
            dry_run=bool(args.dry_run),
        )

    # With --num_shards > 1 each shard only owns part of the raw JSONs, so the
    # caller aggregates once after all shards finish (--aggregate_only).
    if not args.dry_run and num_shards == 1:
        _aggregate_seed_results(output_dir)
        _write_values_markdown(output_dir)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
