import argparse
import ast
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from gluonts.dataset.loader import InferenceDataLoader
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.batchify import batchify

from gslice.dataset import get_dataset_name_from_params, get_gts_dataset_from_params, infer_target_dim
from gslice.model.tsflow_cond import TSFlowCond
from gslice.utils import create_transforms
from gslice.utils.util import create_splitter
from gslice.utils.variables import frequencies_match, get_lags_for_freq, get_season_length


SUPPORTED_MODEL_TYPES = {"tsflow", "lcde", "auto"}
TOLERATED_MISSING_STATE_KEYS = {
    "_continuous_gp_regressor.gamma",
    "_continuous_gp_regressor.noise",
    "_continuous_gp_regressor.jitter",
}


def create_multivariate_transforms(*args, **kwargs):
    raise NotImplementedError(
        "Multivariate GluonTS evaluation is not available in the current gslice.utils API."
    )


@dataclass
class BenchmarkSpec:
    checkpoint_path: str
    config_path: str
    model_type: str
    label: str


def _parse_list_arg(value: str, name: str) -> List[str]:
    parsed = ast.literal_eval(value)
    if isinstance(parsed, str):
        return [parsed]
    if not isinstance(parsed, list):
        raise ValueError(f"{name} must be a Python list literal, got {type(parsed).__name__}.")
    values = [str(x) for x in parsed]
    if len(values) == 1 and "," in values[0]:
        split_values = [item.strip() for item in values[0].split(",") if item.strip()]
        if len(split_values) > 1:
            return split_values
    return values


def _parse_int_list_arg(value: str, name: str) -> List[int]:
    parsed = ast.literal_eval(value)
    if isinstance(parsed, int):
        return [int(parsed)]
    if not isinstance(parsed, list):
        raise ValueError(f"{name} must be a Python list literal, got {type(parsed).__name__}.")
    return [int(x) for x in parsed]


def _resolve_device(device: str) -> torch.device:
    requested = device.lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device)


def _read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as fp:
        config = yaml.safe_load(fp)

    cfg_path = Path(path)
    hparams_candidates = sorted(cfg_path.parent.glob("csv_logs/version_*/hparams.yaml"))
    if not hparams_candidates:
        return config

    hparams_path = hparams_candidates[-1]
    with open(hparams_path, "r") as fp:
        hparams = yaml.safe_load(fp)
    if not isinstance(hparams, dict):
        return config

    model_params = dict(config.get("model_params", {}))
    if "backbone_params" in hparams:
        # TSFlow runs save the runtime-resolved architecture in hparams.yaml.
        for key in [
            "context_length",
            "prediction_length",
            "optimizer_params",
            "prior_params",
            "normalization",
            "use_lags",
            "use_ema",
            "num_steps",
            "solver",
            "matching",
            "ema_params",
            "gp_time_mode",
            "gp_fit_context_only",
            "lags_seq",
            "prior_context_length_override",
        ]:
            if key in hparams:
                model_params[key] = hparams[key]
        if "frequency" in hparams:
            model_params["freq"] = hparams["frequency"]
        model_params["backbone_params"] = hparams["backbone_params"]
        if "setting" in hparams:
            config["setting"] = hparams["setting"]
        if "target_dim" in hparams:
            config["target_dim"] = hparams["target_dim"]
    else:
        # LCDE-style runs log their model kwargs flat in hparams.yaml.
        for key, value in hparams.items():
            if key == "setting":
                config["setting"] = value
            else:
                model_params[key] = value

    config["model_params"] = model_params
    return config


def _infer_config_path(checkpoint_path: str) -> str:
    ckpt = Path(checkpoint_path)
    for parent in ckpt.parents:
        candidate = parent / "config.yaml"
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"Could not infer config path for checkpoint {checkpoint_path}. "
        "Pass --config_paths explicitly."
    )


def _infer_model_type(config: Dict[str, Any]) -> str:
    model_name = str(config.get("model", "")).lower()
    model_params = config.get("model_params", {})
    if model_name in {"conditional", "tsflow", "tsflow_cond"}:
        return "tsflow"
    if model_name in {"lcde_gluonts", "lcde"}:
        return "lcde"
    if "backbone_params" in model_params:
        return "tsflow"
    if "hidden_dim" in model_params and "n_blocks" in model_params:
        return "lcde"
    raise ValueError(f"Could not infer model type from config model={config.get('model')!r}.")


def _extract_state_dict(raw_ckpt: Any) -> Dict[str, torch.Tensor]:
    if isinstance(raw_ckpt, dict) and "state_dict" in raw_ckpt and isinstance(raw_ckpt["state_dict"], dict):
        return raw_ckpt["state_dict"]
    if isinstance(raw_ckpt, dict) and raw_ckpt and all(torch.is_tensor(v) for v in raw_ckpt.values()):
        return raw_ckpt
    raise ValueError("Checkpoint format not recognized. Expected state_dict or Lightning checkpoint dict.")


def _maybe_strip_prefix(state_dict: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(k.startswith(prefix) for k in state_dict):
        return {k[len(prefix) :]: v for k, v in state_dict.items()}
    return state_dict


def _extract_incompatible_keys(result: Any) -> tuple[List[str], List[str]]:
    if hasattr(result, "missing_keys") and hasattr(result, "unexpected_keys"):
        return list(result.missing_keys), list(result.unexpected_keys)
    if isinstance(result, tuple) and len(result) == 2:
        missing, unexpected = result
        return list(missing), list(unexpected)
    return [], []


def _can_tolerate_incompatible_keys(missing: List[str], unexpected: List[str]) -> bool:
    return len(unexpected) == 0 and set(missing).issubset(TOLERATED_MISSING_STATE_KEYS)


def _maybe_pad_legacy_slice_time_bias_weights(
    state_dict: Dict[str, torch.Tensor],
    model: torch.nn.Module,
) -> Dict[str, torch.Tensor]:
    if not hasattr(model, "state_dict"):
        return state_dict
    model_state = model.state_dict()
    padded: Dict[str, torch.Tensor] = dict(state_dict)
    changed = False
    for key, value in state_dict.items():
        target = model_state.get(key)
        if target is None:
            continue
        if not isinstance(value, torch.Tensor) or not isinstance(target, torch.Tensor):
            continue
        if not key.endswith(("vf_A.weight", "vf_B.weight")):
            continue
        if value.ndim != 2 or target.ndim != 2:
            continue
        if value.shape[0] != target.shape[0]:
            continue
        if value.shape[1] + 1 != target.shape[1]:
            continue
        upgraded = value.new_zeros(target.shape)
        upgraded[:, 1:] = value
        padded[key] = upgraded
        changed = True
    return padded if changed else state_dict


def _load_weights(model: torch.nn.Module, checkpoint_path: str) -> None:
    raw_ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = _extract_state_dict(raw_ckpt)

    candidates = [
        state_dict,
        _maybe_strip_prefix(state_dict, "model."),
        _maybe_strip_prefix(state_dict, "module."),
    ]

    last_error: Optional[RuntimeError] = None
    for candidate in candidates:
        adapted_candidate = _maybe_pad_legacy_slice_time_bias_weights(candidate, model)
        try:
            model.load_state_dict(adapted_candidate, strict=True)
            return
        except RuntimeError as err:
            last_error = err
        try:
            missing, unexpected = _extract_incompatible_keys(
                model.load_state_dict(adapted_candidate, strict=False)
            )
        except RuntimeError:
            continue
        if _can_tolerate_incompatible_keys(missing, unexpected):
            if missing:
                print(
                    "WARNING: Loaded checkpoint with tolerated missing keys: "
                    + ", ".join(sorted(missing))
                )
            return

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to load checkpoint: {checkpoint_path}")


def _clean_prior_params(prior_params: Optional[dict]) -> Optional[dict]:
    if prior_params is None:
        return None
    allowed = {
        "kernel",
        "gamma",
        "noise",
        "jitter",
        "use_data_mean",
        "season_length",
        "context_freqs",
        "iso",
    }
    return {k: v for k, v in prior_params.items() if k in allowed}


def _create_tsflow_model(config: Dict[str, Any], target_dim: int) -> TSFlowCond:
    setting = config["setting"]
    model_params = config["model_params"]
    return TSFlowCond(
        setting=setting,
        target_dim=target_dim,
        context_length=model_params["context_length"],
        prediction_length=model_params["prediction_length"],
        backbone_params=model_params["backbone_params"],
        prior_params=model_params["prior_params"],
        optimizer_params=model_params.get("optimizer_params", {"lr": 1e-3}),
        ema_params=model_params.get("ema_params", {"beta": 0.9999, "update_after_step": 128, "update_every": 1}),
        frequency=model_params["freq"],
        normalization=model_params.get("normalization", None),
        use_lags=model_params.get("use_lags", True),
        use_ema=model_params.get("use_ema", False),
        num_steps=model_params.get("num_steps", 16),
        solver=model_params.get("solver", "euler"),
        matching=model_params.get("matching", "random"),
        gp_time_mode=model_params.get("gp_time_mode", "discrete"),
        num_samples=model_params.get("num_samples", 1),
        lags_seq=model_params.get("lags_seq"),
        prior_context_length_override=model_params.get("prior_context_length_override"),
        gp_fit_context_only=model_params.get("gp_fit_context_only", False),
    )


def _create_lcde_model(config: Dict[str, Any], target_dim: int):
    model_name = str(config.get("model", "")).lower()
    setting = config["setting"]
    model_params = config["model_params"]

    try:
        from gslice.arch.res_lcde_gluonts import LCDEGluonTSModule
        from gslice.arch.res_lcde_gluonts_latent_fm import LCDELatentFMGluonTSModule
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "LCDE/SLiCE benchmarking requires the optional 'slices' dependency. "
            "TSFlow checkpoints can still be benchmarked in this environment, but SLiCE/LCDE cannot be loaded."
        ) from exc

    input_dim = model_params.get("input_dim", 1 if setting == "univariate" else target_dim)
    output_dim = model_params.get("output_dim", input_dim)
    prior_params = _clean_prior_params(model_params.get("prior_params", None))
    use_lags = bool(model_params.get("use_lags", False))
    lags_seq = model_params.get("lags_seq", None)
    if use_lags and lags_seq is None:
        long_context_length = int(model_params.get("long_context_length", 0))
        lags_seq = get_lags_for_freq(
            model_params["freq"],
            context_length=int(model_params["context_length"]),
            max_lag=long_context_length,
        )

    common_kwargs = dict(
        context_length=model_params["context_length"],
        prediction_length=model_params["prediction_length"],
        input_dim=input_dim,
        hidden_dim=model_params["hidden_dim"],
        output_dim=output_dim,
        n_blocks=model_params["n_blocks"],
        num_features=model_params.get("num_features", input_dim),
        long_context_length=model_params.get("long_context_length", 0),
        gp_fit_source=model_params.get("gp_fit_source", "context"),
        learning_rate=model_params.get("learning_rate", 1e-3),
        frequency=model_params["freq"],
        prior_params=prior_params,
        num_samples=model_params.get("num_samples", 1),
        inpaint_factor=model_params.get("inpaint_factor", 1),
        use_ema=model_params.get("use_ema", False),
        ema_params=model_params.get("ema_params", None),
        use_lags=use_lags,
        lags_seq=lags_seq,
        concat_tsflow_features=model_params.get("concat_tsflow_features", False),
        long_context_mode=model_params.get("long_context_mode", "embed"),
        normalization=model_params.get("normalization", "longmean"),
    )
    if "latent_fm" in model_name:
        return LCDELatentFMGluonTSModule(
            **common_kwargs,
            latent_fm_weight=model_params.get("latent_fm_weight", 1.0),
            reconstruction_weight=model_params.get("reconstruction_weight", 1.0),
            latent_fm_sigmin=model_params.get("latent_fm_sigmin", 1e-3),
            latent_fm_sigmax=model_params.get("latent_fm_sigmax", None),
            fm_stack_depth=model_params.get("fm_stack_depth", 3),
            fm_step_emb=model_params.get("fm_step_emb", 64),
        )
    return LCDEGluonTSModule(**common_kwargs)


def _build_model(spec: BenchmarkSpec, config: Dict[str, Any], device: torch.device):
    dataset_params = config["dataset_params"]
    dataset = get_gts_dataset_from_params(
        dataset_params,
        regenerate=bool(dataset_params.get("regenerate", False)),
    )
    target_dim = infer_target_dim(dataset)

    if spec.model_type == "tsflow":
        model = _create_tsflow_model(config, target_dim)
    elif spec.model_type == "lcde":
        model = _create_lcde_model(config, target_dim)
    else:
        raise ValueError(f"Unknown model type: {spec.model_type}")

    _load_weights(model, spec.checkpoint_path)
    model.to(device)
    model.eval()
    return model


def _resolve_num_samples(config: Dict[str, Any], override_num_samples: Optional[int]) -> int:
    if override_num_samples is not None:
        return int(override_num_samples)
    model_params = config.get("model_params", {})
    eval_params = config.get("evaluation_params", {})
    if "num_samples" in model_params:
        return int(model_params["num_samples"])
    if "num_samples" in eval_params:
        return int(eval_params["num_samples"])
    return 1


def _prepare_gluonts_batch(
    config: Dict[str, Any],
    model,
    model_type: str,
    device: torch.device,
    batch_size_override: Optional[int],
) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    setting = config["setting"]
    model_params = config["model_params"]
    dataset_params = config["dataset_params"]

    dataset = get_gts_dataset_from_params(
        dataset_params,
        regenerate=bool(dataset_params.get("regenerate", False)),
    )
    target_dim = infer_target_dim(dataset)
    freq = str(dataset.metadata.freq)
    prediction_length = int(dataset.metadata.prediction_length)
    context_length = int(prediction_length)

    num_rolling_evals = int(len(dataset.test) / len(dataset.train))
    time_features = time_features_from_frequency_str(freq)

    if setting == "univariate":
        transformation = create_transforms(
            time_features=time_features,
            prediction_length=prediction_length,
            freq=get_season_length(freq),
            train_length=len(dataset.train),
        )
        training_data = dataset.train
        test_data = dataset.test
    elif setting == "multivariate":
        train_grouper = MultivariateGrouper(max_target_dim=target_dim)
        test_grouper = MultivariateGrouper(
            num_test_dates=num_rolling_evals,
            max_target_dim=target_dim,
        )
        transformation = create_multivariate_transforms(
            time_features=time_features,
            prediction_length=prediction_length,
            target_dim=target_dim,
            freq=get_season_length(freq),
            train_length=len(dataset.train),
        )
        training_data = train_grouper(dataset.train)
        test_data = test_grouper(dataset.test)
    else:
        raise ValueError(f"Unknown setting: {setting}")

    # Prime stateful transforms (e.g. AddMeanFeature.train_means) exactly as in training.
    # Without this, evaluating on test-only data can raise KeyError for unseen ids.
    for _ in transformation.apply(training_data, is_train=True):
        pass

    transformed_testdata = transformation.apply(test_data, is_train=False)

    if model_type == "tsflow":
        max_lag = max(getattr(model, "lags_seq", [0]) or [0])
        past_length = max(
            int(context_length) + max_lag,
            int(model.prior_context_length),
        )
    else:
        past_length = int(context_length) + int(model_params.get("long_context_length", 0))

    test_splitter = create_splitter(
        past_length=past_length,
        future_length=prediction_length,
        mode="test",
    )

    loader_batch_size = int(batch_size_override or dataset_params.get("test_batch_size", dataset_params["batch_size"]))
    inference_data_loader = InferenceDataLoader(
        transformed_testdata,
        transform=test_splitter,
        batch_size=loader_batch_size,
        stack_fn=lambda data: batchify(data, device),
    )
    batch = next(iter(inference_data_loader))

    metadata = {
        "dataset": get_dataset_name_from_params(dataset_params),
        "setting": setting,
        "freq": freq,
        "context_length": context_length,
        "prediction_length": prediction_length,
        "past_length": past_length,
        "batch_size_actual": int(batch["past_target"].shape[0]),
    }
    return batch, metadata


@torch.no_grad()
def _run_single_inference(model_type: str, model, batch: Dict[str, torch.Tensor], num_samples: int) -> torch.Tensor:
    past_target = batch["past_target"]
    past_observed_values = batch.get("past_observed_values", None)
    mean = batch.get("mean", None)
    scale = batch.get("scale", None)
    past_time_grid = batch.get("past_time_grid", None)
    future_time_grid = batch.get("future_time_grid", None)
    lag_features = batch.get("lag_features", None)
    dense_past_target = batch.get("dense_past_target", None)
    dense_past_observed_values = batch.get("dense_past_observed_values", None)
    dense_past_time_grid = batch.get("dense_past_time_grid", None)

    if mean is None:
        if past_target.dim() == 2:
            mean = past_target.mean(dim=1, keepdim=True)
        else:
            mean = past_target.mean(dim=1)

    if past_observed_values is None:
        if past_target.dim() == 2:
            past_observed_values = torch.ones_like(past_target, dtype=torch.bool)
        else:
            past_observed_values = torch.ones_like(past_target[..., 0], dtype=torch.bool)

    if model_type == "tsflow":
        model.num_samples = int(num_samples)
        return model(
            past_target=past_target,
            past_observed_values=past_observed_values,
            mean=mean,
            past_time_grid=past_time_grid,
            future_time_grid=future_time_grid,
            lag_features=lag_features,
            dense_past_target=dense_past_target,
            dense_past_observed_values=dense_past_observed_values,
            dense_past_time_grid=dense_past_time_grid,
        )

    if hasattr(model, "num_samples"):
        model.num_samples = int(num_samples)
    return model.predict_samples_from_past(
        past_target=past_target,
        past_observed_values=past_observed_values,
        scale=scale,
        mean=mean,
        num_samples=int(num_samples),
    )


def _to_batch_sample_time_dim(model_type: str, output: torch.Tensor) -> torch.Tensor:
    """Convert outputs to (B, S, T, D) for unified handling."""
    if model_type == "tsflow":
        if output.dim() == 3:
            return output.unsqueeze(-1)  # (B,S,T,1)
        if output.dim() == 4:
            return output  # (B,S,T,D)
        raise ValueError(f"Unexpected TSFlow output shape: {tuple(output.shape)}")

    # LCDE returns (S,B,T[,D])
    if output.dim() == 3:
        return output.permute(1, 0, 2).unsqueeze(-1)  # (B,S,T,1)
    if output.dim() == 4:
        return output.permute(1, 0, 2, 3)  # (B,S,T,D)
    raise ValueError(f"Unexpected LCDE output shape: {tuple(output.shape)}")


@torch.no_grad()
def _run_inference_for_horizon(
    model_type: str,
    model,
    batch: Dict[str, torch.Tensor],
    num_samples: int,
    horizon_length: int,
) -> torch.Tensor:
    """Run one-step or recursive rollout to produce exactly `horizon_length` steps.

    Returns:
        Tensor shaped (B, S, H, D), where H == horizon_length.
    """
    if horizon_length <= 0:
        raise ValueError(f"horizon_length must be > 0, got {horizon_length}.")

    chunk_len = int(getattr(model, "prediction_length"))
    if chunk_len <= 0:
        raise ValueError(f"Model prediction_length must be > 0, got {chunk_len}.")

    # If horizon fits in one native call, avoid recursive overhead.
    if horizon_length <= chunk_len:
        out = _run_single_inference(model_type, model, batch, num_samples)
        out_bstd = _to_batch_sample_time_dim(model_type, out)
        return out_bstd[:, :, :horizon_length, :]

    # Recursive rollout for longer horizons.
    past_target = batch["past_target"]
    past_observed_values = batch.get("past_observed_values", None)
    mean = batch.get("mean", None)
    scale = batch.get("scale", None)
    dense_past_target = batch.get("dense_past_target", None)
    dense_past_observed_values = batch.get("dense_past_observed_values", None)
    dense_past_time_grid = batch.get("dense_past_time_grid", None)

    past_window = int(past_target.shape[1])
    remaining = int(horizon_length)
    chunks: List[torch.Tensor] = []

    while remaining > 0:
        step_batch = {
            "past_target": past_target,
            "past_observed_values": past_observed_values,
            "mean": mean,
            "scale": scale,
            "dense_past_target": dense_past_target,
            "dense_past_observed_values": dense_past_observed_values,
            "dense_past_time_grid": dense_past_time_grid,
        }
        out = _run_single_inference(model_type, model, step_batch, num_samples)
        out_bstd = _to_batch_sample_time_dim(model_type, out)  # (B,S,T,D)
        take = min(chunk_len, remaining)
        chunk = out_bstd[:, :, :take, :]
        chunks.append(chunk)
        remaining -= take

        if remaining <= 0:
            break

        # Recursive conditioning: roll deterministic point estimate into past window.
        point_est = chunk.mean(dim=1)  # (B,take,D)
        if past_target.dim() == 2:
            point_est_for_past = point_est.squeeze(-1)  # (B,take)
        else:
            point_est_for_past = point_est  # (B,take,D)

        past_target = torch.cat([past_target, point_est_for_past], dim=1)
        past_target = past_target[:, -past_window:]

        if past_observed_values is not None:
            if past_observed_values.dim() == 2:
                obs_append = torch.ones(
                    past_observed_values.shape[0],
                    take,
                    dtype=torch.bool,
                    device=past_observed_values.device,
                )
            else:
                obs_append = torch.ones(
                    past_observed_values.shape[0],
                    take,
                    past_observed_values.shape[2],
                    dtype=torch.bool,
                    device=past_observed_values.device,
                )
            past_observed_values = torch.cat([past_observed_values, obs_append], dim=1)
            past_observed_values = past_observed_values[:, -past_window:]

    return torch.cat(chunks, dim=2)


def _infer_output_dims(output_bstd: torch.Tensor) -> Dict[str, int]:
    if output_bstd.dim() != 4:
        raise ValueError(f"Expected output shape (B,S,T,D), got {tuple(output_bstd.shape)}")
    bsz, nsamples, pred_len, target_dim = output_bstd.shape
    return {
        "batch_size": int(bsz),
        "num_samples": int(nsamples),
        "prediction_length": int(pred_len),
        "target_dim": int(target_dim),
    }


def measure_gluonts_inference_efficiency(
    spec: BenchmarkSpec,
    n_warmup: int,
    n_iterations: int,
    device: torch.device,
    batch_size_override: Optional[int],
    num_samples_override: Optional[int],
    horizon_length: Optional[int] = None,
) -> Dict[str, Any]:
    config = _read_yaml(spec.config_path)
    model = _build_model(spec, config, device)
    num_samples = _resolve_num_samples(config, num_samples_override)
    model_chunk_prediction_length = int(config["model_params"]["prediction_length"])
    horizon = int(horizon_length) if horizon_length is not None else model_chunk_prediction_length
    batch, batch_meta = _prepare_gluonts_batch(config, model, spec.model_type, device, batch_size_override)

    print(
        f"\nRunning {spec.label} ({spec.model_type}) "
        f"on dataset={batch_meta['dataset']} "
        f"batch={batch_meta['batch_size_actual']} "
        f"past={batch_meta['past_length']} pred={horizon} (chunk={model_chunk_prediction_length}) "
        f"samples={num_samples}"
    )

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = _run_inference_for_horizon(spec.model_type, model, batch, num_samples, horizon)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    latencies = []
    last_output = None
    with torch.no_grad():
        for _ in range(n_iterations):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            last_output = _run_inference_for_horizon(spec.model_type, model, batch, num_samples, horizon)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            end = time.perf_counter()
            latencies.append(end - start)

    if last_output is None:
        raise RuntimeError("Inference loop did not produce output.")

    dims = _infer_output_dims(last_output)
    latencies_np = np.asarray(latencies, dtype=np.float64)
    mean_latency = float(latencies_np.mean())
    std_latency = float(latencies_np.std())
    median_latency = float(np.median(latencies_np))
    p95_latency = float(np.percentile(latencies_np, 95))
    p99_latency = float(np.percentile(latencies_np, 99))

    forecasts_per_sec = dims["batch_size"] / mean_latency
    sample_points = dims["batch_size"] * dims["num_samples"] * dims["prediction_length"] * dims["target_dim"]
    sample_points_per_sec = sample_points / mean_latency

    metrics = {
        "label": spec.label,
        "model_type": spec.model_type,
        "checkpoint_path": spec.checkpoint_path,
        "config_path": spec.config_path,
        "device": str(device),
        "dataset": batch_meta["dataset"],
        "setting": batch_meta["setting"],
        "freq": batch_meta["freq"],
        "past_length": batch_meta["past_length"],
        "batch_size": dims["batch_size"],
        "num_samples": dims["num_samples"],
        "prediction_length": dims["prediction_length"],
        "model_chunk_prediction_length": model_chunk_prediction_length,
        "target_dim": dims["target_dim"],
        "mean_latency_ms": mean_latency * 1000.0,
        "std_latency_ms": std_latency * 1000.0,
        "median_latency_ms": median_latency * 1000.0,
        "p95_latency_ms": p95_latency * 1000.0,
        "p99_latency_ms": p99_latency * 1000.0,
        "forecasts_per_sec": forecasts_per_sec,
        "sample_points_per_sec": sample_points_per_sec,
        "n_warmup": int(n_warmup),
        "n_iterations": int(n_iterations),
    }
    return metrics


def print_metrics(metrics: Dict[str, Any]) -> None:
    print("\n" + "=" * 80)
    print(f"Inference Efficiency Report: {metrics['label']} ({metrics['model_type']})")
    print("=" * 80)
    print(f"Checkpoint: {metrics['checkpoint_path']}")
    print(f"Config:     {metrics['config_path']}")
    print(f"Device:     {metrics['device']}")
    print(
        "Data:       "
        f"dataset={metrics['dataset']} setting={metrics['setting']} freq={metrics['freq']} "
        f"past={metrics['past_length']} pred={metrics['prediction_length']} "
        f"(chunk={metrics['model_chunk_prediction_length']}) "
        f"batch={metrics['batch_size']} samples={metrics['num_samples']} target_dim={metrics['target_dim']}"
    )
    print(f"Iterations: warmup={metrics['n_warmup']} timed={metrics['n_iterations']}")
    print("-" * 80)
    print(f"Mean latency:   {metrics['mean_latency_ms']:.3f} ms")
    print(f"Std latency:    {metrics['std_latency_ms']:.3f} ms")
    print(f"Median latency: {metrics['median_latency_ms']:.3f} ms")
    print(f"P95 latency:    {metrics['p95_latency_ms']:.3f} ms")
    print(f"P99 latency:    {metrics['p99_latency_ms']:.3f} ms")
    print("-" * 80)
    print(f"Forecasts/sec:      {metrics['forecasts_per_sec']:.2f}")
    print(f"Sample points/sec:  {metrics['sample_points_per_sec']:.2f}")
    print("=" * 80 + "\n")


def print_comparison(metrics_list: List[Dict[str, Any]]) -> None:
    if len(metrics_list) <= 1:
        return

    print("\n" + "=" * 80)
    print("Model Comparison Summary")
    print("=" * 80)
    by_horizon = {}
    for m in metrics_list:
        by_horizon.setdefault(int(m["prediction_length"]), []).append(m)

    for horizon in sorted(by_horizon):
        rows = by_horizon[horizon]
        print(f"Horizon={horizon}")
        print(f"{'Label':<24} {'Type':<10} {'Mean Lat (ms)':<15} {'Forecast/s':<15} {'SamplePts/s':<15}")
        print("-" * 80)
        for m in rows:
            print(
                f"{m['label']:<24} {m['model_type']:<10} {m['mean_latency_ms']:<15.3f} "
                f"{m['forecasts_per_sec']:<15.2f} {m['sample_points_per_sec']:<15.2f}"
            )
        latencies = [m["mean_latency_ms"] for m in rows]
        slowest = max(latencies)
        fastest = rows[int(np.argmin(latencies))]
        print(f"Fastest @ horizon {horizon}: {fastest['label']} ({fastest['mean_latency_ms']:.3f} ms)")
        print("Speedup relative to slowest:")
        for m in rows:
            print(f"  {m['label']}: {slowest / m['mean_latency_ms']:.3f}x")
        print("-" * 80)
    print("=" * 80 + "\n")


def plot_horizon_sweep(metrics_list: List[Dict[str, Any]], save_path: str) -> None:
    if len(metrics_list) == 0:
        return

    df = pd.DataFrame(metrics_list)
    if "prediction_length" not in df.columns or df["prediction_length"].nunique() <= 1:
        return

    df = df.sort_values(["prediction_length", "label"]).copy()

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

    sns.lineplot(
        data=df,
        x="prediction_length",
        y="mean_latency_ms",
        hue="label",
        style="label",
        markers=True,
        dashes=False,
        ax=axes[0],
    )
    axes[0].set_title("Latency vs Forecast Horizon")
    axes[0].set_xlabel("Prediction Length")
    axes[0].set_ylabel("Mean Latency (ms)")
    axes[0].set_xscale("log", base=2)

    sns.lineplot(
        data=df,
        x="prediction_length",
        y="sample_points_per_sec",
        hue="label",
        style="label",
        markers=True,
        dashes=False,
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title("Throughput vs Forecast Horizon")
    axes[1].set_xlabel("Prediction Length")
    axes[1].set_ylabel("Sample Points / sec")
    axes[1].set_xscale("log", base=2)

    save_path_obj = Path(save_path)
    save_path_obj.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path_obj, dpi=200)
    plt.close(fig)
    print(f"Saved seaborn horizon plot to: {save_path_obj}")


def _build_specs(
    checkpoint_paths: List[str],
    config_paths: Optional[List[str]],
    model_types: Optional[List[str]],
    labels: Optional[List[str]],
) -> List[BenchmarkSpec]:
    n = len(checkpoint_paths)
    if config_paths is not None and len(config_paths) != n:
        raise ValueError("Number of config paths must match number of checkpoint paths.")
    if model_types is not None and len(model_types) != n:
        raise ValueError("Number of model types must match number of checkpoint paths.")
    if labels is not None and len(labels) != n:
        raise ValueError("Number of labels must match number of checkpoint paths.")

    specs: List[BenchmarkSpec] = []
    for idx, ckpt_path in enumerate(checkpoint_paths):
        cfg_path = config_paths[idx] if config_paths is not None else _infer_config_path(ckpt_path)
        cfg = _read_yaml(cfg_path)

        inferred_model_type = _infer_model_type(cfg)
        model_type = model_types[idx].lower() if model_types is not None else "auto"
        if model_type not in SUPPORTED_MODEL_TYPES:
            raise ValueError(f"Unsupported model type {model_type!r}. Choose from {sorted(SUPPORTED_MODEL_TYPES)}.")
        if model_type == "auto":
            model_type = inferred_model_type
        elif model_type != inferred_model_type:
            print(
                f"WARNING: model_type={model_type!r} does not match config {cfg_path} "
                f"(inferred {inferred_model_type!r}). Using {inferred_model_type!r}."
            )
            model_type = inferred_model_type

        label = labels[idx] if labels is not None else f"model_{idx}"
        specs.append(
            BenchmarkSpec(
                checkpoint_path=ckpt_path,
                config_path=cfg_path,
                model_type=model_type,
                label=label,
            )
        )
    return specs


def main():
    parser = argparse.ArgumentParser(description="Measure GluonTS inference efficiency for TSFlowCond and LCDE models.")
    parser.add_argument(
        "--checkpoint_paths",
        type=str,
        required=True,
        help="Python list of checkpoint paths, e.g. \"['results/1/best_checkpoint.ckpt','results/2/best_checkpoint.ckpt']\"",
    )
    parser.add_argument(
        "--config_paths",
        type=str,
        default=None,
        help="Optional Python list of config paths. If omitted, inferred from checkpoint directory.",
    )
    parser.add_argument(
        "--model_types",
        type=str,
        default=None,
        help="Optional Python list with entries in {'tsflow','lcde','auto'}. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Optional Python list of labels for reporting.",
    )
    parser.add_argument("--batch_size", type=int, default=None, help="Optional batch size override for inference loader.")
    parser.add_argument("--num_samples", type=int, default=None, help="Optional sample count override for all models.")
    parser.add_argument("--n_warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--n_iterations", type=int, default=100, help="Timed inference iterations.")
    parser.add_argument(
        "--horizon_powers",
        type=str,
        default="[0]",
        help="List of powers for horizon sweep: horizon = base_prediction_length * (2**p). Ignored if --prediction_lengths is provided.",
    )
    parser.add_argument(
        "--prediction_lengths",
        type=str,
        default=None,
        help="Optional explicit list of absolute prediction lengths to benchmark.",
    )
    parser.add_argument(
        "--base_prediction_length",
        type=int,
        default=None,
        help="Optional base prediction length for --horizon_powers. Defaults to first config's model_params.prediction_length.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device, e.g. cuda, cuda:0, cpu.")
    parser.add_argument("--save_json", type=str, default=None, help="Optional path to save metrics JSON.")
    parser.add_argument(
        "--plot_path",
        type=str,
        default="./results/inference_efficiency_horizon.png",
        help="Path to save seaborn horizon sweep plot (only when multiple horizons are evaluated).",
    )
    args = parser.parse_args()

    checkpoint_paths = _parse_list_arg(args.checkpoint_paths, "checkpoint_paths")
    config_paths = _parse_list_arg(args.config_paths, "config_paths") if args.config_paths is not None else None
    model_types = _parse_list_arg(args.model_types, "model_types") if args.model_types is not None else None
    labels = _parse_list_arg(args.labels, "labels") if args.labels is not None else None

    specs = _build_specs(
        checkpoint_paths=checkpoint_paths,
        config_paths=config_paths,
        model_types=model_types,
        labels=labels,
    )

    if args.prediction_lengths is not None:
        horizon_lengths = sorted({int(h) for h in _parse_int_list_arg(args.prediction_lengths, "prediction_lengths")})
    else:
        powers = _parse_int_list_arg(args.horizon_powers, "horizon_powers")
        if args.base_prediction_length is not None:
            base_prediction_length = int(args.base_prediction_length)
        else:
            first_cfg = _read_yaml(specs[0].config_path)
            base_prediction_length = int(first_cfg["model_params"]["prediction_length"])
        horizon_lengths = sorted({base_prediction_length * (2 ** int(p)) for p in powers})

    if any(h <= 0 for h in horizon_lengths):
        raise ValueError(f"All horizon lengths must be > 0, got {horizon_lengths}.")

    print(f"Benchmark horizons: {horizon_lengths}")

    device = _resolve_device(args.device)

    all_metrics: List[Dict[str, Any]] = []
    for horizon in horizon_lengths:
        for spec in specs:
            metrics = measure_gluonts_inference_efficiency(
                spec=spec,
                n_warmup=args.n_warmup,
                n_iterations=args.n_iterations,
                device=device,
                batch_size_override=args.batch_size,
                num_samples_override=args.num_samples,
                horizon_length=horizon,
            )
            print_metrics(metrics)
            all_metrics.append(metrics)

    print_comparison(all_metrics)
    plot_horizon_sweep(all_metrics, args.plot_path)

    if args.save_json is not None:
        save_path = Path(args.save_json)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as fp:
            json.dump(all_metrics, fp, indent=2)
        print(f"Saved metrics to: {save_path}")


if __name__ == "__main__":
    main()
