import argparse
import logging
import os
import tempfile
from pathlib import Path

import numpy as np
import pykeops
import pytorch_lightning as pl
import torch
import yaml
import pandas as pd
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.loader import TrainDataLoader
from gluonts.dataset.split import OffsetSplitter, split
from gluonts.dataset.util import period_index
from gluonts.evaluation import (
    Evaluator,
    make_evaluation_predictions,
)
from gluonts.itertools import Cached
from gluonts.model.forecast import SampleForecast
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.batchify import batchify
from gluonts.transform import Chain
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm.auto import tqdm

from gslice.callbacks.gluonts_eval_plot_callback import GluonTSEvalPlotCallback
from gslice.callbacks.training_batch_plot_callback import TrainingBatchPlotCallback
from gslice.dataset import get_dataset_name_from_params, get_gts_dataset_from_params, infer_target_dim
from gslice.irregular import (
    DENSE_PAST_OBSERVED_FIELD,
    DENSE_PAST_TARGET_FIELD,
    DENSE_PAST_TIME_GRID_FIELD,
    has_irregular_grid,
    maybe_get_irregular_input_names,
    resolve_model_window_lengths,
    resolve_source_window_lengths,
    FUTURE_TIME_GRID_FIELD,
    PAST_TIME_GRID_FIELD,
    get_irregular_grid_spec,
)
from gslice.model import TSFlowCond
from gslice.utils import create_transforms
from gslice.utils.transforms import IrregularInstanceTransform
from gslice.utils.util import (
    ConcatDataset,
    add_config_to_argparser,
    create_splitter,
    filter_metrics,
    temporary_random_seed,
)
from gslice.utils.variables import frequencies_match, get_lags_for_freq, get_relative_time_step, get_season_length

try:
    pykeops.clean_pykeops()
except FileNotFoundError:
    # Multiple Slurm jobs can start together and race while deleting the same
    # shared KeOps cache entry. The per-job build folder below is still fresh.
    pass
temp_build_folder = tempfile.mkdtemp(prefix="pykeops_build_")
pykeops.set_build_folder(temp_build_folder)


def create_model(setting, target_dim, model_params):
    if setting != "univariate":
        raise ValueError(f"Unsupported setting {setting!r}; only 'univariate' is supported.")
    model = TSFlowCond(
        setting=setting,
        target_dim=target_dim,
        context_length=model_params["context_length"],
        prediction_length=model_params["prediction_length"],
        backbone_params=model_params["backbone_params"],
        prior_params=model_params["prior_params"],
        optimizer_params=model_params["optimizer_params"],
        ema_params=model_params["ema_params"],
        frequency=model_params["freq"],
        normalization=model_params["normalization"],
        use_lags=model_params["use_lags"],
        use_ema=model_params["use_ema"],
        num_steps=model_params["num_steps"],
        solver=model_params["solver"],
        matching=model_params["matching"],
        gp_time_mode=model_params.get("gp_time_mode", "discrete"),
        num_samples=model_params.get("num_samples", 1),
        lags_seq=model_params.get("lags_seq"),
        prior_context_length_override=model_params.get("prior_context_length_override"),
        gp_fit_context_only=model_params.get("gp_fit_context_only", False),
    )
    model.to(model_params["device"])
    return model


def _resolve_irregular_model_setup(dataset_params, freq: str, model_params: dict) -> tuple[int, int, int, int, bool]:
    uses_irregular_grid = has_irregular_grid(dataset_params)
    if not uses_irregular_grid:
        prediction_length = int(model_params.get("prediction_length"))
        return prediction_length, prediction_length, prediction_length, prediction_length, False

    context_length, prediction_length = resolve_model_window_lengths(dataset_params, fallback_freq=freq)
    source_context_length, source_prediction_length = resolve_source_window_lengths(dataset_params, fallback_freq=freq)
    model_params["context_length"] = int(context_length)
    model_params["prediction_length"] = int(prediction_length)
    model_params["lags_seq"] = get_lags_for_freq(
        freq,
        context_length=int(source_context_length),
        max_lag=max(int(source_context_length), int(source_context_length) * 28),
    )
    prior_params = dict(model_params.get("prior_params", {}) or {})
    context_freqs = int(prior_params.get("context_freqs", 1))
    if bool(model_params.get("gp_fit_context_only", False)):
        model_params["prior_context_length_override"] = int(context_length)
    else:
        model_params["prior_context_length_override"] = int(context_freqs * source_prediction_length)
    return (
        int(context_length),
        int(prediction_length),
        int(source_context_length),
        int(source_prediction_length),
        True,
    )


def _build_instance_transform(splitter, dataset_params, model_params: dict, freq: str):
    if not has_irregular_grid(dataset_params):
        return splitter

    return Chain(
        [
            splitter,
            IrregularInstanceTransform(
                irregular_spec=get_irregular_grid_spec(dataset_params, fallback_freq=freq),
                lag_steps=(
                    list(model_params.get("lags_seq", []) or [])
                    if bool(model_params.get("use_lags", True))
                    else []
                ),
            ),
        ]
    )


def _resolve_checkpoint_monitor(
    *,
    eval_every: int,
    uses_irregular_grid: bool,
    dataset_name: str,
) -> tuple[str, str]:
    # Irregular runs still execute the validation callback, but the best
    # checkpoint should track the training objective rather than validation CRPS.
    if eval_every > 0 and not uses_irregular_grid:
        return "val_CRPS", dataset_name + "-{epoch:03d}-{val_CRPS:.3f}"
    return "train_loss", dataset_name + "-{epoch:03d}-{train_loss:.3f}"


def _resolve_splitter_past_length(model: TSFlowCond, model_params: dict, source_context_length: int, uses_irregular_grid: bool) -> int:
    lag_context = int(source_context_length if uses_irregular_grid else model_params["context_length"])
    return max(
        lag_context + max(model.lags_seq),
        int(model.prior_context_length),
    )


def _build_irregular_eval_entry(input_entry, label_entry, dataset_params: dict, freq: str, model_params: dict):
    dense_past_target = np.asarray(input_entry[FieldName.TARGET], dtype=np.float32)
    dense_future_target = np.asarray(label_entry[FieldName.TARGET], dtype=np.float32)
    full_time_grid = np.asarray(input_entry["time_grid"], dtype=np.float32)
    full_observed = np.asarray(
        input_entry.get(
            "observed_values",
            np.ones((dense_past_target.shape[0] + dense_future_target.shape[0],), dtype=np.float32),
        ),
        dtype=np.float32,
    )
    past_len = int(dense_past_target.shape[0])
    future_len = int(dense_future_target.shape[0])
    dense_entry = {
        "past_target": dense_past_target,
        "future_target": dense_future_target,
        PAST_TIME_GRID_FIELD: full_time_grid[:past_len],
        FUTURE_TIME_GRID_FIELD: full_time_grid[past_len : past_len + future_len],
        "past_observed_values": full_observed[:past_len],
        "future_observed_values": full_observed[past_len : past_len + future_len],
        "mean": np.asarray(input_entry.get("mean"), dtype=np.float32),
        FieldName.ITEM_ID: input_entry.get(FieldName.ITEM_ID, label_entry.get(FieldName.ITEM_ID)),
        FieldName.FORECAST_START: label_entry[FieldName.START],
        FieldName.START: input_entry[FieldName.START],
    }
    if "time_feat" in input_entry:
        full_time_feat = np.asarray(input_entry["time_feat"])
        dense_entry["past_feat_time"] = np.swapaxes(full_time_feat[..., :past_len], 0, -1)
        dense_entry["future_feat_time"] = np.swapaxes(full_time_feat[..., past_len : past_len + future_len], 0, -1)
    transform = IrregularInstanceTransform(
        irregular_spec=get_irregular_grid_spec(dataset_params, fallback_freq=freq),
        lag_steps=(
            list(model_params.get("lags_seq", []) or [])
            if bool(model_params.get("use_lags", True))
            else []
        ),
    )
    return transform.map_transform(dense_entry, is_train=False)


def _irregular_entry_to_dataframe(entry) -> pd.DataFrame:
    past_target = np.asarray(entry["past_target"])
    future_target = np.asarray(entry["future_target"])
    full_target = np.concatenate([past_target, future_target], axis=-1)

    forecast_start = entry.get(FieldName.FORECAST_START)
    start = entry[FieldName.START]
    if forecast_start is not None:
        start = forecast_start - int(past_target.shape[-1])

    index = period_index({FieldName.START: start, FieldName.TARGET: full_target})
    return pd.DataFrame(full_target.transpose(), index=index)


def _chunked_batchify(entries, batch_size: int):
    for start in range(0, len(entries), batch_size):
        yield batchify(entries[start : start + batch_size])


def _clear_s4_runtime_caches(module: torch.nn.Module) -> None:
    for submodule in module.modules():
        for attr in ("omega", "z"):
            if hasattr(submodule, attr):
                value = getattr(submodule, attr)
                if torch.is_tensor(value):
                    delattr(submodule, attr)


def evaluate_conditional(
    model_params,
    model: TSFlowCond,
    test_dataset,
    transformation,
    trainer,
    num_samples=100,
    eval_seed=None,
    batch_size=None,
):
    if batch_size is None:
        batch_size = int(model_params["dataset_params"]["batch_size"])
    else:
        batch_size = int(batch_size)
    logging.info(f"Evaluating with {num_samples} samples and batch_size={batch_size}.")
    results = {}
    model.num_samples = int(num_samples)
    uses_irregular_grid = bool(model_params.get("uses_irregular_grid", False))

    with temporary_random_seed(eval_seed):
        if model.setting != "univariate":
            raise ValueError(f"Unsupported setting {model.setting!r}; only 'univariate' is supported.")
        evaluator = Evaluator(num_workers=1)

        if uses_irregular_grid:
            transformed_testdata = transformation.apply(test_dataset, is_train=False)
            window_length = int(model_params.get("source_prediction_length", model_params["prediction_length"]))
            _, test_template = split(test_dataset, offset=-window_length)
            test_pairs = list(test_template.generate_instances(window_length))
            input_entries = [
                next(iter(transformation.apply([input_entry], is_train=False)))
                for input_entry, _ in test_pairs
            ]
            transformed_entries = [
                _build_irregular_eval_entry(
                    input_entry=input_entry,
                    label_entry=label_entry,
                    dataset_params=model_params["dataset_params"],
                    freq=model_params["freq"],
                    model_params=model_params,
                )
                for input_entry, (_, label_entry) in zip(input_entries, test_pairs)
            ]
            forecasts = []
            tss = [_irregular_entry_to_dataframe(entry) for entry in transformed_entries]

            with torch.no_grad():
                for batch_idx, batch in enumerate(
                    tqdm(
                        _chunked_batchify(transformed_entries, batch_size),
                        total=(len(transformed_entries) + batch_size - 1) // batch_size,
                    )
                ):
                    past_target = torch.as_tensor(batch["past_target"], dtype=torch.float32, device=model.device)
                    past_observed_values = torch.as_tensor(
                        batch["past_observed_values"], dtype=torch.float32, device=model.device
                    )
                    mean = torch.as_tensor(batch["mean"], dtype=torch.float32, device=model.device)
                    past_time_grid = torch.as_tensor(batch[PAST_TIME_GRID_FIELD], dtype=torch.float32, device=model.device)
                    future_time_grid = torch.as_tensor(batch[FUTURE_TIME_GRID_FIELD], dtype=torch.float32, device=model.device)
                    lag_features = batch.get("lag_features", None)
                    if lag_features is not None:
                        lag_features = torch.as_tensor(lag_features, dtype=torch.float32, device=model.device)
                    dense_past_target = batch.get(DENSE_PAST_TARGET_FIELD, None)
                    if dense_past_target is not None:
                        dense_past_target = torch.as_tensor(
                            dense_past_target,
                            dtype=torch.float32,
                            device=model.device,
                        )
                    dense_past_observed_values = batch.get(DENSE_PAST_OBSERVED_FIELD, None)
                    if dense_past_observed_values is not None:
                        dense_past_observed_values = torch.as_tensor(
                            dense_past_observed_values,
                            dtype=torch.float32,
                            device=model.device,
                        )
                    dense_past_time_grid = batch.get(DENSE_PAST_TIME_GRID_FIELD, None)
                    if dense_past_time_grid is not None:
                        dense_past_time_grid = torch.as_tensor(
                            dense_past_time_grid,
                            dtype=torch.float32,
                            device=model.device,
                        )

                    forecast_batch = model.forward(
                        past_target=past_target,
                        past_observed_values=past_observed_values,
                        mean=mean,
                        past_time_grid=past_time_grid,
                        future_time_grid=future_time_grid,
                        lag_features=lag_features,
                        dense_past_target=dense_past_target,
                        dense_past_observed_values=dense_past_observed_values,
                        dense_past_time_grid=dense_past_time_grid,
                    ).detach().cpu().numpy()

                    offset = batch_idx * batch_size
                    batch_pairs = test_pairs[offset : offset + int(forecast_batch.shape[0])]
                    for idx, (_, label_entry) in enumerate(batch_pairs):
                        samples = forecast_batch[idx]
                        if samples.ndim == 3 and samples.shape[-1] == 1:
                            samples = np.squeeze(samples, axis=-1)
                        forecasts.append(
                            SampleForecast(
                                samples=samples,
                                start_date=label_entry[FieldName.START],
                                item_id=label_entry.get(FieldName.ITEM_ID),
                            )
                        )
        else:
            transformed_testdata = transformation.apply(test_dataset, is_train=False)
            test_splitter = create_splitter(
                past_length=_resolve_splitter_past_length(
                    model,
                    model_params,
                    int(model_params.get("source_context_length", model_params["context_length"])),
                    uses_irregular_grid,
                ),
                future_length=int(model_params.get("source_prediction_length", model_params["prediction_length"])),
                mode="test",
                include_time_grid=uses_irregular_grid,
            )

            test_transform = _build_instance_transform(
                test_splitter,
                model_params["dataset_params"],
                model_params,
                model_params["freq"],
            )
            predictor = model.get_predictor(
                test_transform,
                batch_size=batch_size,
                device=model_params["device"],
                input_names=maybe_get_irregular_input_names(
                    model_params["dataset_params"],
                    use_lags=bool(model_params.get("use_lags", True)),
                ),
            )
            forecast_it, ts_it = make_evaluation_predictions(
                dataset=transformed_testdata,
                predictor=predictor,
                num_samples=num_samples,
            )
            forecasts = list(tqdm(forecast_it, total=len(transformed_testdata)))
            tss = list(ts_it)
        metrics, _ = evaluator(tss, forecasts)
    metrics["CRPS"] = metrics["mean_wQuantileLoss"]
    select = ["CRPS", "ND", "NRMSE"]
    metrics = filter_metrics(metrics, select)
    [
        logger.log_metrics(
            {f"test_{key}": val for key, val in metrics.items()},
            step=trainer.global_step + 1,
        )
        for logger in trainer.loggers
    ]
    results["test"] = dict(**metrics)
    return results


def main(
    model,
    setting,
    model_params,
    dataset_params,
    trainer_params,
    evaluation_params,
    logdir=None,
    seed=None,
    config=None,
    loggers=None,
):
    if loggers is None:
        # use csv logger if no loggers provided to ensure metrics are saved
        loggers = [pl.loggers.CSVLogger(save_dir=logdir, name="csv_logs")]
    if logdir:
        Path(logdir).mkdir(parents=True, exist_ok=True)
    pl.seed_everything(seed)
    # Load parameters
    dataset_name = get_dataset_name_from_params(dataset_params)
    dataset = get_gts_dataset_from_params(
        dataset_params,
        regenerate=bool(dataset_params.get("regenerate", False)),
    )
    freq = str(dataset.metadata.freq)
    if setting != "univariate":
        raise ValueError(f"Unsupported setting {setting!r}; only 'univariate' is supported.")
    dense_prediction_length = int(dataset.metadata.prediction_length)
    model_params["freq"] = freq
    model_params["prediction_length"] = dense_prediction_length
    model_params["context_length"] = dense_prediction_length
    (
        context_length,
        prediction_length,
        source_context_length,
        source_prediction_length,
        uses_irregular_grid,
    ) = _resolve_irregular_model_setup(dataset_params, freq, model_params)
    model_params["source_context_length"] = int(source_context_length)
    model_params["source_prediction_length"] = int(source_prediction_length)
    model_params["uses_irregular_grid"] = bool(uses_irregular_grid)
    model_params["dataset_params"] = dict(dataset_params)
    target_dim = infer_target_dim(dataset)
    # Create model
    model = create_model(setting, target_dim, model_params)

    # Setup dataset and data loading
    if not frequencies_match(dataset.metadata.freq, freq):
        raise ValueError(f"Config freq ({freq}) does not match dataset metadata ({dataset.metadata.freq}).")
    if dataset.metadata.prediction_length != dense_prediction_length:
        raise ValueError(
            f"Config prediction_length ({dense_prediction_length}) does not match dataset metadata "
            f"({dataset.metadata.prediction_length})."
        )

    num_rolling_evals = int(len(dataset.test) / len(dataset.train))
    time_features = time_features_from_frequency_str(freq)
    transformation = create_transforms(
        time_features=time_features,
        prediction_length=int(source_prediction_length),
        freq=get_season_length(freq),
        train_length=len(dataset.train),
        time_step_hours=float(get_relative_time_step(freq)),
        include_time_grid=uses_irregular_grid,
    )
    training_data = dataset.train
    test_data = dataset.test

    training_splitter = create_splitter(
        past_length=_resolve_splitter_past_length(
            model,
            model_params,
            int(source_context_length),
            uses_irregular_grid,
        ),
        future_length=int(source_prediction_length),
        mode="train",
        include_time_grid=uses_irregular_grid,
    )
    training_splitter = _build_instance_transform(training_splitter, dataset_params, model_params, freq)
    callbacks = []
    callback_eval_dataset = test_data
    callback_eval_split_name = "test"
    validation_num_samples = evaluation_params.get(
        "validation_num_samples",
        evaluation_params.get("num_samples", 16),
    )
    test_num_samples = evaluation_params.get("test_num_samples", 100)
    validation_eval_seed = evaluation_params.get(
        "validation_eval_seed",
        evaluation_params.get("eval_seed", seed),
    )
    test_eval_seed = evaluation_params.get(
        "test_eval_seed",
        evaluation_params.get("eval_seed", seed),
    )

    if evaluation_params["use_validation_set"]:
        train_val_splitter = OffsetSplitter(offset=-int(source_prediction_length) * num_rolling_evals)
        training_data, val_gen = train_val_splitter.split(training_data)
        transformed_data = transformation.apply(training_data, is_train=True)
        val_data = val_gen.generate_instances(int(source_prediction_length), num_rolling_evals)
        callback_eval_dataset = ConcatDataset(val_data)
        callback_eval_split_name = "validation"
    else:
        transformed_data = transformation.apply(training_data, is_train=True)

    # -- Periodic validation callback with visualization --
    eval_every = evaluation_params.get("eval_every", 0)
    if eval_every > 0:
        val_plot_dir = Path(logdir) / "val_plots"
        eval_callback = GluonTSEvalPlotCallback(
            test_dataset=callback_eval_dataset,
            transformation=transformation,
            model_params=model_params,
            setting=setting,
            eval_every=eval_every,
            num_samples=validation_num_samples,
            max_eval_instances=evaluation_params.get("max_eval_instances", None),
            batch_size=dataset_params.get("test_batch_size", dataset_params["batch_size"]),
            max_show=evaluation_params.get("max_show", 8),
            save_dir=str(val_plot_dir),
            enable_worst_crps_plot=evaluation_params.get("enable_worst_crps_plot", False),
            enable_quantile_loss_plots=evaluation_params.get("enable_quantile_loss_plots", False),
            checkpoint_dir=logdir,
            eval_seed=validation_eval_seed,
        )
        callbacks.append(eval_callback)
        logging.info(
            f"Periodic validation every {eval_every} epochs on the {callback_eval_split_name} split → {val_plot_dir}"
        )

    # -- Periodic training batch visualization callback --
    train_plot_every = evaluation_params.get("train_plot_every", 0)
    if train_plot_every > 0:
        train_plot_dir = Path(logdir) / "train_plots"
        train_batch_callback = TrainingBatchPlotCallback(
            plot_every=train_plot_every,
            num_batches=evaluation_params.get("train_plot_num_batches", 4),
            num_samples=evaluation_params.get("train_plot_num_samples", 16),
            max_show=evaluation_params.get("train_plot_max_show", 8),
            save_dir=str(train_plot_dir),
        )
        callbacks.append(train_batch_callback)
        logging.info(f"Training batch visualization every {train_plot_every} epochs → {train_plot_dir}")

    log_monitor, filename = _resolve_checkpoint_monitor(
        eval_every=eval_every,
        uses_irregular_grid=uses_irregular_grid,
        dataset_name=dataset_name,
    )

    data_loader = TrainDataLoader(
        Cached(transformed_data),
        batch_size=dataset_params["batch_size"],
        stack_fn=batchify,
        transform=training_splitter,
        num_batches_per_epoch=dataset_params["num_batches_per_epoch"],
        shuffle_buffer_length=10000,
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor=f"{log_monitor}",
        mode="min",
        filename=filename,
        save_last=True,
        save_weights_only=True,
    )

    callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else None,
        devices=[int(model_params["device"].split(":")[-1])],
        default_root_dir=logdir,
        logger=loggers,
        enable_progress_bar=False,
        callbacks=callbacks,
        **trainer_params,
    )

    logging.info(f"Logging to {logdir}")
    trainer.fit(model, train_dataloaders=data_loader)
    logging.info("Training completed.")

    best_ckpt_path = Path(logdir) / "best_checkpoint.ckpt"
    if not best_ckpt_path.exists():
        torch.save(
            torch.load(checkpoint_callback.best_model_path)["state_dict"],
            best_ckpt_path,
        )
    logging.info(f"Loading {best_ckpt_path}.")
    best_state_dict = torch.load(best_ckpt_path)
    model.load_state_dict(best_state_dict, strict=True)
    model.to(model_params["device"])
    model.eval()
    _clear_s4_runtime_caches(model)

    metrics = (
        evaluate_conditional(
            model_params,
            model,
            test_data,
            transformation,
            trainer,
            num_samples=test_num_samples,
            eval_seed=test_eval_seed,
            batch_size=dataset_params["batch_size"],
        )
        if evaluation_params.get("do_final_eval", True)
        else "Final eval not performed"
    )

    logger_version = "none"
    if trainer.logger is not None and hasattr(trainer.logger, "version"):
        logger_version = trainer.logger.version

    with open(Path(logdir) / "results.yaml", "w") as fp:
        yaml.dump(
            {
                "config": config,
                "version": logger_version,
                "metrics": metrics,
            },
            fp,
        )
    return metrics


def _resolve_logdir(cli_logdir: str) -> str:
    """Build a results directory.

    An explicit ``--logdir`` should win even on Slurm so runner scripts can keep
    all artifacts under a single job directory. If the default CLI logdir is
    left unchanged, fall back to ``results/<SLURM_JOB_ID>`` on Slurm.
    """
    slurm_id = os.environ.get("SLURM_JOB_ID", None)
    default_cli_logdir = "./logs"
    if cli_logdir and cli_logdir != default_cli_logdir:
        logdir = cli_logdir
        if slurm_id:
            logging.info(f"SLURM_JOB_ID detected → using explicit logdir: {logdir}")
    elif slurm_id:
        logdir = str(Path("results") / slurm_id)
        logging.info(f"SLURM_JOB_ID detected → results dir: {logdir}")
    else:
        logdir = cli_logdir
    Path(logdir).mkdir(parents=True, exist_ok=True)
    return logdir


if __name__ == "__main__":
    # Setup Logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to yaml config")
    parser.add_argument("--logdir", type=str, default="./logs", help="Path to results dir")
    args, _ = parser.parse_known_args()

    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp)

    parser = add_config_to_argparser(config=config, parser=parser)
    args = parser.parse_args()
    config_updates = vars(args)
    for k in config.keys() & config_updates.keys():
        orig_val = config[k]
        updated_val = config_updates[k]
        if updated_val != orig_val:
            logging.info(f"Updated key '{k}': {orig_val} -> {updated_val}")
    config.update(config_updates)

    logdir = _resolve_logdir(args.logdir)

    config["logdir"] = logdir
    main(**config, loggers=None)
