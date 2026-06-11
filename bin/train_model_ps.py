import argparse
import logging
import tempfile
from pathlib import Path

import pykeops
import pytorch_lightning as pl
import torch
import yaml
from aim.pytorch_lightning import AimLogger
from gluonts.dataset.loader import TrainDataLoader
from gluonts.dataset.split import OffsetSplitter
from gluonts.evaluation import (
    Evaluator,
    make_evaluation_predictions,
)
from gluonts.itertools import Cached
from gluonts.time_feature import time_features_from_frequency_str
from gluonts.torch.batchify import batchify
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm.auto import tqdm

from gslice.callback import EvaluateCallback
from gslice.dataset import get_gts_dataset_from_params, infer_target_dim
from gslice.model import TSFlowPS
from gslice.utils import create_transforms
from gslice.utils.util import ConcatDataset, add_config_to_argparser, create_splitter, filter_metrics
from gslice.utils.variables import get_season_length

pykeops.clean_pykeops()
temp_build_folder = tempfile.mkdtemp(prefix="pykeops_build_")
pykeops.set_build_folder(temp_build_folder)


def create_model(setting, target_dim, model_params):
    if setting != "univariate":
        raise ValueError(f"Unsupported setting {setting!r}; only 'univariate' is supported.")
    model = TSFlowPS(
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
        iterations=model_params["iterations"],
        noise_level=model_params["noise_level"],
        alpha=model_params["alpha"],
        num_samples=model_params.get("num_samples", 1),
    )
    model.to(model_params["device"])
    return model


def evaluate_conditional(
    model_params,
    model: TSFlowPS,
    test_dataset,
    transformation,
    trainer,
    num_samples=100,
):
    logging.info(f"Evaluating with {num_samples} samples.")
    results = {}

    transformed_testdata = transformation.apply(test_dataset, is_train=False)
    test_splitter = create_splitter(
        past_length=max(
            model_params["context_length"] + max(model.lags_seq),
            model.prior_context_length,
        ),
        future_length=model_params["prediction_length"],
        mode="test",
    )

    test_transform = test_splitter

    model.num_samples = num_samples
    predictor = model.get_predictor(
        test_transform,
        batch_size=512 // num_samples,
        device=model_params["device"],
    )
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=transformed_testdata,
        predictor=predictor,
        num_samples=num_samples,
    )
    forecasts = list(tqdm(forecast_it, total=len(transformed_testdata)))
    tss = list(ts_it)

    if model.setting != "univariate":
        raise ValueError(f"Unsupported setting {model.setting!r}; only 'univariate' is supported.")
    evaluator = Evaluator(num_workers=1)
    metrics, _ = evaluator(tss, forecasts)
    metrics["CRPS"] = metrics["mean_wQuantileLoss"]

    metrics = filter_metrics(metrics)
    [
        logger.log_metrics(
            {f"test_{key}_": val for key, val in metrics.items()},
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
    loggers=[],
):
    if logdir:
        Path(logdir).mkdir(parents=True, exist_ok=True)
    pl.seed_everything(seed)
    if setting != "univariate":
        raise ValueError(f"Unsupported setting {setting!r}; only 'univariate' is supported.")
    # Load parameters
    dataset = get_gts_dataset_from_params(
        dataset_params,
        regenerate=bool(dataset_params.get("regenerate", False)),
    )
    freq = str(dataset.metadata.freq)
    prediction_length = int(dataset.metadata.prediction_length)
    context_length = int(prediction_length)
    model_params["freq"] = freq
    model_params["prediction_length"] = prediction_length
    model_params["context_length"] = context_length
    target_dim = infer_target_dim(dataset)
    # Create model
    model = create_model(setting, target_dim, model_params)

    # Setup dataset and data loading
    assert dataset.metadata.freq == freq
    assert dataset.metadata.prediction_length == prediction_length

    num_rolling_evals = int(len(dataset.test) / len(dataset.train))
    time_features = time_features_from_frequency_str(freq)
    transformation = create_transforms(
        time_features=time_features,
        prediction_length=model_params["prediction_length"],
        freq=get_season_length(freq),
        train_length=len(dataset.train),
    )
    training_data = dataset.train
    test_data = dataset.test

    training_splitter = create_splitter(
        past_length=max(
            model_params["context_length"] + max(model.lags_seq),
            model.prior_context_length,
        ),
        future_length=model_params["prediction_length"],
        mode="train",
    )
    callbacks = []
    if evaluation_params["use_validation_set"]:
        train_val_splitter = OffsetSplitter(offset=-model_params["prediction_length"] * num_rolling_evals)
        training_data, val_gen = train_val_splitter.split(training_data)
        transformed_data = transformation.apply(training_data, is_train=True)
        val_data = val_gen.generate_instances(model_params["prediction_length"], num_rolling_evals)
        transformed_valdata = transformation.apply(ConcatDataset(val_data), is_train=False)

        callbacks = [
            EvaluateCallback(
                context_length=model_params["context_length"],
                prediction_length=model_params["prediction_length"],
                model=model,
                datasets={"val": transformed_valdata},
                logdir=logdir,
                **evaluation_params,
            )
        ]
    else:
        transformed_data = transformation.apply(training_data, is_train=True)

    log_monitor = "train_loss"
    filename = dataset_name + "-{epoch:03d}-{train_loss:.3f}"

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
        enable_progress_bar=True,
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

    metrics = (
        evaluate_conditional(model_params, model, test_data, transformation, trainer)
        if evaluation_params.get("do_final_eval", True)
        else "Final eval not performed"
    )
    with open(Path(logdir) / "results.yaml", "w") as fp:
        yaml.dump(
            {
                "config": config,
                "version": trainer.logger.version,
                "metrics": metrics,
            },
            fp,
        )
    return metrics


if __name__ == "__main__":
    # Setup Logger
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to yaml config")
    parser.add_argument("--logdir", type=str, default="./logs", help="Path to results dir")
    args, _ = parser.parse_known_args()

    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp)

    # Update config from command line
    parser = add_config_to_argparser(config=config, parser=parser)
    args = parser.parse_args()
    config_updates = vars(args)
    for k in config.keys() & config_updates.keys():
        orig_val = config[k]
        updated_val = config_updates[k]
        if updated_val != orig_val:
            logging.info(f"Updated key '{k}': {orig_val} -> {updated_val}")
    config.update(config_updates)
    aim_logger = AimLogger()
    aim_logger.log_hyperparams(config)
    aim_logger.log_hyperparams(config)
    config["logdir"] = args.logdir + "/" + aim_logger.version
    main(**config, loggers=[aim_logger])
