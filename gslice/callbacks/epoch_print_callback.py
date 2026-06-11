import torch
from pytorch_lightning import Callback


class EpochPrintCallback(Callback):
    """
    Print selected metrics once per epoch.
    """

    def __init__(self, keys=None, key_filter: str | None = "loss"):
        super().__init__()
        self.keys = set(keys) if keys is not None else None
        self.key_filter = key_filter
        self._last_epoch = None

    def _format_metrics(self, metrics):
        items = []
        for key, value in metrics.items():
            if self.keys is not None and key not in self.keys:
                continue
            if self.keys is None and self.key_filter and self.key_filter not in key:
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    value = value.item()
                else:
                    value = value.mean().item()
            try:
                value_str = f"{float(value):.6g}"
            except (TypeError, ValueError):
                continue
            items.append((key, value_str))
        items.sort(key=lambda item: item[0])
        return items

    def _print_epoch(self, trainer):
        epoch = trainer.current_epoch
        if self._last_epoch == epoch:
            return
        metrics = self._format_metrics(trainer.callback_metrics)
        if not metrics:
            return
        parts = [f"epoch={epoch}"] + [f"{k}={v}" for k, v in metrics]
        print(" ".join(parts), flush=True)
        self._last_epoch = epoch

    def on_validation_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False):
            return
        self._print_epoch(trainer)

    def on_train_epoch_end(self, trainer, pl_module):
        if getattr(trainer, "sanity_checking", False):
            return
        has_val = bool(trainer.val_dataloaders)
        if not has_val:
            self._print_epoch(trainer)
