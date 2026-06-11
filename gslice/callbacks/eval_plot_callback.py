import torch
from pytorch_lightning import Callback
import matplotlib.pyplot as plt
from pathlib import Path
from gslice.utils.signal_utils import gp_regression, shift_time_by_portion


class EvaluationPlotCallback(Callback):
    """
    PyTorch Lightning callback for visualizing model predictions during validation.
    
    This callback evaluates the model on validation data with GP-based control inputs
    and creates comprehensive visualizations showing:
    - Mean trajectory comparison
    - Individual predictions vs targets
    - GP control samples
    - Pointwise MSE over time
    
    Args:
        n_batches: Number of validation batches to evaluate
        n_repeats: Number of GP samples per batch for uncertainty estimation
        plot_every_n_epochs: Create plots every N epochs (default: 5)
        save_dir: Directory to save plots (default: "./results/val_plots")
        signal_func: Signal generation function for creating validation data
        seq_length: Sequence length for validation data
    """
    
    def __init__(
        self,
        n_batches: int = 1,
        n_repeats: int = 5,
        plot_every_n_epochs: int = 1,
        max_show : int = 10,
        save_dir: str = "./results/val_plots",
    ):
        super().__init__()
        self.n_batches = n_batches
        self.n_repeats = n_repeats
        self.plot_every_n_epochs = plot_every_n_epochs
        self.max_show = max_show
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Called at the end of validation epoch."""
        # Only plot every N epochs
        if (trainer.current_epoch + 1) % self.plot_every_n_epochs != 0:
            return
        
        # Skip if no validation dataloader
        if trainer.val_dataloaders is None:
            return
        
        device = pl_module.device
        val_dataloader = trainer.val_dataloaders[0] if isinstance(trainer.val_dataloaders, list) else trainer.val_dataloaders
        
        preds_list = []
        gp_samples_list = []
        targets_list = []
        t_batch = None
        obs_len_list = []  # Store observation lengths
        
        # Set model to eval mode
        pl_module.eval()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_dataloader):
                if batch_idx >= self.n_batches:
                    break
                
                batch_preds = []
                batch_gp_samples = []
                batch_targets = []
                
                # Handle PrecomputedSignalDataset dict format (used in train_cde.py validation)
                if isinstance(batch, dict):
                    t = batch["t"].to(device)  # (B, T, 1) - no centering
                    target = batch["target"].to(device)  # (B, T, D)
                    control = batch["control"].to(device)  # (B, S, T, D) with S samples
                    obs_mask = batch.get("obs_mask", None)  # (B, T) bool mask
                    
                    B, S, T, D = control.shape
                    # Use min of available samples and requested repeats
                    n_samples = min(S, self.n_repeats)
                    
                    for s in range(n_samples):
                        # Pass obs_mask to inference_step so the model sees the exact observation split
                        obs_mask_for_call = obs_mask.to(device) if obs_mask is not None else None
                        pred = pl_module.inference_step(
                            t,
                            control[:, s, :, :],
                            obs_mask=obs_mask_for_call,
                        )
                        batch_preds.append(pred.cpu())
                        batch_gp_samples.append(control[:, s, :, :].cpu())
                        batch_targets.append(target.cpu())
                    
                    # Derive obs_len from obs_mask for plotting
                    if obs_mask is not None:
                        # obs_len = number of True values per sample
                        obs_lens = obs_mask.sum(dim=-1).cpu().tolist()
                        if isinstance(obs_lens, int):
                            obs_lens = [obs_lens] * B
                        obs_len_list.extend(obs_lens)
                    else:
                        obs_len_list.extend([T // 2] * B)
                
                else:
                    # Tuple format (t, y) from SignalDataset - on-the-fly GP sampling
                    
                    t, target = batch
                    t, target = t.to(device), target.to(device)
                    
                    # Observe first k% of sequence (no time centering)
                    k = max(1, int(t.shape[1] * 0.5))
                    y_past = target[:, :k, :]
                    t_past = t[:, :k, :]
                    
                    # Store k as obs_len for each sample in batch
                    obs_len_list.extend([k] * t.shape[0])
                    
                    # Generate multiple GP samples for uncertainty estimation
                    gp = gp_regression(t_past, y_past)
                    gp_sampled = gp.sample(t, num_samples=self.n_repeats)
                    # Build obs_mask for tuple-based samples (first k timesteps observed)
                    obs_mask_for_call = torch.zeros(t.shape[0], t.shape[1], dtype=torch.bool, device=device)
                    obs_mask_for_call[:, :k] = True
                    for i in range(self.n_repeats):
                        gp_samples = gp_sampled[i]
                        pred = pl_module.inference_step(
                            t,
                            gp_samples,
                            obs_mask=obs_mask_for_call,
                        )
                        batch_preds.append(pred.cpu())
                        batch_gp_samples.append(gp_samples.cpu())
                        batch_targets.append(target.cpu())
                
                # Store time for plotting
                t_batch = t.cpu() if not isinstance(batch, dict) else t.cpu()
                
                preds_list.append(batch_preds)
                gp_samples_list.append(batch_gp_samples)
                targets_list.append(batch_targets)
        
        # Concatenate results
        preds = torch.cat([torch.stack(bp) for bp in preds_list], dim=1)  # (n_repeats, total_batches, seq_length, output_dim)
        gp_samples = torch.cat([torch.stack(gs) for gs in gp_samples_list], dim=1)
        targets = torch.cat([torch.stack(tg) for tg in targets_list], dim=1)
        
        # Create plot
        fig = self._plot_eval(preds, gp_samples, targets, t_batch, obs_len_list, epoch=trainer.current_epoch)
        
        # Save plot
        save_path = self.save_dir / f"eval_epoch_{trainer.current_epoch:04d}.png"
        fig.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        
        pl_module.train()
    
    def _plot_eval(self, preds, gp_samples, targets, t, obs_len_list, epoch=None):
        """
        Create comprehensive evaluation plots.
        
        Layout:
        - Top row (spanning 3 cols): Mean trajectory comparison
        - Middle row (3 cols): Real samples | Model predictions | GP control samples
        - Bottom row (spanning 3 cols): Pointwise MSE over time
        """
        n_repeats, batch_size, seq_length, output_dim = preds.shape
        
        # Calculate statistics over repeats
        mean_preds = preds.mean(dim=0)  # (batch_size, seq_length, output_dim)
        std_preds = preds.std(dim=0)
        mean_gp = gp_samples.mean(dim=0)  # (batch_size, seq_length, input_dim)
        std_gp = gp_samples.std(dim=0)
        mean_targets = targets.mean(dim=0)  # (batch_size, seq_length, output_dim)
        std_targets = targets.std(dim=0)
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, height_ratios=[2, 3, 2], width_ratios=[1, 1, 1])
        
        # Top: mean trajectory (spans all columns)
        ax_mean = fig.add_subplot(gs[0, :])
        
        # Middle row: three separate plots
        ax_real = fig.add_subplot(gs[1, 0])
        ax_pred = fig.add_subplot(gs[1, 1], sharex=ax_real, sharey=ax_real)
        ax_gp = fig.add_subplot(gs[1, 2], sharex=ax_real, sharey=ax_real)
        
        # Bottom: MSE (spans all columns)
        ax_mse = fig.add_subplot(gs[2, :], sharex=ax_real)
        
        cmap = plt.colormaps['tab20'].colors
        n_show = min(batch_size, self.max_show)
        
        # === Top: Mean trajectory comparison ===
        t_plot = t[0, :, 0].cpu()  # Use first batch's time grid
        
        # Aggregate means and stds over all batches
        global_target_mean = mean_targets.mean(dim=0)[:, 0].cpu()
        global_target_std = mean_targets.std(dim=0)[:, 0].cpu()
        global_pred_mean = mean_preds.mean(dim=0)[:, 0].cpu()
        global_pred_std = mean_preds.std(dim=0)[:, 0].cpu()
        
        global_control_mean = mean_gp.mean(dim=0)[:, 0].cpu()

        ax_mean.plot(t_plot, global_target_mean, label='Target Mean', linewidth=2, color='black')
        ax_mean.plot(t_plot, global_pred_mean, label='Prediction Mean', linewidth=2, linestyle='--', color='red')
        ax_mean.fill_between(t_plot, 
                              global_target_mean - global_target_std, 
                              global_target_mean + global_target_std,
                              alpha=0.2, color='black', label='Target ±1 std')
        ax_mean.fill_between(t_plot, 
                              global_pred_mean - global_pred_std, 
                              global_pred_mean + global_pred_std,
                              alpha=0.2, color='red', label='Pred ±1 std')
        ax_mean.set_title('Mean Trajectory Comparison', fontsize=14, fontweight='bold')
        ax_mean.legend(loc='best', fontsize=9)
        ax_mean.grid(True, alpha=0.3)
        ax_mean.set_xticks([])
        ax_mean.set_ylabel('Value', fontweight='bold')
        if obs_len_list:
            context_len = min(obs_len_list)
        else:
            context_len = len(t_plot) // 2
        context_len = max(1, min(len(t_plot), context_len))
        context_times = t_plot[:context_len]
        ax_mean.plot(
            context_times,
            global_control_mean[:context_len],
            label='Control Mean (context)',
            linewidth=2,
            linestyle='-.',
            color='tab:blue',
        )
        ax_mean.plot(
            context_times,
            global_pred_mean[:context_len],
            label='Forecast (context)',
            linewidth=2,
            linestyle=':',
            color='tab:orange',
        )
        ax_mean.axvspan(
            context_times[0],
            context_times[-1],
            facecolor='gray',
            alpha=0.1,
            label='Context Window' if context_len > 1 else None,
        )
        
        # === Middle: Real samples ===
        for i in range(n_show):
            t_i = t[i, :, 0].cpu()
            ax_real.plot(t_i, mean_targets[i, :, 0].cpu(), color=cmap[i], alpha=0.7, linewidth=2, label=f'Target {i+1}')
            ax_real.fill_between(t_i, 
                                 (mean_targets[i, :, 0] - std_targets[i, :, 0]).cpu(),
                                 (mean_targets[i, :, 0] + std_targets[i, :, 0]).cpu(),
                                 color=cmap[i], alpha=0.15)
        # Mark observation boundary using obs_len
        if n_show > 0 and obs_len_list:
            obs_idx = obs_len_list[0]  # Use first sample's obs_len
        ax_real.set_title('Real Targets', fontsize=12, fontweight='bold')
        ax_real.legend(fontsize=8)
        ax_real.grid(True, alpha=0.3)
        ax_real.set_ylabel('Value', fontweight='bold')
        
        # === Middle: Model predictions ===
        for i in range(n_show):
            t_i = t[i, :, 0].cpu()
            ax_pred.plot(t_i, mean_preds[i, :, 0].cpu(), color=cmap[i], alpha=0.7, linewidth=2, label=f'Pred {i+1}')
            ax_pred.fill_between(t_i,
                                (mean_preds[i, :, 0] - std_preds[i, :, 0]).cpu(),
                                (mean_preds[i, :, 0] + std_preds[i, :, 0]).cpu(),
                                color=cmap[i], alpha=0.15)
        # Mark observation boundary
        if n_show > 0 and obs_len_list:
            obs_idx = obs_len_list[0]
        ax_pred.set_title('Model Predictions', fontsize=12, fontweight='bold')
        ax_pred.legend(fontsize=8)
        ax_pred.grid(True, alpha=0.3)
        
        # === Middle: GP control samples ===
        for i in range(n_show):
            t_i = t[i, :, 0].cpu()
            k = obs_len_list[i] if i < len(obs_len_list) else int(len(t_i) * 0.5)
            # Show observed part in solid, GP samples in dashed
            ax_gp.plot(t_i[:k], mean_targets[i, :k, 0].cpu(), 
                       color=cmap[i], linewidth=2.5, alpha=0.8, label=f'Observed {i+1}')
            ax_gp.plot(t_i, mean_gp[i, :, 0].cpu(), color=cmap[i], linestyle='--', 
                       alpha=0.6, linewidth=1.5, label=f'GP {i+1}')
            ax_gp.fill_between(t_i,
                              (mean_gp[i, :, 0] - std_gp[i, :, 0]).cpu(),
                              (mean_gp[i, :, 0] + std_gp[i, :, 0]).cpu(),
                              color=cmap[i], alpha=0.1)
        # Mark observation boundary
        if n_show > 0 and obs_len_list:
            obs_idx = obs_len_list[0]
        ax_gp.set_title('GP Control Samples', fontsize=12, fontweight='bold')
        ax_gp.legend(fontsize=8)
        ax_gp.grid(True, alpha=0.3)
        
        # === Bottom: Pointwise MSE ===
        squared_error = (preds - targets) ** 2
        mean_mse = squared_error.mean(dim=0)  # Mean over repeats: (batch_size, seq_length, output_dim)
        std_mse = squared_error.std(dim=0)
        
        # Plot MSE for each sample
        for i in range(n_show):
            t_i = t[i, :, 0].cpu()
            ax_mse.plot(t_i, mean_mse[i, :, 0].cpu(), color=cmap[i], alpha=0.7, linewidth=2, label=f'MSE {i+1}')
            ax_mse.fill_between(t_i,
                               (mean_mse[i, :, 0] - std_mse[i, :, 0]).clamp(min=0).cpu(),
                               (mean_mse[i, :, 0] + std_mse[i, :, 0]).cpu(),
                               color=cmap[i], alpha=0.15)
        
        # Mark observation boundary
        if n_show > 0 and obs_len_list:
            obs_idx = obs_len_list[0]
        ax_mse.set_title('Pointwise MSE: Prediction Error Over Time', fontsize=14, fontweight='bold')
        ax_mse.set_xlabel('Time', fontweight='bold')
        ax_mse.set_ylabel('Squared Error', fontweight='bold')
        ax_mse.legend(fontsize=8)
        ax_mse.grid(True, alpha=0.3)
        if epoch is not None:
            fig.suptitle(f'Evaluation Plots at Epoch {epoch}', fontsize=16, fontweight='bold')
        
        fig.tight_layout()
        
        return fig
