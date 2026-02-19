"""DDPM training for conditional MRI super-resolution (SR3-style).

Denoising diffusion probabilistic model conditioned on low-field MRI slices.
Input to denoiser: concat([noisy_HF (1ch), LF_context (5ch)]) = (B, 6, H, W).
Loss: MSE on predicted noise epsilon (standard DDPM objective).

Usage:
    python diffusion_train.py --config diffusion_config.yaml --experiment diffusion_v1
    python diffusion_train.py --config diffusion_config.yaml --experiment diffusion_v1 --resume experiments/diffusion_v1/checkpoints/checkpoint_latest.pth
"""

import argparse
import copy
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from diffusion_model import DiffusionUNet
from dataset import SliceMRIDataset


# ---------------------------------------------------------------------------
# DDPM schedule utilities
# ---------------------------------------------------------------------------

def make_schedule(T, beta_start, beta_end, device):
    """Build linear DDPM noise schedule tensors (all on device)."""
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_acp': torch.sqrt(alphas_cumprod),
        'sqrt_1macp': torch.sqrt(1.0 - alphas_cumprod),
    }


def q_sample(x0, t, noise, schedule):
    """Forward diffusion: add noise to x0 at timestep t.

    x_t = sqrt(acp_t) * x0 + sqrt(1 - acp_t) * epsilon

    Args:
        x0:    (B, 1, H, W) clean HF slice, float in [0, 1]
        t:     (B,) integer timesteps
        noise: (B, 1, H, W) Gaussian noise
        schedule: dict from make_schedule()

    Returns:
        xt: (B, 1, H, W) noisy image at timestep t
    """
    s_acp  = schedule['sqrt_acp'][t][:, None, None, None]
    s_1macp = schedule['sqrt_1macp'][t][:, None, None, None]
    return s_acp * x0 + s_1macp * noise


# ---------------------------------------------------------------------------
# DDIM sampler (used for fast validation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ddim_sample(model, lf_cond, schedule, n_steps, device, use_amp=True):
    """Deterministic DDIM sampling (eta=0).

    Args:
        model:    DiffusionUNet in eval mode
        lf_cond:  (B, 5, H, W) upsampled LF conditioning slices
        schedule: dict from make_schedule()
        n_steps:  number of DDIM denoising steps
        device:   torch device
        use_amp:  use autocast

    Returns:
        (B, 1, H, W) predicted HF slice, clamped to [0, 1]
    """
    T = schedule['betas'].shape[0]
    B, _, H, W = lf_cond.shape

    # Evenly-spaced timesteps from T-1 down to 0
    timesteps = torch.linspace(T - 1, 0, n_steps, dtype=torch.long, device=device)

    x = torch.randn(B, 1, H, W, device=device)

    sqrt_acp   = schedule['sqrt_acp']
    sqrt_1macp = schedule['sqrt_1macp']

    for i, t_cur in enumerate(timesteps):
        t_batch = t_cur.expand(B)
        model_in = torch.cat([x, lf_cond], dim=1)  # (B, 6, H, W)

        with autocast(enabled=use_amp):
            eps_pred = model(model_in, t_batch)     # (B, 1, H, W)

        # Estimate clean x0 from predicted noise
        x0_hat = (x - sqrt_1macp[t_cur] * eps_pred) / sqrt_acp[t_cur]
        x0_hat = x0_hat.clamp(0.0, 1.0)

        if i < n_steps - 1:
            t_next = timesteps[i + 1]
            x = sqrt_acp[t_next] * x0_hat + sqrt_1macp[t_next] * eps_pred
        else:
            x = x0_hat

    return x.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Training utilities (mirrors train.py)
# ---------------------------------------------------------------------------

def get_cosine_lr(step, total_steps, warmup_steps, base_lr, min_lr):
    """Cosine annealing with linear warmup."""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def update_ema(ema_model, model, decay):
    """Exponential moving average update."""
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)


@torch.no_grad()
def recalibrate_bn(model, data_loader, device):
    """Recalculate BatchNorm running stats for EMA model."""
    was_training = model.training
    model.train()
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.running_mean.zero_()
            module.running_var.fill_(1)
            module.num_batches_tracked.zero_()
    for batch in data_loader:
        if isinstance(batch, (list, tuple)):
            inp = batch[0].to(device)
        else:
            inp = batch['input'].to(device)
        t_dummy = torch.zeros(inp.shape[0], dtype=torch.long, device=device)
        noise = torch.randn(inp.shape[0], 1, inp.shape[2], inp.shape[3], device=device)
        model_in = torch.cat([noise, inp], dim=1)
        model(model_in, t_dummy)
    if not was_training:
        model.eval()


def compute_msssim_metric(pred, target):
    """Compute MS-SSIM matching phase_2/metric.py."""
    from pytorch_msssim import ms_ssim
    with torch.no_grad():
        val = ms_ssim(
            pred, target,
            data_range=1.0,
            size_average=True,
            win_size=11,
            win_sigma=1.5,
            weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333],
            K=(0.01, 0.03),
        )
        val = torch.nan_to_num(val, nan=0.0)
        return val.clamp(0.0, 1.0).item()


def validate_diffusion(model, val_loader, schedule, ddim_steps_val, device, use_amp):
    """Run DDIM inference on val set and compute MS-SSIM."""
    model.eval()
    total_msssim = 0.0
    n_batches = 0

    for batch in val_loader:
        if isinstance(batch, (list, tuple)):
            inp, tgt = batch[0].to(device), batch[1].to(device)
        else:
            inp = batch['input'].to(device)
            tgt = batch['target'].to(device)

        pred = ddim_sample(model, inp, schedule, ddim_steps_val, device, use_amp)
        total_msssim += compute_msssim_metric(pred, tgt)
        n_batches += 1

    return {'msssim': total_msssim / max(n_batches, 1)}


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config, experiment_name, resume_path=None, device=None):
    """Main DDPM training function."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Directories
    exp_dir  = os.path.join('experiments', experiment_name)
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    log_dir  = os.path.join(exp_dir, 'logs')
    tb_dir   = os.path.join(exp_dir, 'tensorboard')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)
    os.makedirs(tb_dir,   exist_ok=True)

    # Data (same pipeline as U-Net training)
    train_dataset = SliceMRIDataset(config, is_train=True)
    has_val = bool(config['data'].get('val_samples', []))
    val_dataset = SliceMRIDataset(config, is_train=False) if has_val else None

    tcfg = config['training']
    train_loader = DataLoader(
        train_dataset, batch_size=tcfg['batch_size'], shuffle=True,
        num_workers=tcfg['num_workers'], pin_memory=True, drop_last=True,
    )
    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset, batch_size=tcfg['batch_size'], shuffle=False,
            num_workers=tcfg['num_workers'], pin_memory=True,
        )

    print(f"Train: {len(train_dataset)} slices, {len(train_loader)} batches")
    if val_loader:
        print(f"Val:   {len(val_dataset)} slices, {len(val_loader)} batches")
    else:
        print("Val:   disabled (all volumes used for training)")

    # DDPM schedule
    dcfg = config['diffusion']
    schedule = make_schedule(dcfg['T'], dcfg['beta_start'], dcfg['beta_end'], device)
    print(f"DDPM: T={dcfg['T']}, beta=[{dcfg['beta_start']}, {dcfg['beta_end']}]")

    # Model: in_channels = LF context (5) + 1 noisy HF channel
    n_in = config['data']['slice_context'] + 1
    model = DiffusionUNet(
        in_channels=n_in,
        out_channels=1,
        base_features=dcfg['base_features'],
        time_emb_dim=dcfg['time_emb_dim'],
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DiffusionUNet parameters: {param_count:,}")

    # EMA model
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg['lr'],
        betas=tuple(tcfg['adam_betas']),
        weight_decay=tcfg['weight_decay'],
    )

    scaler = GradScaler(enabled=tcfg['use_amp'])
    writer  = SummaryWriter(tb_dir)

    # Training state
    start_epoch = 0
    global_step = 0
    best_score  = 0.0
    epochs_without_improvement = 0
    patience    = tcfg.get('early_stopping_patience', 0)
    total_steps = tcfg['epochs'] * len(train_loader)

    # Resume
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        ema_model.load_state_dict(ckpt['ema_model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        global_step = ckpt['global_step']
        best_score  = ckpt.get('best_score', 0.0)
        print(f"Resumed epoch={start_epoch}, step={global_step}, best={best_score:.4f}")

    # Log file
    log_path = os.path.join(log_dir, 'training.log')
    log_file = open(log_path, 'a')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log(f"Training diffusion {experiment_name}: epochs={tcfg['epochs']}, lr={tcfg['lr']}, bs={tcfg['batch_size']}")
    log(f"DDPM T={dcfg['T']}, val_ddim={dcfg['ddim_steps_val']}, infer_ddim={dcfg['ddim_steps']}")

    for epoch in range(start_epoch, tcfg['epochs']):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, (list, tuple)):
                lf_cond = batch[0].to(device, non_blocking=True)
                x0      = batch[1].to(device, non_blocking=True)
            else:
                lf_cond = batch['input'].to(device, non_blocking=True)
                x0      = batch['target'].to(device, non_blocking=True)
            # lf_cond: (B, 5, H, W), x0: (B, 1, H, W)

            # LR schedule
            lr = get_cosine_lr(global_step, total_steps, tcfg['lr_warmup_steps'],
                               tcfg['lr'], tcfg['min_lr'])
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # Sample random timestep and Gaussian noise
            B = x0.shape[0]
            t = torch.randint(0, dcfg['T'], (B,), device=device)
            noise = torch.randn_like(x0)

            # Forward diffusion: corrupt x0 to x_t
            xt = q_sample(x0, t, noise, schedule)

            # Denoiser forward: predict noise conditioned on LF slices
            model_in = torch.cat([xt, lf_cond], dim=1)  # (B, 6, H, W)

            with autocast(enabled=tcfg['use_amp']):
                pred_noise = model(model_in, t)
                loss = F.mse_loss(pred_noise, noise)

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                global_step += 1
                continue

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), tcfg['gradient_clip_norm'])
            scaler.step(optimizer)
            scaler.update()

            update_ema(ema_model, model, tcfg['ema_decay'])

            epoch_loss  += loss.item()
            global_step += 1

            if global_step % 50 == 0:
                writer.add_scalar('train/mse_loss', loss.item(), global_step)
                writer.add_scalar('train/lr',       lr,          global_step)

        n       = len(train_loader)
        elapsed = time.time() - t0
        log(f"Epoch {epoch:03d} | mse={epoch_loss/n:.4f} | lr={lr:.2e} | {elapsed:.0f}s")

        # Validation every 5 epochs (DDIM-20 for speed)
        if val_loader and (epoch % 5 == 0 or epoch == tcfg['epochs'] - 1):
            val_m = validate_diffusion(model, val_loader, schedule,
                                       dcfg['ddim_steps_val'], device, tcfg['use_amp'])
            recalibrate_bn(ema_model, train_loader, device)
            ema_m = validate_diffusion(ema_model, val_loader, schedule,
                                       dcfg['ddim_steps_val'], device, tcfg['use_amp'])

            log(f"  Val (model): MS-SSIM={val_m['msssim']:.4f}")
            log(f"  Val (EMA):   MS-SSIM={ema_m['msssim']:.4f}")

            current_score = max(val_m['msssim'], ema_m['msssim'])

            writer.add_scalar('val/msssim',     val_m['msssim'], epoch)
            writer.add_scalar('val_ema/msssim', ema_m['msssim'], epoch)

            if current_score > best_score:
                best_score = current_score
                epochs_without_improvement = 0
                best_is_ema = ema_m['msssim'] >= val_m['msssim']
                best_state  = ema_model.state_dict() if best_is_ema else model.state_dict()
                torch.save({
                    'model':      best_state,
                    'epoch':      epoch,
                    'score':      best_score,
                    'is_ema':     best_is_ema,
                    'config':     config,
                    'model_type': 'diffusion',
                }, os.path.join(ckpt_dir, 'best_model.pth'))
                log(f"  ** New best MS-SSIM: {best_score:.4f} ({'EMA' if best_is_ema else 'model'}) **")
            else:
                epochs_without_improvement += 5

            if patience > 0 and epochs_without_improvement >= patience:
                log(f"  Early stopping: no improvement for {epochs_without_improvement} epochs")
                break

        # Checkpoint every 10 epochs
        if epoch % 10 == 0 or epoch == tcfg['epochs'] - 1:
            torch.save({
                'model':        model.state_dict(),
                'ema_model':    ema_model.state_dict(),
                'optimizer':    optimizer.state_dict(),
                'scaler':       scaler.state_dict(),
                'epoch':        epoch,
                'global_step':  global_step,
                'best_score':   best_score,
                'config':       config,
                'model_type':   'diffusion',
            }, os.path.join(ckpt_dir, 'checkpoint_latest.pth'))

        if not val_loader and epoch == tcfg['epochs'] - 1:
            torch.save({
                'model':      model.state_dict(),
                'epoch':      epoch,
                'score':      0.0,
                'is_ema':     False,
                'config':     config,
                'model_type': 'diffusion',
            }, os.path.join(ckpt_dir, 'best_model.pth'))
            log("  Saved final model as best_model.pth (no validation)")

    log(f"\nTraining complete. Best MS-SSIM: {best_score:.4f}")
    writer.close()
    log_file.close()


def main():
    parser = argparse.ArgumentParser(description='DDPM training for MRI super-resolution')
    parser.add_argument('--config',     type=str, default='diffusion_config.yaml')
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--resume',     type=str, default=None)
    parser.add_argument('--device',     type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.batch_size:
        config['training']['batch_size'] = args.batch_size

    train(config, args.experiment, args.resume, args.device)


if __name__ == '__main__':
    main()
