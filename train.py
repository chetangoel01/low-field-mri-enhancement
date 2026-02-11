"""Training script for 2.5D U-Net MRI Super Resolution.

Features: EMA, AMP, cosine LR with warmup, gradient clipping, TensorBoard.
Single-phase training (no GAN). Optimizes Charbonnier L1 + Competition SSIM.

Usage:
    python train.py --config config.yaml --experiment unet_v1
    python train.py --config config.yaml --experiment unet_v1 --resume experiments/unet_v1/checkpoints/checkpoint_latest.pth
"""

import argparse
import copy
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from model import UNet2D
from losses import CharbonnierLoss, CompetitionSSIMLoss
from dataset import SliceMRIDataset


def get_cosine_lr(step, total_steps, warmup_steps, base_lr, min_lr):
    """Cosine annealing with linear warmup."""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def update_ema(ema_model, model, decay):
    """Update EMA model parameters."""
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)


def compute_competition_metrics(pred, target):
    """Compute SSIM and PSNR matching metric.py exactly (per-slice, on [0,1] data)."""
    # pred, target: (B, 1, H, W) tensors in [0, 1]
    b = pred.shape[0]
    ssim_vals = []
    psnr_vals = []

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    for i in range(b):
        p = pred[i, 0].detach().cpu().numpy()
        t = target[i, 0].detach().cpu().numpy()

        # Per-slice normalize (matching metric.py)
        def normalize(x):
            x_min, x_max = x.min(), x.max()
            if x_max - x_min > 0:
                return (x - x_min) / (x_max - x_min)
            return np.zeros_like(x)

        p_n = normalize(p)
        t_n = normalize(t)

        # SSIM
        mu_p, mu_t = p_n.mean(), t_n.mean()
        sig_p = ((p_n - mu_p) ** 2).mean()
        sig_t = ((t_n - mu_t) ** 2).mean()
        sig_pt = ((p_n - mu_p) * (t_n - mu_t)).mean()
        ssim = float(((2 * mu_p * mu_t + C1) * (2 * sig_pt + C2)) /
                      ((mu_p**2 + mu_t**2 + C1) * (sig_p + sig_t + C2)))
        ssim_vals.append(ssim)

        # PSNR
        mse = ((p_n - t_n) ** 2).mean()
        if mse == 0:
            psnr_vals.append(50.0)
        else:
            psnr = float(10 * np.log10(1.0 / mse))
            psnr_vals.append(min(max(psnr, 0), 50))

    mean_ssim = np.mean(ssim_vals)
    mean_psnr = np.mean(psnr_vals)
    score = 0.5 * mean_ssim + 0.5 * (mean_psnr / 50)
    return mean_ssim, mean_psnr, score


def validate(model, val_loader, l1_loss_fn, ssim_loss_fn, device, config):
    """Run validation and return metrics."""
    model.eval()
    total_l1 = 0.0
    total_ssim_loss = 0.0
    total_ssim = 0.0
    total_psnr = 0.0
    total_score = 0.0
    n_batches = 0

    with torch.no_grad():
        for inp, tgt in val_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            with autocast(enabled=config['training']['use_amp']):
                pred = model(inp)
                l1 = l1_loss_fn(pred, tgt)
                ssim_l = ssim_loss_fn(pred, tgt)

            total_l1 += l1.item()
            total_ssim_loss += ssim_l.item()

            # Competition metrics
            ssim, psnr, score = compute_competition_metrics(pred, tgt)
            total_ssim += ssim
            total_psnr += psnr
            total_score += score
            n_batches += 1

    n = max(n_batches, 1)
    return {
        'l1': total_l1 / n,
        'ssim_loss': total_ssim_loss / n,
        'ssim': total_ssim / n,
        'psnr': total_psnr / n,
        'score': total_score / n,
    }


def train(config, experiment_name, resume_path=None, device=None):
    """Main training function."""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Directories
    exp_dir = os.path.join('experiments', experiment_name)
    ckpt_dir = os.path.join(exp_dir, 'checkpoints')
    log_dir = os.path.join(exp_dir, 'logs')
    tb_dir = os.path.join(exp_dir, 'tensorboard')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    # Data
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

    # Model
    model = UNet2D(
        in_channels=config['data']['slice_context'],
        out_channels=1,
        base_features=config['model']['base_features'],
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    # EMA model
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad_(False)

    # Losses
    lcfg = config['loss']
    l1_loss_fn = CharbonnierLoss().to(device)
    ssim_loss_fn = CompetitionSSIMLoss().to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg['lr'],
        betas=tuple(tcfg['adam_betas']),
        weight_decay=tcfg['weight_decay'],
    )

    # AMP
    scaler = GradScaler(enabled=tcfg['use_amp'])

    # TensorBoard
    writer = SummaryWriter(tb_dir)

    # State
    start_epoch = 0
    global_step = 0
    best_score = 0.0
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
        best_score = ckpt.get('best_score', 0.0)
        print(f"Resumed at epoch {start_epoch}, step {global_step}, best_score={best_score:.4f}")

    # Training log file
    log_path = os.path.join(log_dir, 'training.log')
    log_file = open(log_path, 'a')

    def log(msg):
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()

    log(f"Training {experiment_name}: {tcfg['epochs']} epochs, lr={tcfg['lr']}, bs={tcfg['batch_size']}")
    log(f"Loss weights: L1={lcfg['l1_weight']}, SSIM={lcfg['ssim_weight']}")

    for epoch in range(start_epoch, tcfg['epochs']):
        model.train()
        epoch_l1 = 0.0
        epoch_ssim = 0.0
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (inp, tgt) in enumerate(train_loader):
            inp, tgt = inp.to(device, non_blocking=True), tgt.to(device, non_blocking=True)

            # LR schedule
            lr = get_cosine_lr(global_step, total_steps, tcfg['lr_warmup_steps'],
                               tcfg['lr'], tcfg['min_lr'])
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # Forward
            with autocast(enabled=tcfg['use_amp']):
                pred = model(inp)
                l1 = l1_loss_fn(pred, tgt)
                ssim_l = ssim_loss_fn(pred, tgt)
                loss = lcfg['l1_weight'] * l1 + lcfg['ssim_weight'] * ssim_l

            # Backward
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), tcfg['gradient_clip_norm'])
            scaler.step(optimizer)
            scaler.update()

            # EMA update
            update_ema(ema_model, model, tcfg['ema_decay'])

            # Logging
            epoch_l1 += l1.item()
            epoch_ssim += ssim_l.item()
            epoch_loss += loss.item()
            global_step += 1

            if global_step % 50 == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/l1', l1.item(), global_step)
                writer.add_scalar('train/ssim_loss', ssim_l.item(), global_step)
                writer.add_scalar('train/lr', lr, global_step)

        # Epoch stats
        n = len(train_loader)
        elapsed = time.time() - t0
        log(f"Epoch {epoch:03d} | loss={epoch_loss/n:.4f} l1={epoch_l1/n:.4f} "
            f"ssim_loss={epoch_ssim/n:.4f} | lr={lr:.2e} | {elapsed:.0f}s")

        # Validation (every 5 epochs or last epoch)
        if val_loader and (epoch % 5 == 0 or epoch == tcfg['epochs'] - 1):
            val_metrics = validate(model, val_loader, l1_loss_fn, ssim_loss_fn, device, config)
            ema_metrics = validate(ema_model, val_loader, l1_loss_fn, ssim_loss_fn, device, config)

            log(f"  Val (model): SSIM={val_metrics['ssim']:.4f} PSNR={val_metrics['psnr']:.2f} "
                f"Score={val_metrics['score']:.4f}")
            log(f"  Val (EMA):   SSIM={ema_metrics['ssim']:.4f} PSNR={ema_metrics['psnr']:.2f} "
                f"Score={ema_metrics['score']:.4f}")

            current_score = max(val_metrics['score'], ema_metrics['score'])

            writer.add_scalar('val/ssim', val_metrics['ssim'], epoch)
            writer.add_scalar('val/psnr', val_metrics['psnr'], epoch)
            writer.add_scalar('val/score', val_metrics['score'], epoch)
            writer.add_scalar('val_ema/ssim', ema_metrics['ssim'], epoch)
            writer.add_scalar('val_ema/psnr', ema_metrics['psnr'], epoch)
            writer.add_scalar('val_ema/score', ema_metrics['score'], epoch)

            if current_score > best_score:
                best_score = current_score
                best_is_ema = ema_metrics['score'] >= val_metrics['score']
                best_state = ema_model.state_dict() if best_is_ema else model.state_dict()
                torch.save({
                    'model': best_state,
                    'epoch': epoch,
                    'score': best_score,
                    'is_ema': best_is_ema,
                    'config': config,
                }, os.path.join(ckpt_dir, 'best_model.pth'))
                log(f"  ** New best score: {best_score:.4f} ({'EMA' if best_is_ema else 'model'}) **")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0 or epoch == tcfg['epochs'] - 1:
            torch.save({
                'model': model.state_dict(),
                'ema_model': ema_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'global_step': global_step,
                'best_score': best_score,
                'config': config,
            }, os.path.join(ckpt_dir, 'checkpoint_latest.pth'))

        # When no validation, save best_model.pth as the final model at end
        if not val_loader and epoch == tcfg['epochs'] - 1:
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'score': 0.0,
                'is_ema': False,
                'config': config,
            }, os.path.join(ckpt_dir, 'best_model.pth'))
            log("  Saved final model as best_model.pth (no validation)")

    log(f"\nTraining complete. Best score: {best_score:.4f}")
    writer.close()
    log_file.close()


def main():
    parser = argparse.ArgumentParser(description='Train 2.5D U-Net for MRI Super Resolution')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.batch_size:
        config['training']['batch_size'] = args.batch_size

    train(config, args.experiment, args.resume, args.device)


if __name__ == '__main__':
    main()
