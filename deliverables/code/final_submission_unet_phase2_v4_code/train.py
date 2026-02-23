"""Training script for 2.5D U-Net MRI Super Resolution.

Features: EMA, AMP, cosine LR with warmup, gradient clipping, TensorBoard.
Single-phase training (no GAN). Optimizes Charbonnier L1 + MS-SSIM (Phase 2).

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
from losses import CharbonnierLoss, MSSSIMLoss
from dataset import SliceMRIDataset


def get_cosine_lr(step, total_steps, warmup_steps, base_lr, min_lr):
    """Cosine annealing with linear warmup."""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def update_ema(ema_model, model, decay):
    """Update EMA model parameters (buffers copied separately via recalibrate_bn)."""
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(decay).add_(p.data, alpha=1 - decay)


@torch.no_grad()
def recalibrate_bn(model, data_loader, device):
    """Recalculate BatchNorm running stats for EMA model by running a forward pass."""
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
        model(inp)
    if not was_training:
        model.eval()


def compute_msssim_metric(pred, target):
    """Compute MS-SSIM matching phase_2/metric.py (GPU-accelerated via pytorch_msssim)."""
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


def validate(model, val_loader, l1_loss_fn, msssim_loss_fn, device, config):
    """Run validation and return metrics."""
    model.eval()
    total_l1 = 0.0
    total_msssim_loss = 0.0
    total_msssim = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if isinstance(batch, (list, tuple)):
                inp, tgt = batch[0].to(device), batch[1].to(device)
            else:
                inp, tgt = batch['input'].to(device), batch['target'].to(device)
            with autocast(enabled=config['training']['use_amp']):
                pred = model(inp)
                l1 = l1_loss_fn(pred, tgt)
                msssim_l = msssim_loss_fn(pred, tgt)

            total_l1 += l1.item()
            total_msssim_loss += msssim_l.item()
            total_msssim += compute_msssim_metric(pred, tgt)
            n_batches += 1

    n = max(n_batches, 1)
    return {
        'l1': total_l1 / n,
        'msssim_loss': total_msssim_loss / n,
        'msssim': total_msssim / n,
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
        attention=config['model'].get('attention', 'none'),
        se_reduction=config['model'].get('se_reduction', 16),
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
    msssim_loss_fn = MSSSIMLoss().to(device)

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
    epochs_without_improvement = 0
    patience = tcfg.get('early_stopping_patience', 0)
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
    log(f"Loss weights: L1={lcfg['l1_weight']}, MS-SSIM={lcfg['msssim_weight']}")

    for epoch in range(start_epoch, tcfg['epochs']):
        model.train()
        epoch_l1 = 0.0
        epoch_msssim = 0.0
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, (list, tuple)):
                inp, tgt = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            else:
                inp = batch['input'].to(device, non_blocking=True)
                tgt = batch['target'].to(device, non_blocking=True)

            # LR schedule
            lr = get_cosine_lr(global_step, total_steps, tcfg['lr_warmup_steps'],
                               tcfg['lr'], tcfg['min_lr'])
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            # Forward
            with autocast(enabled=tcfg['use_amp']):
                pred = model(inp)
                l1 = l1_loss_fn(pred, tgt)
                msssim_l = msssim_loss_fn(pred, tgt)
                loss = lcfg['l1_weight'] * l1 + lcfg['msssim_weight'] * msssim_l

            # Skip NaN batches (AMP can produce NaN in forward pass)
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                global_step += 1
                continue

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
            epoch_msssim += msssim_l.item()
            epoch_loss += loss.item()
            global_step += 1

            if global_step % 50 == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/l1', l1.item(), global_step)
                writer.add_scalar('train/msssim_loss', msssim_l.item(), global_step)
                writer.add_scalar('train/lr', lr, global_step)

        # Epoch stats
        n = len(train_loader)
        elapsed = time.time() - t0
        log(f"Epoch {epoch:03d} | loss={epoch_loss/n:.4f} l1={epoch_l1/n:.4f} "
            f"msssim_loss={epoch_msssim/n:.4f} | lr={lr:.2e} | {elapsed:.0f}s")

        # Validation (every 5 epochs or last epoch)
        if val_loader and (epoch % 5 == 0 or epoch == tcfg['epochs'] - 1):
            val_metrics = validate(model, val_loader, l1_loss_fn, msssim_loss_fn, device, config)
            recalibrate_bn(ema_model, train_loader, device)
            ema_metrics = validate(ema_model, val_loader, l1_loss_fn, msssim_loss_fn, device, config)

            log(f"  Val (model): MS-SSIM={val_metrics['msssim']:.4f} l1={val_metrics['l1']:.4f}")
            log(f"  Val (EMA):   MS-SSIM={ema_metrics['msssim']:.4f} l1={ema_metrics['l1']:.4f}")

            current_score = max(val_metrics['msssim'], ema_metrics['msssim'])

            writer.add_scalar('val/msssim', val_metrics['msssim'], epoch)
            writer.add_scalar('val/l1', val_metrics['l1'], epoch)
            writer.add_scalar('val_ema/msssim', ema_metrics['msssim'], epoch)
            writer.add_scalar('val_ema/l1', ema_metrics['l1'], epoch)

            if current_score > best_score:
                best_score = current_score
                epochs_without_improvement = 0
                best_is_ema = ema_metrics['msssim'] >= val_metrics['msssim']
                best_state = ema_model.state_dict() if best_is_ema else model.state_dict()
                torch.save({
                    'model': best_state,
                    'epoch': epoch,
                    'score': best_score,
                    'is_ema': best_is_ema,
                    'config': config,
                }, os.path.join(ckpt_dir, 'best_model.pth'))
                log(f"  ** New best MS-SSIM: {best_score:.4f} ({'EMA' if best_is_ema else 'model'}) **")
            else:
                epochs_without_improvement += 5  # validated every 5 epochs

            if patience > 0 and epochs_without_improvement >= patience:
                log(f"  Early stopping: no improvement for {epochs_without_improvement} epochs")
                break

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

    log(f"\nTraining complete. Best MS-SSIM: {best_score:.4f}")
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
