import os
import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUTPUT_DIR = PROJECT_ROOT / "log_plots"


def parse_training_log(log_path: Path):
    """
    Parse a single training.log and return:
        - epochs: list[int]
        - metrics: dict[str, list[float]]  (loss, mse, ssim_loss, l1, msssim_loss, etc.)
        - val_epochs: list[int]
        - val_msssim_model: list[float]
        - val_msssim_ema: list[float]
        - best_msssim: float | None
    Handles both U-Net (train.py) and diffusion (diffusion_train.py) formats.
    """
    epochs = []
    metrics = defaultdict(list)

    val_epochs = []
    val_msssim_model = []
    val_msssim_ema = []

    best_msssim = None
    current_epoch = None

    epoch_re = re.compile(r"^Epoch\s+(\d+)\s+\|\s+(.*)$")
    kv_re = re.compile(r"([a-zA-Z0-9_]+)=([0-9.+-eE]+)")
    val_model_re = re.compile(r"Val \(model\): MS-SSIM=([0-9.]+)")
    val_ema_re = re.compile(r"Val \(EMA\):\s+MS-SSIM=([0-9.]+)")
    best_re = re.compile(r"Best MS-SSIM:\s*([0-9.]+)")

    with log_path.open("r") as f:
        for line in f:
            line = line.strip()

            # Epoch lines
            m = epoch_re.match(line)
            if m:
                current_epoch = int(m.group(1))
                metrics_part = m.group(2)

                # Extract key=value pairs (loss, mse, ssim, l1, msssim_loss, etc.)
                for kv_match in kv_re.finditer(metrics_part):
                    key = kv_match.group(1)
                    val = float(kv_match.group(2))
                    metrics[key].append(val)

                epochs.append(current_epoch)
                continue

            # Validation lines (model / EMA)
            m = val_model_re.search(line)
            if m and current_epoch is not None:
                val_epochs.append(current_epoch)
                val_msssim_model.append(float(m.group(1)))
                continue

            m = val_ema_re.search(line)
            if m and current_epoch is not None:
                # Ensure same length as val_epochs; align with latest epoch
                if len(val_epochs) == 0 or val_epochs[-1] != current_epoch:
                    val_epochs.append(current_epoch)
                    val_msssim_model.append(float("nan"))
                val_msssim_ema.append(float(m.group(1)))
                continue

            # Final best MS-SSIM line
            m = best_re.search(line)
            if m:
                best_msssim = float(m.group(1))

    # Pad EMA list if needed to match model list length
    if val_msssim_ema and len(val_msssim_ema) < len(val_epochs):
        # Just repeat last value; it's only for plotting
        last = val_msssim_ema[-1]
        while len(val_msssim_ema) < len(val_epochs):
            val_msssim_ema.append(last)

    return {
        "epochs": epochs,
        "metrics": dict(metrics),
        "val_epochs": val_epochs,
        "val_msssim_model": val_msssim_model,
        "val_msssim_ema": val_msssim_ema,
        "best_msssim": best_msssim,
    }


def plot_experiment(experiment_name: str, log_info: dict):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    epochs = log_info["epochs"]
    metrics = log_info["metrics"]
    val_epochs = log_info["val_epochs"]
    val_msssim_model = log_info["val_msssim_model"]
    val_msssim_ema = log_info["val_msssim_ema"]
    best_msssim = log_info["best_msssim"]

    if not epochs:
        print(f"[{experiment_name}] No epoch data found; skipping.")
        return

    # --- Plot training metrics ---
    plt.figure(figsize=(8, 5))
    for key, values in metrics.items():
        plt.plot(epochs[: len(values)], values, label=key)

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(f"{experiment_name} - Training Metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    train_fig_path = OUTPUT_DIR / f"{experiment_name}_train_metrics.png"
    plt.savefig(train_fig_path, dpi=200)
    plt.close()
    print(f"[{experiment_name}] Saved training metrics plot -> {train_fig_path}")

    # --- Plot validation MS-SSIM ---
    if val_epochs and (val_msssim_model or val_msssim_ema):
        plt.figure(figsize=(8, 5))
        if val_msssim_model:
            plt.plot(val_epochs[: len(val_msssim_model)], val_msssim_model,
                    marker="o", label="Val MS-SSIM (model)")
        if val_msssim_ema:
            plt.plot(val_epochs[: len(val_msssim_ema)], val_msssim_ema,
                    marker="s", label="Val MS-SSIM (EMA)")

        if best_msssim is not None:
            plt.axhline(best_msssim, color="red", linestyle="--",
                        label=f"Best MS-SSIM = {best_msssim:.4f}")

        plt.xlabel("Epoch")
        plt.ylabel("MS-SSIM")
        plt.title(f"{experiment_name} - Validation MS-SSIM")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        val_fig_path = OUTPUT_DIR / f"{experiment_name}_val_msssim.png"
        plt.savefig(val_fig_path, dpi=200)
        plt.close()
        print(f"[{experiment_name}] Saved validation MS-SSIM plot -> {val_fig_path}")
    else:
        print(f"[{experiment_name}] No validation data found; skipping val plot.")


def main():
    if not EXPERIMENTS_DIR.exists():
        print(f"No experiments directory found at {EXPERIMENTS_DIR}")
        return

    experiments = sorted(
        p for p in EXPERIMENTS_DIR.iterdir()
        if p.is_dir()
    )

    if not experiments:
        print(f"No experiment subdirectories found under {EXPERIMENTS_DIR}")
        return

    print(f"Found {len(experiments)} experiments:")
    for exp_dir in experiments:
        print(f"  - {exp_dir.name}")

    for exp_dir in experiments:
        log_path = exp_dir / "logs" / "training.log"
        if not log_path.exists():
            print(f"[{exp_dir.name}] No training.log found at {log_path}; skipping.")
            continue

        print(f"[{exp_dir.name}] Parsing {log_path} ...")
        log_info = parse_training_log(log_path)
        plot_experiment(exp_dir.name, log_info)

    print(f"\nAll done. Plots saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()