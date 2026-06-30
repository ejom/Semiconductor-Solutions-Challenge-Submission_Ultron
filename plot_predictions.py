"""
plot_predictions.py
~~~~~~~~~~~~~~~~~~~
Visualises ground-truth vs predicted IR-drop maps for every sample in the
hidden-real-circuit-data split and reports per-sample + aggregate scores.

Regression metrics  : RMSE, MAE, NRMSE, R²
Classification metrics (F1 / Precision / Recall):
    A pixel is labelled "high IR-drop" (positive) when its value exceeds a
    threshold.  Default: top-K% of pixels in the ground-truth map (K=10),
    making the threshold adaptive per sample so it works regardless of the
    absolute IR-drop scale.

Layout per sample (one PNG per testcase + one summary PDF)
──────────────────────────────────────────────────────────
  [ Ground Truth ]  [ Prediction ]  [ Absolute Error ]  [ Scatter plot ]

Usage
─────
  python plot_predictions.py
  python plot_predictions.py --threshold_pct 5   # top-5% = "high IR-drop"
  python plot_predictions.py --out_dir results/plots
  python plot_predictions.py --ckpt checkpoints/best_model.pt
"""

import argparse
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from sklearn.metrics import f1_score, precision_score, recall_score


# ══════════════════════════════════════════════════════════════════════════════
# Paths / defaults
# ══════════════════════════════════════════════════════════════════════════════

TENSOR_DIR = Path("tensors")
CKPT_PATH  = Path("checkpoints/best_model.pt")
OUT_DIR    = Path("results/plots")


# ══════════════════════════════════════════════════════════════════════════════
# Model  (must match fcn_ir_drop.py exactly)
# ══════════════════════════════════════════════════════════════════════════════

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, pad=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(ConvBnRelu(in_ch, out_ch),
                                  ConvBnRelu(out_ch, out_ch))
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        skip = self.conv(x)
        return self.pool(skip), skip

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(ConvBnRelu(in_ch + skip_ch, out_ch),
                                  ConvBnRelu(out_ch, out_ch))

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                          align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))

class IRDropFCN(nn.Module):
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()
        b = base_ch
        self.enc1 = EncoderBlock(in_ch, b)
        self.enc2 = EncoderBlock(b,     b * 2)
        self.enc3 = EncoderBlock(b * 2, b * 4)
        self.enc4 = EncoderBlock(b * 4, b * 8)
        self.bottleneck = nn.Sequential(ConvBnRelu(b * 8,  b * 16),
                                        ConvBnRelu(b * 16, b * 16))
        self.dec4 = DecoderBlock(b * 16, b * 8,  b * 8)
        self.dec3 = DecoderBlock(b * 8,  b * 4,  b * 4)
        self.dec2 = DecoderBlock(b * 4,  b * 2,  b * 2)
        self.dec1 = DecoderBlock(b * 2,  b,      b)
        self.head = nn.Conv2d(b, 1, kernel_size=1)

    def forward(self, x):
        x, s1 = self.enc1(x)
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)
        x = self.bottleneck(x)
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        return self.head(x)


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def regression_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """RMSE, MAE, NRMSE, R² on flat arrays."""
    mse   = float(np.mean((pred - gt) ** 2))
    mae   = float(np.mean(np.abs(pred - gt)))
    rmse  = mse ** 0.5
    rng   = float(gt.max() - gt.min()) or 1e-8
    nrmse = rmse / rng
    ss_res = float(np.sum((pred - gt) ** 2))
    ss_tot = float(np.sum((gt - gt.mean()) ** 2)) or 1e-8
    r2    = 1.0 - ss_res / ss_tot
    return dict(rmse=rmse, mae=mae, nrmse=nrmse, r2=r2)


def classification_metrics(pred: np.ndarray, gt: np.ndarray,
                            threshold_pct: float) -> dict:
    """
    Binarise pred and gt at the top-K% threshold of the *ground-truth* map,
    then compute Precision, Recall, F1.

    Using gt's threshold for both makes the positive class definition
    independent of prediction quality, giving a fair per-sample comparison.
    """
    thresh = float(np.percentile(gt, 100.0 - threshold_pct))
    gt_bin   = (gt   >= thresh).astype(int).ravel()
    pred_bin = (pred >= thresh).astype(int).ravel()

    kw = dict(zero_division=0)
    return dict(
        threshold = thresh,
        precision = float(precision_score(gt_bin, pred_bin, **kw)),
        recall    = float(recall_score(gt_bin,    pred_bin, **kw)),
        f1        = float(f1_score(gt_bin,        pred_bin, **kw)),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Denormalise helper
# ══════════════════════════════════════════════════════════════════════════════

def denorm(arr: np.ndarray, stats: dict, key: str) -> np.ndarray:
    """Undo z-score normalisation so maps are in original IR-drop units."""
    mu, sigma = stats[key]
    return arr * sigma + mu


# ══════════════════════════════════════════════════════════════════════════════
# Per-sample plot
# ══════════════════════════════════════════════════════════════════════════════

CMAP_MAP   = "inferno"   # heatmap colourmap
CMAP_ERR   = "RdBu_r"   # error map (diverging, centred at 0)


def plot_sample(
    gt_raw: np.ndarray,      # (H, W) denormalised ground truth
    pred_raw: np.ndarray,    # (H, W) denormalised prediction
    label: str,
    reg_m: dict,
    cls_m: dict,
    threshold_pct: float,
    out_path: Path,
) -> None:
    err = pred_raw - gt_raw
    vmin = min(gt_raw.min(), pred_raw.min())
    vmax = max(gt_raw.max(), pred_raw.max())
    err_abs_max = max(abs(err.min()), abs(err.max()))

    fig = plt.figure(figsize=(20, 5.5), constrained_layout=True)
    fig.suptitle(label, fontsize=13, fontweight="bold")

    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.08)

    # ── shared normalisation for GT / Pred ───────────────────────────────────
    norm_map = Normalize(vmin=vmin, vmax=vmax)
    norm_err = Normalize(vmin=-err_abs_max, vmax=err_abs_max)

    def imshow(ax, data, norm, cmap, title):
        im = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto",
                       interpolation="nearest")
        ax.set_title(title, fontsize=10, pad=4)
        ax.axis("off")
        return im

    ax0 = fig.add_subplot(gs[0])
    imshow(ax0, gt_raw,   norm_map, CMAP_MAP, "Ground Truth")
    cb0 = fig.colorbar(ScalarMappable(norm=norm_map, cmap=CMAP_MAP),
                       ax=ax0, fraction=0.046, pad=0.04)
    cb0.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    ax1 = fig.add_subplot(gs[1])
    imshow(ax1, pred_raw, norm_map, CMAP_MAP, "Prediction")
    fig.colorbar(ScalarMappable(norm=norm_map, cmap=CMAP_MAP),
                 ax=ax1, fraction=0.046, pad=0.04).ax.yaxis.set_major_formatter(
                     ticker.FormatStrFormatter("%.3f"))

    ax2 = fig.add_subplot(gs[2])
    imshow(ax2, err, norm_err, CMAP_ERR, "Error  (Pred − GT)")
    cb2 = fig.colorbar(ScalarMappable(norm=norm_err, cmap=CMAP_ERR),
                       ax=ax2, fraction=0.046, pad=0.04)
    cb2.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))

    # ── scatter plot  GT vs Pred ──────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[3])
    gt_flat   = gt_raw.ravel()
    pred_flat = pred_raw.ravel()

    # Subsample for speed (max 20 k points)
    n = len(gt_flat)
    if n > 20_000:
        idx = np.random.choice(n, 20_000, replace=False)
        gt_flat   = gt_flat[idx]
        pred_flat = pred_flat[idx]

    ax3.scatter(gt_flat, pred_flat, s=1, alpha=0.3, c="steelblue",
                linewidths=0, rasterized=True)
    lo = min(gt_raw.min(), pred_raw.min())
    hi = max(gt_raw.max(), pred_raw.max())
    ax3.plot([lo, hi], [lo, hi], "r--", lw=1.2, label="ideal")
    ax3.set_xlabel("Ground Truth", fontsize=9)
    ax3.set_ylabel("Prediction",   fontsize=9)
    ax3.set_title("GT vs Pred",    fontsize=10, pad=4)
    ax3.legend(fontsize=8)
    ax3.tick_params(labelsize=8)
    ax3.set_aspect("equal", adjustable="datalim")

    # ── score text box ────────────────────────────────────────────────────────
    score_text = (
        f"RMSE  = {reg_m['rmse']:.4f}\n"
        f"MAE   = {reg_m['mae']:.4f}\n"
        f"NRMSE = {reg_m['nrmse']:.4f}\n"
        f"R²    = {reg_m['r2']:.4f}\n"
        f"─────────────────\n"
        f"Threshold (top {threshold_pct:.0f}%) = {cls_m['threshold']:.4f}\n"
        f"Precision = {cls_m['precision']:.4f}\n"
        f"Recall    = {cls_m['recall']:.4f}\n"
        f"F1 Score  = {cls_m['f1']:.4f}"
    )
    ax3.text(
        1.04, 0.98, score_text,
        transform=ax3.transAxes,
        va="top", ha="left",
        fontsize=8,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                  edgecolor="grey", alpha=0.9),
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved  {out_path.name}")


# ══════════════════════════════════════════════════════════════════════════════
# Summary bar-chart page
# ══════════════════════════════════════════════════════════════════════════════

def plot_summary(all_labels: list, all_reg: list, all_cls: list,
                 threshold_pct: float, out_path: Path) -> None:
    n = len(all_labels)
    short_labels = [lbl.split("/")[-1] for lbl in all_labels]
    x = np.arange(n)
    bar_w = 0.35

    reg_keys = ["rmse", "mae", "nrmse", "r2"]
    cls_keys = ["precision", "recall", "f1"]

    fig, axes = plt.subplots(2, 4, figsize=(22, 9), constrained_layout=True)
    fig.suptitle(
        f"Hidden-real-circuit-data — Summary  "
        f"(F1 threshold = top {threshold_pct:.0f}% pixels)",
        fontsize=13, fontweight="bold",
    )

    colours = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    # ── regression metrics ────────────────────────────────────────────────────
    for col, key in enumerate(reg_keys):
        ax = axes[0, col]
        vals = [m[key] for m in all_reg]
        agg  = float(np.mean(vals))
        bars = ax.bar(x, vals, color=colours[col], edgecolor="white",
                      linewidth=0.5, zorder=3)
        ax.axhline(agg, color="red", lw=1.5, ls="--",
                   label=f"mean={agg:.4f}", zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=8)
        ax.set_title(key.upper(), fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_xlim(-0.6, n - 0.4)

        # value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7)

    # ── classification metrics ────────────────────────────────────────────────
    for col, key in enumerate(cls_keys):
        ax = axes[1, col]
        vals = [m[key] for m in all_cls]
        agg  = float(np.mean(vals))
        bars = ax.bar(x, vals, color=colours[col], edgecolor="white",
                      linewidth=0.5, zorder=3)
        ax.axhline(agg, color="red", lw=1.5, ls="--",
                   label=f"mean={agg:.4f}", zorder=4)
        ax.set_ylim(0, 1.12)
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=8)
        ax.set_title(key.capitalize(), fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.set_xlim(-0.6, n - 0.4)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7)

    # ── hide the 4th slot in the bottom row ───────────────────────────────────
    axes[1, 3].axis("off")

    # ── aggregate text ────────────────────────────────────────────────────────
    agg_reg = {k: float(np.mean([m[k] for m in all_reg])) for k in reg_keys}
    agg_cls = {k: float(np.mean([m[k] for m in all_cls])) for k in cls_keys}
    summary = (
        "── Aggregate (mean across all samples) ──\n\n"
        f"  RMSE      = {agg_reg['rmse']:.5f}\n"
        f"  MAE       = {agg_reg['mae']:.5f}\n"
        f"  NRMSE     = {agg_reg['nrmse']:.5f}\n"
        f"  R²        = {agg_reg['r2']:.5f}\n\n"
        f"  Precision = {agg_cls['precision']:.5f}\n"
        f"  Recall    = {agg_cls['recall']:.5f}\n"
        f"  F1 Score  = {agg_cls['f1']:.5f}"
    )
    axes[1, 3].text(
        0.05, 0.95, summary,
        transform=axes[1, 3].transAxes,
        va="top", ha="left", fontsize=10,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.7", facecolor="lightyellow",
                  edgecolor="grey", alpha=0.95),
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSummary chart → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )

    # ── Load checkpoint ────────────────────────────────────────────────────────
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}\n"
                                "Run  python fcn_ir_drop.py  first.")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    stats    = ckpt["stats"]          # channel-wise (mean, std) from fake split
    base_ch  = args.base_ch
    model = IRDropFCN(in_ch=3, base_ch=base_ch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint  epoch={ckpt['epoch']}  "
          f"val_rmse={ckpt['val_rmse']:.5f}\n")

    # ── Load hidden tensors ────────────────────────────────────────────────────
    hidden_path = Path(args.tensor_dir) / "hidden.pt"
    if not hidden_path.exists():
        raise FileNotFoundError(f"{hidden_path} not found.\n"
                                "Run  python build_tensors.py  first.")
    data   = torch.load(hidden_path, weights_only=False)
    X_list = data["X"]      # list[Tensor(3,H,W)]  — already normalised
    Y_list = data["Y"]      # list[Tensor(1,H,W)]
    labels = data["labels"]

    print(f"Hidden samples : {len(X_list)}\n")

    all_reg, all_cls = [], []

    with torch.no_grad():
        for i, (X, Y, label) in enumerate(zip(X_list, Y_list, labels)):
            # Inference
            pred_norm = model(X.unsqueeze(0).to(device))   # (1,1,H,W)
            pred_norm = pred_norm.squeeze().cpu().numpy()   # (H,W)
            gt_norm   = Y.squeeze().numpy()                 # (H,W)

            # Denormalise back to original IR-drop units for interpretable plots
            pred_raw = denorm(pred_norm, stats, "ir_drop")
            gt_raw   = denorm(gt_norm,   stats, "ir_drop")

            # Scores (computed in original units)
            reg_m = regression_metrics(pred_raw, gt_raw)
            cls_m = classification_metrics(pred_raw, gt_raw, args.threshold_pct)
            all_reg.append(reg_m)
            all_cls.append(cls_m)

            # Per-sample console output
            short = label.split("/")[-1]
            print(f"[{short}]  RMSE={reg_m['rmse']:.4f}  MAE={reg_m['mae']:.4f}"
                  f"  R²={reg_m['r2']:.4f}  "
                  f"F1={cls_m['f1']:.4f}  "
                  f"P={cls_m['precision']:.4f}  R={cls_m['recall']:.4f}")

            # Per-sample figure
            safe_name = label.replace("/", "_")
            plot_sample(
                gt_raw, pred_raw, label,
                reg_m, cls_m, args.threshold_pct,
                out_path=out_dir / f"{safe_name}.png",
            )

    # ── Summary chart ──────────────────────────────────────────────────────────
    plot_summary(labels, all_reg, all_cls, args.threshold_pct,
                 out_path=out_dir / "summary.png")

    # ── Aggregate console print ────────────────────────────────────────────────
    def avg(lst, key): return float(np.mean([m[key] for m in lst]))

    print("\n══ Aggregate metrics (hidden-real-circuit-data) ══")
    print(f"  RMSE      = {avg(all_reg,'rmse'):.5f}")
    print(f"  MAE       = {avg(all_reg,'mae'):.5f}")
    print(f"  NRMSE     = {avg(all_reg,'nrmse'):.5f}")
    print(f"  R²        = {avg(all_reg,'r2'):.5f}")
    print(f"  Precision = {avg(all_cls,'precision'):.5f}")
    print(f"  Recall    = {avg(all_cls,'recall'):.5f}")
    print(f"  F1 Score  = {avg(all_cls,'f1'):.5f}")
    print(f"\nAll plots saved to  {out_dir}/")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot hidden IR-drop predictions and compute scores."
    )
    parser.add_argument("--ckpt",           default=str(CKPT_PATH),
                        help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--tensor_dir",     default=str(TENSOR_DIR),
                        help="Directory containing hidden.pt")
    parser.add_argument("--out_dir",        default=str(OUT_DIR),
                        help="Where to save PNG plots")
    parser.add_argument("--base_ch",        type=int, default=32,
                        help="base_ch used when training (must match checkpoint)")
    parser.add_argument("--threshold_pct",  type=float, default=10.0,
                        help="Top-K%% of GT pixels treated as 'high IR-drop' "
                             "positives for F1/Precision/Recall (default: 10)")
    args = parser.parse_args()
    main(args)
