"""
fcn_ir_drop.py
~~~~~~~~~~~~~~
Fully Convolutional U-Net for IR-drop prediction.

  Input  X : (B, 3, H, W)  — current_map | eff_dist | pdn_density
  Output Y : (B, 1, H, W)  — ir_drop

Reads pre-built tensor files produced by build_tensors.py:
  tensors/fake.pt    → train
  tensors/real.pt    → val   (logged every epoch)
  tensors/hidden.pt  → test  (evaluated once after training)

Usage
─────
  # Step 1 — build tensors from CSVs (one-time)
  python build_tensors.py

  # Step 2 — train + final eval
  python fcn_ir_drop.py

  # Load checkpoint, skip training, run evaluation only
  python fcn_ir_drop.py --eval_only

  # Override hyper-parameters
  python fcn_ir_drop.py --epochs 200 --base_ch 64 --lr 3e-4
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════

TENSOR_DIR = Path("tensors")
CKPT_PATH  = Path("checkpoints/best_model.pt")
CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Hyper-parameters  (can be overridden via CLI)
# ══════════════════════════════════════════════════════════════════════════════

CFG = dict(
    epochs      = 1000,
    batch_size  = 4,    # each sample is already a separate tensor; loader uses 1
    lr          = 1e-3,
    weight_decay= 1e-4,
    base_ch     = 32,   # channel width of first encoder block
    patience    = 950,   # early-stopping patience
    grad_clip   = 3.0,
    num_workers = 1,
)
# ══════════════════════════════════════════════════════════════════════════════
# Dataset  — wraps a pre-built .pt file
# ══════════════════════════════════════════════════════════════════════════════

class TensorListDataset(Dataset):
    """
    Wraps one of the .pt files produced by build_tensors.py.

    Each .pt file contains:
        X      : list[Tensor(3, H_i, W_i)]
        Y      : list[Tensor(1, H_i, W_i)]
        labels : list[str]
        stats  : dict  (present in fake.pt only)

    Samples can have different (H, W) — we never pad or crop them here.
    The DataLoader must therefore use batch_size=1 (see identity_collate).
    """

    def __init__(self, pt_path: Path):
        if not pt_path.exists():
            raise FileNotFoundError(
                f"{pt_path} not found.\n"
                f"Run  python build_tensors.py  first."
            )
        data          = torch.load(pt_path, weights_only=False)
        self.X        = data["X"]       # list[Tensor(3,H,W)]
        self.Y        = data["Y"]       # list[Tensor(1,H,W)]
        self.labels   = data["labels"]
        self.stats    = data.get("stats", None)   # only in fake.pt

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def identity_collate(batch):
    """Return a single (1,3,H,W) / (1,1,H,W) pair — avoids torch's default
    collate which would try to stack tensors of different sizes."""
    assert len(batch) == 1
    x, y = batch[0]
    return x.unsqueeze(0), y.unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════════════
# Model — 4-level U-Net FCN
# ══════════════════════════════════════════════════════════════════════════════

class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, pad: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class EncoderBlock(nn.Module):
    """2× ConvBnRelu then MaxPool; returns (pooled_features, skip)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(ConvBnRelu(in_ch, out_ch),
                                  ConvBnRelu(out_ch, out_ch))
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor):
        skip = self.conv(x)
        return self.pool(skip), skip


class DecoderBlock(nn.Module):
    """Bilinear upsample (handles any H×W) + skip concat + 2× conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(ConvBnRelu(in_ch + skip_ch, out_ch),
                                  ConvBnRelu(out_ch, out_ch))

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Upsample to skip's exact spatial size — cleanly handles odd dims
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                          align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class IRDropFCN(nn.Module):
    """
    Fully Convolutional U-Net for IR-drop prediction.

    Encoder  : 4 stages (doubles channels, halves spatial size each stage)
    Bottleneck: 2× ConvBnRelu at the deepest level
    Decoder  : 4 stages (bilinear upsample + skip + 2× conv)
    Head     : 1×1 Conv → single-channel regression output

    Accepts any (H, W) input — output is the same (H, W).
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 32):
        super().__init__()
        b = base_ch                   # 32 → 64 → 128 → 256 | 512

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

        self.head = nn.Conv2d(b, 1, kernel_size=1)   # no activation — regression

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    with torch.no_grad():
        mse   = F.mse_loss(pred, target).item()
        mae   = F.l1_loss(pred, target).item()
        rmse  = mse ** 0.5
        rng   = (target.max() - target.min()).clamp(min=1e-8).item()
        nrmse = rmse / rng
        ss_res = ((pred - target) ** 2).sum().item()
        ss_tot = ((target - target.mean()) ** 2).sum().item() + 1e-8
        r2    = 1.0 - ss_res / ss_tot
    return dict(mse=mse, rmse=rmse, mae=mae, nrmse=nrmse, r2=r2)


def aggregate_metrics(ms: list[dict]) -> dict:
    return {k: float(np.mean([m[k] for m in ms])) for k in ms[0]}


# ══════════════════════════════════════════════════════════════════════════════
# Train / eval helpers
# ══════════════════════════════════════════════════════════════════════════════

def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> dict:
    model.train(train)
    metrics_list = []

    with torch.set_grad_enabled(train):
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            pred = model(X)
            loss = F.mse_loss(pred, Y)

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
                optimizer.step()

            metrics_list.append(compute_metrics(pred, Y))

    return aggregate_metrics(metrics_list)


def evaluate_dataset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    tag: str,
) -> dict:
    """Per-sample + aggregate metrics, printed to stdout."""
    model.eval()
    all_metrics = []
    print(f"\n{'─'*64}")
    print(f"  Final evaluation — {tag}")
    print(f"{'─'*64}")

    with torch.no_grad():
        for i, (X, Y) in enumerate(loader):
            X, Y  = X.to(device), Y.to(device)
            pred  = model(X)
            m     = compute_metrics(pred, Y)
            all_metrics.append(m)
            lbl   = loader.dataset.labels[i]
            print(f"  {lbl:30s}  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}"
                  f"  NRMSE={m['nrmse']:.4f}  R²={m['r2']:.4f}")

    agg = aggregate_metrics(all_metrics)
    print(f"{'─'*64}")
    print(f"  {'AGGREGATE':30s}  RMSE={agg['rmse']:.4f}  MAE={agg['mae']:.4f}"
          f"  NRMSE={agg['nrmse']:.4f}  R²={agg['r2']:.4f}")
    print(f"{'─'*64}\n")
    return agg


# ══════════════════════════════════════════════════════════════════════════════
# Training entry point
# ══════════════════════════════════════════════════════════════════════════════

def make_loader(pt_path: Path, shuffle: bool) -> DataLoader:
    ds = TensorListDataset(pt_path)
    return DataLoader(
        ds,
        batch_size  = 1,              # variable (H,W) — must be 1
        shuffle     = shuffle,
        collate_fn  = identity_collate,
        num_workers = CFG["num_workers"],
        pin_memory  = torch.cuda.is_available(),
    )


def train(args) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"\nDevice : {device}")
    print(f"Config : {CFG}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader  = make_loader(TENSOR_DIR / "fake.pt",   shuffle=True)
    val_loader    = make_loader(TENSOR_DIR / "real.pt",   shuffle=False)
    test_loader   = make_loader(TENSOR_DIR / "hidden.pt", shuffle=False)

    print(f"Train  : {len(train_loader.dataset)} samples  (fake-circuit-data)")
    print(f"Val    : {len(val_loader.dataset)} samples  (real-circuit-data)")
    print(f"Test   : {len(test_loader.dataset)} samples  (hidden-real-circuit-data)\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = IRDropFCN(in_ch=3, base_ch=CFG["base_ch"]).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model  : IRDropFCN  ({n_params:,} trainable parameters)\n")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=20,          # Restart every 50 epochs
        T_mult=1,        # Keep restart intervals constant (or set to 2 to double interval each time)
        eta_min=1e-6     # The minimum learning rate it will drop to
    )

    # ── Loop ──────────────────────────────────────────────────────────────────
    best_val_rmse = float("inf")
    patience_ctr  = 0

    hdr = (f"{'Epoch':>6}  {'Train RMSE':>10}  {'Train R²':>8}  "
           f"{'Val RMSE':>10}  {'Val R²':>8}  {'LR':>10}  Time")
    print(hdr)
    print("─" * len(hdr))

    for epoch in range(1, CFG["epochs"] + 1):
        t0      = time.time()
        tr_m    = run_epoch(model, train_loader, optimizer, device, train=True)
        val_m   = run_epoch(model, val_loader,   None,      device, train=False)
        lr_now  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        scheduler.step(val_m["rmse"])

        marker = ""
        if val_m["rmse"] < best_val_rmse:
            best_val_rmse = val_m["rmse"]
            patience_ctr  = 0
            torch.save(
                {
                    "epoch":    epoch,
                    "model":    model.state_dict(),
                    "optimizer":optimizer.state_dict(),
                    "val_rmse": best_val_rmse,
                    # stats embedded so eval-only mode is self-contained
                    "stats":    train_loader.dataset.stats,
                },
                CKPT_PATH,
            )
            marker = "  ✓"
        else:
            patience_ctr += 1

        print(f"{epoch:6d}  {tr_m['rmse']:10.5f}  {tr_m['r2']:8.4f}  "
              f"{val_m['rmse']:10.5f}  {val_m['r2']:8.4f}  "
              f"{lr_now:10.2e}  {elapsed:.1f}s{marker}")

        if patience_ctr >= CFG["patience"]:
            print(f"\nEarly stopping — no val improvement for {CFG['patience']} epochs.")
            break

    # ── Final evaluation ───────────────────────────────────────────────────────
    print(f"\nLoading best checkpoint  ({CKPT_PATH}) …")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    evaluate_dataset(model, val_loader,  device, "real-circuit-data  (val)")
    evaluate_dataset(model, test_loader, device, "hidden-real-circuit-data  (test)")


# ══════════════════════════════════════════════════════════════════════════════
# Eval-only entry point
# ══════════════════════════════════════════════════════════════════════════════

def eval_only(args) -> None:
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    print(f"Loaded checkpoint — epoch {ckpt['epoch']}, "
          f"best val RMSE {ckpt['val_rmse']:.5f}")

    model = IRDropFCN(in_ch=3, base_ch=CFG["base_ch"]).to(device)
    model.load_state_dict(ckpt["model"])

    val_loader  = make_loader(TENSOR_DIR / "real.pt",   shuffle=False)
    test_loader = make_loader(TENSOR_DIR / "hidden.pt", shuffle=False)

    evaluate_dataset(model, val_loader,  device, "real-circuit-data  (val)")
    evaluate_dataset(model, test_loader, device, "hidden-real-circuit-data  (test)")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_only",   action="store_true")
    parser.add_argument("--tensor_dir",  type=Path, default=TENSOR_DIR)
    parser.add_argument("--epochs",      type=int,   default=CFG["epochs"])
    parser.add_argument("--batch_size",  type=int,   default=CFG["batch_size"])
    parser.add_argument("--lr",          type=float, default=CFG["lr"])
    parser.add_argument("--base_ch",     type=int,   default=CFG["base_ch"])
    parser.add_argument("--patience",    type=int,   default=CFG["patience"])
    parser.add_argument("--num_workers", type=int,   default=CFG["num_workers"])
    args = parser.parse_args()

    # Apply CLI overrides to CFG
    for k in ("epochs", "lr", "base_ch", "patience", "num_workers"):
        CFG[k] = getattr(args, k)

    TENSOR_DIR = args.tensor_dir

    if args.eval_only:
        eval_only(args)
    else:
        train(args)