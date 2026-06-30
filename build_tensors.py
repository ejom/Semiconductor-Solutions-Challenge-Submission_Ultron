"""
build_tensors.py
~~~~~~~~~~~~~~~~
Reads all CSV data from the ML-for-IR-drop benchmark directories and saves
PyTorch tensor files — one per dataset split.

Because samples can have different spatial dimensions (H, W), each split is
saved as a list of individual tensors rather than a single stacked tensor.

Output files  (in tensors/)
────────────────────────────
  fake.pt    ─┐
  real.pt     ├─  each is a dict:
  hidden.pt  ─┘

  {
    "X"     : [Tensor(3, H_i, W_i), ...],   # current | eff_dist | pdn_density
    "Y"     : [Tensor(1, H_i, W_i), ...],   # ir_drop
    "labels": [str, ...],                   # human-readable sample names
    "stats" : { ... }                       # only in fake.pt  (train stats)
  }

  stats layout:
    { "current": (mean, std), "eff_dist": (mean, std),
      "pdn_density": (mean, std), "ir_drop": (mean, std) }

Run
───
  python build_tensors.py
  python build_tensors.py --data_root /path/to/benchmarks --out_dir tensors
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch


# ══════════════════════════════════════════════════════════════════════════════
# CSV helper
# ══════════════════════════════════════════════════════════════════════════════

def read_csv(path: Path) -> np.ndarray:
    """Read a headerless CSV grid → float32 (H, W) numpy array."""
    return pd.read_csv(path, header=None).to_numpy(dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Raw sample loaders  →  list[dict]  with numpy arrays
# ══════════════════════════════════════════════════════════════════════════════

def load_fake(root: Path) -> list[dict]:
    """
    Flat directory layout:
        current_mapNN_current.csv
        current_mapNN_eff_dist.csv
        current_mapNN_ir_drop.csv
        current_mapNN_pdn_density.csv
    """
    samples = []
    prefixes = sorted({
        m.group(1)
        for f in root.glob("current_map*_current.csv")
        if (m := re.search(r"current_map(\d+)_current\.csv", f.name))
    })

    if not prefixes:
        print(f"  [fake] No samples found in {root}")
        return samples

    for nn in prefixes:
        p = {
            "current":     root / f"current_map{nn}_current.csv",
            "eff_dist":    root / f"current_map{nn}_eff_dist.csv",
            "pdn_density": root / f"current_map{nn}_pdn_density.csv",
            "ir_drop":     root / f"current_map{nn}_ir_drop.csv",
        }
        missing = [k for k, v in p.items() if not v.exists()]
        if missing:
            print(f"  [fake] Skipping map{nn} — missing: {missing}")
            continue
        #convert to numpy
        arrs = {k: read_csv(v) for k, v in p.items()}
        arrs["label"] = f"fake/map{nn}"
        samples.append(arrs)
        print(f"  [fake] map{nn}  shape={arrs['current'].shape}")

    print(f"  [fake] {len(samples)} samples\n")
    return samples


def load_testcases(root: Path, tag: str) -> list[dict]:
    """
    Per-testcase folder layout:
        <folder>/current_map.csv
        <folder>/eff_dist_map.csv
        <folder>/ir_drop_map.csv
        <folder>/pdn_density.csv
    """
    samples = []
    for folder in sorted(root.iterdir()):
        if not folder.is_dir():
            continue
        p = {
            "current":     folder / "current_map.csv",
            "eff_dist":    folder / "eff_dist_map.csv",
            "pdn_density": folder / "pdn_density.csv",
            "ir_drop":     folder / "ir_drop_map.csv",
        }
        missing = [k for k, v in p.items() if not v.exists()]
        if missing:
            print(f"  [{tag}] Skipping {folder.name} — missing: {missing}")
            continue
        arrs = {k: read_csv(v) for k, v in p.items()}
        arrs["label"] = f"{tag}/{folder.name}"
        samples.append(arrs)
        print(f"  [{tag}] {folder.name}  shape={arrs['current'].shape}")

    print(f"  [{tag}] {len(samples)} samples\n")
    return samples


# ══════════════════════════════════════════════════════════════════════════════
# Normalisation stats  (always computed from the fake/train split only)
# ══════════════════════════════════════════════════════════════════════════════

CHANNELS = ["current", "eff_dist", "pdn_density", "ir_drop"]


def compute_stats(samples: list[dict]) -> dict:
    """Return channel-wise (mean, std) computed over all fake-circuit samples."""
    accum = {k: [] for k in CHANNELS}
    for s in samples:
        for k in CHANNELS:
            accum[k].append(s[k].ravel())
    stats = {}
    for k in CHANNELS:
        arr = np.concatenate(accum[k])
        stats[k] = (float(arr.mean()), float(arr.std()) + 1e-8)
        print(f"  [{k}]  mean={stats[k][0]:.5f}  std={stats[k][1]:.5f}")
    return stats


# ══════════════════════════════════════════════════════════════════════════════
# Normalise + convert to per-sample tensors
# ══════════════════════════════════════════════════════════════════════════════

def to_tensor_lists(
    samples: list[dict],
    stats: dict,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str]]:
    """
    Z-score each channel (using training stats) and convert to tensors.

    Returns
    -------
    X_list : list of float32 Tensor(3, H_i, W_i)
    Y_list : list of float32 Tensor(1, H_i, W_i)
    labels : list of str

    Spatial dimensions are kept as-is so the FCN can handle arbitrary (H, W).
    """
    X_list, Y_list, labels = [], [], []

    for s in samples:
        def norm(key: str, arr: np.ndarray) -> np.ndarray:
            mu, sigma = stats[key]
            return (arr - mu) / sigma

        x = np.stack([
            norm("current",     s["current"]),
            norm("eff_dist",    s["eff_dist"]),
            norm("pdn_density", s["pdn_density"]),
        ])                                           # (3, H, W)
        y = norm("ir_drop", s["ir_drop"])[None]     # (1, H, W)

        X_list.append(torch.from_numpy(x))
        Y_list.append(torch.from_numpy(y))
        labels.append(s["label"])

    return X_list, Y_list, labels


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main(data_root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    fake_dir   = data_root / "fake-circuit-data"
    real_dir   = data_root / "real-circuit-data"
    hidden_dir = data_root / "hidden-real-circuit-data"

    # ── 1. Load raw numpy arrays ───────────────────────────────────────────────
    print("── fake-circuit-data ──────────────────────────────────────────────")
    fake_raw = load_fake(fake_dir) if fake_dir.exists() else []
    if not fake_raw:
        raise RuntimeError(f"No fake samples found — check --data_root ({data_root})")

    print("── real-circuit-data ──────────────────────────────────────────────")
    real_raw = load_testcases(real_dir, "real") if real_dir.exists() else []

    print("── hidden-real-circuit-data ───────────────────────────────────────")
    hidden_raw = load_testcases(hidden_dir, "hidden") if hidden_dir.exists() else []

    # ── 2. Compute normalisation stats from FAKE (train) only ─────────────────
    print("── normalisation stats (fake split) ───────────────────────────────")
    stats = compute_stats(fake_raw)
    print()

    # ── 3. Normalise all splits and convert to tensor lists ───────────────────
    splits = {
        "fake":   fake_raw,
        "real":   real_raw,
        "hidden": hidden_raw,
    }

    for split_name, raw in splits.items():
        if not raw:
            print(f"[warn] '{split_name}' split is empty — skipping\n")
            continue

        X_list, Y_list, labels = to_tensor_lists(raw, stats)

        payload: dict = {
            "X":      X_list,    # list[Tensor(3, H_i, W_i)]
            "Y":      Y_list,    # list[Tensor(1, H_i, W_i)]
            "labels": labels,    # list[str]
        }
        # Embed stats only in the training file so the FCN can load them
        if split_name == "fake":
            payload["stats"] = stats

        out_path = out_dir / f"{split_name}.pt"
        torch.save(payload, out_path)

        unique_shapes = sorted({tuple(x.shape) for x in X_list})
        x0, y0 = X_list[0], Y_list[0]
        print(f"Saved  {out_path}  ({len(X_list)} samples)")
        print(f"  Unique shapes (3,H,W) : {unique_shapes}")
        print(f"  Sample[0]  X={tuple(x0.shape)}  "
              f"x∈[{x0.min():.3f}, {x0.max():.3f}]  "
              f"y∈[{y0.min():.3f}, {y0.max():.3f}]")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build per-split tensor files.")
    parser.add_argument("--data_root", type=Path,
                        default=Path("Data/ML-for-IR-drop/benchmarks"),
                        help="Root of the benchmarks directory tree")
    parser.add_argument("--out_dir",   type=Path, default=Path("tensors"),
                        help="Where to write fake.pt / real.pt / hidden.pt")
    args = parser.parse_args()
    main(args.data_root, args.out_dir)