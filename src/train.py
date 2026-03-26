"""
train.py
Training loop untuk PatchTST dengan early stopping, LR scheduler, dan model checkpoint.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Pastikan src/ ada di path
sys.path.insert(0, os.path.dirname(__file__))
from model import PatchTST

# ---------------------------------------------------------------------------
# Konfigurasi default
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = dict(
    # Data
    data_dir   = os.path.join(os.path.dirname(__file__), "..", "data", "processed"),
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models"),
    # Model
    seq_len    = 60,
    pred_len   = 14,
    patch_len  = 16,
    stride     = 8,
    d_model    = 128,
    n_heads    = 8,
    n_layers   = 3,
    d_ff       = 256,
    dropout    = 0.1,
    n_channels = 1,
    # Training
    epochs     = 100,
    batch_size = 64,
    lr         = 1e-3,
    weight_decay = 1e-4,
    patience   = 10,   # early stopping
    # Misc
    device     = "cuda" if torch.cuda.is_available() else "cpu",
    debug      = False,
)


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------
def load_dataset(data_dir: str, split: str) -> TensorDataset:
    X = np.load(os.path.join(data_dir, f"X_{split}.npy"))
    y = np.load(os.path.join(data_dir, f"y_{split}.npy"))
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(cfg: dict):
    os.makedirs(cfg["models_dir"], exist_ok=True)
    device = torch.device(cfg["device"])
    print(f"Device: {device}")

    # --- Data ---
    train_ds = load_dataset(cfg["data_dir"], "train")
    val_ds   = load_dataset(cfg["data_dir"], "val")
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=0, pin_memory=False)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # --- Model ---
    model = PatchTST(
        seq_len    = cfg["seq_len"],
        pred_len   = cfg["pred_len"],
        patch_len  = cfg["patch_len"],
        stride     = cfg["stride"],
        d_model    = cfg["d_model"],
        n_heads    = cfg["n_heads"],
        n_layers   = cfg["n_layers"],
        d_ff       = cfg["d_ff"],
        dropout    = cfg["dropout"],
        n_channels = cfg["n_channels"],
    ).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PatchTST params: {params:,}")

    # --- Optimizer & Scheduler ---
    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=1e-5)
    criterion = nn.MSELoss()

    # --- Training Loop ---
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = os.path.join(cfg["models_dir"], "best_model.pt")
    history = {"train_loss": [], "val_loss": []}

    max_epochs = 5 if cfg.get("debug") else cfg["epochs"]

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch [{epoch:3d}/{max_epochs}] "
              f"train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  lr={lr_now:.2e}")

        # --- Early Stopping & Checkpoint ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "val_loss":   val_loss,
                "config":     cfg,
            }, best_model_path)
            print(f"  ✅ Model terbaik disimpan (val_loss={val_loss:.6f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg["patience"] and not cfg.get("debug"):
                print(f"\n⏹ Early stopping di epoch {epoch} (patience={cfg['patience']})")
                break

    # Simpan history
    import json
    with open(os.path.join(cfg["models_dir"], "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n🎉 Training selesai! Best val_loss={best_val_loss:.6f}")
    print(f"   Model disimpan → {os.path.abspath(best_model_path)}")
    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train PatchTST untuk prediksi stok")
    p.add_argument("--epochs",      type=int,   default=DEFAULT_CONFIG["epochs"])
    p.add_argument("--batch_size",  type=int,   default=DEFAULT_CONFIG["batch_size"])
    p.add_argument("--lr",          type=float, default=DEFAULT_CONFIG["lr"])
    p.add_argument("--d_model",     type=int,   default=DEFAULT_CONFIG["d_model"])
    p.add_argument("--n_heads",     type=int,   default=DEFAULT_CONFIG["n_heads"])
    p.add_argument("--n_layers",    type=int,   default=DEFAULT_CONFIG["n_layers"])
    p.add_argument("--patience",    type=int,   default=DEFAULT_CONFIG["patience"])
    p.add_argument("--debug",       action="store_true", help="Jalankan 5 epoch saja")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = {**DEFAULT_CONFIG, **vars(args)}
    print("=" * 60)
    print("  Prediksi Stok Bahan Jadi Obat Kain — PatchTST Training")
    print("=" * 60)
    train(cfg)
