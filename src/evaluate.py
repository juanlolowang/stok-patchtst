"""
evaluate.py
Evaluasi model PatchTST pada test set: MAE, RMSE, MAPE + plot prediksi vs aktual.
"""

import os
import sys
import json
import numpy as np
import torch
import joblib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))
from model import PatchTST

BASE        = os.path.dirname(__file__)
DATA_DIR    = os.path.join(BASE, "..", "data", "processed")
MODELS_DIR  = os.path.join(BASE, "..", "models")
OUT_DIR     = os.path.join(BASE, "..", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Metrik
# ---------------------------------------------------------------------------
def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mape(y_true, y_pred, eps=1e-8):
    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
def load_model(model_path: str, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]
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
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Evaluasi
# ---------------------------------------------------------------------------
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(MODELS_DIR, "best_model.pt")

    if not os.path.exists(model_path):
        print("❌ Model belum ditemukan. Jalankan train.py terlebih dahulu.")
        return

    model, cfg = load_model(model_path, device)
    print(f"Model dimuat dari: {model_path}")

    # Load test data (normalized)
    X_test = torch.from_numpy(np.load(os.path.join(DATA_DIR, "X_test.npy"))).to(device)
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))   # tetap di CPU

    # Prediksi
    with torch.no_grad():
        y_pred_norm = model(X_test).cpu().numpy()

    # Evaluasi pada skala ternormalisasi
    mae_v  = mae(y_test, y_pred_norm)
    rmse_v = rmse(y_test, y_pred_norm)

    # Untuk MAPE yang bermakna, hitung pada skala asli (average over all scalers)
    scalers = joblib.load(os.path.join(MODELS_DIR, "scalers.pkl"))
    sample_scaler = list(scalers.values())[0]

    y_test_orig = sample_scaler.inverse_transform(y_test[:, 0].reshape(-1, 1)).flatten()
    y_pred_orig = sample_scaler.inverse_transform(y_pred_norm[:, 0].reshape(-1, 1)).flatten()
    y_pred_orig = np.clip(y_pred_orig, 0, None)
    mape_v = mape(y_test_orig, y_pred_orig)

    print(f"\n{'='*45}")
    print(f"  Evaluasi pada Test Set (skala ternormalisasi)")
    print(f"{'='*45}")
    print(f"  MAE  : {mae_v:.6f}")
    print(f"  RMSE : {rmse_v:.6f}")
    print(f"  MAPE : {mape_v:.2f}%")
    print(f"{'='*45}")

    # Simpan metrik
    metrics = {"MAE": float(mae_v), "RMSE": float(rmse_v), "MAPE": float(mape_v)}
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # -----------------------------------------------------------------------
    # Plot 1: Prediksi vs Aktual (sample 200 titik pertama)
    # -----------------------------------------------------------------------
    N = min(200, len(y_test))
    y_true_flat = y_test[:N, 0]
    y_pred_flat = y_pred_norm[:N, 0]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Prediksi Stok Bahan Jadi Obat Kain — PatchTST", fontsize=14, fontweight="bold", color="white")

    ax = axes[0]
    ax.plot(y_true_flat, label="Aktual",  color="#4FC3F7", linewidth=1.5)
    ax.plot(y_pred_flat, label="Prediksi",color="#FF8A65", linewidth=1.5, linestyle="--")
    ax.fill_between(range(N), y_true_flat, y_pred_flat, alpha=0.15, color="#FFD54F")
    ax.set_title("Prediksi vs Aktual (sample 200 hari · langkah pertama horizon)", color="white")
    ax.set_xlabel("Indeks Sampel"); ax.set_ylabel("Stok (ternormalisasi)")
    ax.legend(); ax.grid(alpha=0.2)

    # -----------------------------------------------------------------------
    # Plot 2: Learning Curve (dari history.json)
    # -----------------------------------------------------------------------
    ax2 = axes[1]
    hist_path = os.path.join(MODELS_DIR, "history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            history = json.load(f)
        ep = range(1, len(history["train_loss"]) + 1)
        ax2.plot(ep, history["train_loss"], label="Train Loss", color="#81C784")
        ax2.plot(ep, history["val_loss"],   label="Val Loss",   color="#E57373")
        ax2.set_title("Learning Curve", color="white")
        ax2.set_xlabel("Epoch"); ax2.set_ylabel("MSE Loss")
        ax2.legend(); ax2.grid(alpha=0.2)
    else:
        ax2.text(0.5, 0.5, "history.json tidak ditemukan", ha="center", va="center", color="gray")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "evaluation.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close()
    print(f"\n✅ Plot disimpan → {os.path.abspath(out_path)}")

    return metrics


if __name__ == "__main__":
    evaluate()
