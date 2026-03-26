"""
predict.py
Inferensi / prediksi baru menggunakan model PatchTST yang sudah ditraining.
Bisa dipanggil dari command line maupun di-import oleh Streamlit app.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import joblib

sys.path.insert(0, os.path.dirname(__file__))
from model import PatchTST

BASE       = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE, "..", "data", "processed")
MODELS_DIR = os.path.join(BASE, "..", "models")
RAW_PATH   = os.path.join(BASE, "..", "data", "raw", "stok_obat_kain.csv")


def load_model_and_scalers(device: torch.device):
    """Muat model terbaik dan scalers dari disk."""
    model_path  = os.path.join(MODELS_DIR, "best_model.pt")
    scaler_path = os.path.join(MODELS_DIR, "scalers.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model belum ada. Jalankan train.py terlebih dahulu.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Scalers belum ada. Jalankan preprocess.py terlebih dahulu.")

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

    scalers = joblib.load(scaler_path)
    return model, scalers, cfg


def predict_product(
    kode_produk: str,
    model: PatchTST,
    scalers: dict,
    cfg: dict,
    device: torch.device,
    raw_df: pd.DataFrame | None = None,
) -> dict:
    """
    Prediksi `pred_len` hari ke depan untuk produk tertentu.

    Returns dict:
        tanggal_prediksi : list of str
        prediksi_stok    : list of float (skala asli)
        last_history     : list of float (lookback window, skala asli)
    """
    seq_len  = cfg["seq_len"]
    pred_len = cfg["pred_len"]

    if raw_df is None:
        raw_df = pd.read_csv(RAW_PATH, parse_dates=["tanggal"])

    subset = raw_df[raw_df["kode_produk"] == kode_produk].sort_values("tanggal")
    if len(subset) < seq_len:
        raise ValueError(f"Data {kode_produk} kurang dari {seq_len} baris.")

    scaler = scalers.get(kode_produk)
    if scaler is None:
        raise KeyError(f"Scaler untuk {kode_produk} tidak ditemukan.")

    # Ambil lookback terakhir
    raw_vals = subset["stok_akhir"].values[-seq_len:].astype(float)
    scaled   = scaler.transform(raw_vals.reshape(-1, 1)).flatten()

    # Tensor (1, 1, seq_len)
    x = torch.from_numpy(scaled.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        y_norm = model(x).cpu().numpy().flatten()   # (pred_len,)

    # Inverse transform
    y_pred = scaler.inverse_transform(y_norm.reshape(-1, 1)).flatten()
    y_pred = np.clip(y_pred, 0, None)                # stok tidak bisa negatif

    # Tanggal prediksi
    last_date = subset["tanggal"].iloc[-1]
    tanggal_pred = pd.date_range(last_date + pd.Timedelta(days=1), periods=pred_len, freq="D")

    return {
        "kode_produk":     kode_produk,
        "tanggal_prediksi": [t.strftime("%Y-%m-%d") for t in tanggal_pred],
        "prediksi_stok":   [round(float(v), 2) for v in y_pred],
        "last_history":    [round(float(v), 2) for v in raw_vals],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json

    p = argparse.ArgumentParser()
    p.add_argument("--kode", type=str, default="KSA-001", help="Kode produk")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scalers, cfg = load_model_and_scalers(device)
    raw_df = pd.read_csv(RAW_PATH, parse_dates=["tanggal"])

    result = predict_product(args.kode, model, scalers, cfg, device, raw_df)

    print(f"\n{'='*50}")
    print(f"  Prediksi Stok: {result['kode_produk']}")
    print(f"{'='*50}")
    for tgl, val in zip(result["tanggal_prediksi"], result["prediksi_stok"]):
        bar = "█" * max(1, int(val / max(result["prediksi_stok"]) * 30))
        print(f"  {tgl}  {val:8.1f}  {bar}")
