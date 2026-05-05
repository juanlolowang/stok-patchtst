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
from model import PatchTST, VanillaLSTM

BASE       = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE, "..", "models")
CSV_PATH   = os.path.join(BASE, "..", "data", "processed", "bahan_jadi_bulanan.csv")


def load_model_and_scalers(device: torch.device, model_type: str = "patchtst"):
    """Muat model terbaik dan scalers dari disk."""
    model_name = model_type.lower()
    model_path  = os.path.join(MODELS_DIR, f"best_model_{model_name}.pt")
    
    # Fallback to old name if new name doesn't exist yet
    if not os.path.exists(model_path) and model_name == "patchtst":
        model_path = os.path.join(MODELS_DIR, "best_model.pt")

    scaler_path = os.path.join(MODELS_DIR, "scalers.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_name} belum ada. Jalankan train.py --model {model_name} terlebih dahulu.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError("Scalers belum ada. Jalankan preprocess.py terlebih dahulu.")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg  = ckpt["config"]

    if model_name == "patchtst":
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
    else:
        model = VanillaLSTM(
            seq_len    = cfg["seq_len"],
            pred_len   = cfg["pred_len"],
            hidden_dim = cfg["d_model"],
            n_layers   = cfg["n_layers"],
            dropout    = cfg["dropout"],
            n_channels = cfg["n_channels"],
        ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    scalers = joblib.load(scaler_path)
    return model, scalers, cfg


def predict_product(
    produk: str,
    model: nn.Module,
    scalers: dict,
    cfg: dict,
    device: torch.device,
    raw_df: pd.DataFrame | None = None,
) -> dict:
    """
    Prediksi `pred_len` bulan ke depan untuk produk tertentu.

    Returns dict:
        bulan_prediksi   : list of str (format YYYY-MM)
        prediksi_stok    : list of float (skala asli, kg)
        last_history     : list of float (lookback window, skala asli)
        history_months   : list of str (bulan lookback)
    """
    seq_len  = cfg["seq_len"]
    pred_len = cfg["pred_len"]

    if raw_df is None:
        raw_df = pd.read_csv(CSV_PATH)

    subset = raw_df[raw_df["produk"] == produk].sort_values("bulan_tahun")
    if len(subset) < seq_len:
        raise ValueError(f"Data {produk} kurang dari {seq_len} baris.")

    scaler = scalers.get(produk)
    if scaler is None:
        raise KeyError(f"Scaler untuk {produk} tidak ditemukan.")

    # Ambil lookback terakhir
    raw_vals = subset["stok_akhir_kg"].values[-seq_len:].astype(float)
    history_months = subset["bulan_tahun"].values[-seq_len:].tolist()
    scaled   = scaler.transform(raw_vals.reshape(-1, 1)).flatten()

    # Tensor (1, 1, seq_len)
    x = torch.from_numpy(scaled.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        y_norm = model(x).cpu().numpy().flatten()   # (pred_len,)

    # Inverse transform
    y_pred = scaler.inverse_transform(y_norm.reshape(-1, 1)).flatten()
    y_pred = np.clip(y_pred, 0, None)                # stok tidak bisa negatif

    # Bulan prediksi: mulai dari bulan setelah data terakhir
    last_month = subset["bulan_tahun"].iloc[-1]
    last_date = pd.to_datetime(last_month + "-01")
    bulan_pred = pd.date_range(last_date + pd.DateOffset(months=1), periods=pred_len, freq="MS")

    return {
        "produk":          produk,
        "bulan_prediksi":  [t.strftime("%Y-%m") for t in bulan_pred],
        "prediksi_stok":   [round(float(v), 2) for v in y_pred],
        "last_history":    [round(float(v), 2) for v in raw_vals],
        "history_months":  history_months,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json

    p = argparse.ArgumentParser()
    p.add_argument("--produk", type=str, default="Chelate HN", help="Nama produk")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scalers, cfg = load_model_and_scalers(device)
    raw_df = pd.read_csv(CSV_PATH)

    result = predict_product(args.produk, model, scalers, cfg, device, raw_df)

    print(f"\n{'='*50}")
    print(f"  Prediksi Stok: {result['produk']}")
    print(f"{'='*50}")
    for tgl, val in zip(result["bulan_prediksi"], result["prediksi_stok"]):
        max_val = max(result["prediksi_stok"]) if max(result["prediksi_stok"]) > 0 else 1
        bar = "█" * max(1, int(val / max_val * 30))
        print(f"  {tgl}  {val:10.1f} kg  {bar}")
