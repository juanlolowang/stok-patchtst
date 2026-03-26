"""
preprocess.py
Normalisasi, pembuatan sliding window, dan split data untuk training PatchTST.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
LOOKBACK   = 60   # panjang input sekuens (hari)
HORIZON    = 14   # panjang prediksi (hari ke depan)
TARGET_COL = "stok_akhir"
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# sisa = 0.15 → test


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["tanggal"])
    df.sort_values(["kode_produk", "tanggal"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def make_windows(series: np.ndarray, lookback: int, horizon: int):
    """
    Membuat pasangan (X, y) dari time series 1D.
    X.shape = (N, lookback)
    y.shape = (N, horizon)
    """
    X, y = [], []
    total = len(series)
    for i in range(total - lookback - horizon + 1):
        X.append(series[i : i + lookback])
        y.append(series[i + lookback : i + lookback + horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def preprocess(raw_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    df = load_data(raw_path)
    products = df["kode_produk"].unique()

    all_X_train, all_y_train = [], []
    all_X_val,   all_y_val   = [], []
    all_X_test,  all_y_test  = [], []

    scalers = {}

    print(f"Preprocessing {len(products)} produk...")

    for kode in products:
        subset = df[df["kode_produk"] == kode][TARGET_COL].values.astype(float)

        # Normalisasi per-produk
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(subset.reshape(-1, 1)).flatten()
        scalers[kode] = scaler

        # Train / Val / Test split (temporal)
        n = len(scaled)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        train_s = scaled[:n_train]
        val_s   = scaled[n_train : n_train + n_val]
        test_s  = scaled[n_train + n_val:]

        X_tr, y_tr = make_windows(train_s, LOOKBACK, HORIZON)
        X_va, y_va = make_windows(val_s,   LOOKBACK, HORIZON)
        X_te, y_te = make_windows(test_s,  LOOKBACK, HORIZON)

        all_X_train.append(X_tr); all_y_train.append(y_tr)
        all_X_val.append(X_va);   all_y_val.append(y_va)
        all_X_test.append(X_te);  all_y_test.append(y_te)

        print(f"  {kode}: total={n} | train={len(X_tr)} | val={len(X_va)} | test={len(X_te)}")

    # Gabungkan semua produk
    def stack(lst):
        return np.concatenate(lst, axis=0) if lst else np.array([])

    X_train, y_train = stack(all_X_train), stack(all_y_train)
    X_val,   y_val   = stack(all_X_val),   stack(all_y_val)
    X_test,  y_test  = stack(all_X_test),  stack(all_y_test)

    # Tambah channel dimension → (N, 1, L) untuk kompatibilitas PatchTST
    X_train = X_train[:, np.newaxis, :]
    X_val   = X_val[:, np.newaxis, :]
    X_test  = X_test[:, np.newaxis, :]

    # Simpan
    np.save(os.path.join(out_dir, "X_train.npy"), X_train)
    np.save(os.path.join(out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(out_dir, "X_val.npy"),   X_val)
    np.save(os.path.join(out_dir, "y_val.npy"),   y_val)
    np.save(os.path.join(out_dir, "X_test.npy"),  X_test)
    np.save(os.path.join(out_dir, "y_test.npy"),  y_test)

    # Simpan scaler
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    scaler_path = os.path.join(project_root, "models", "scalers.pkl")
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    joblib.dump(scalers, scaler_path)

    print(f"\n✅ Preprocessing selesai.")
    print(f"   X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"   X_val:   {X_val.shape}   | y_val:   {y_val.shape}")
    print(f"   X_test:  {X_test.shape}  | y_test:  {y_test.shape}")
    print(f"   Scalers disimpan → {os.path.abspath(scaler_path)}")


if __name__ == "__main__":
    BASE = os.path.dirname(__file__)
    raw_path = os.path.join(BASE, "..", "data", "raw", "stok_obat_kain.csv")
    out_dir  = os.path.join(BASE, "..", "data", "processed")
    preprocess(raw_path, out_dir)
