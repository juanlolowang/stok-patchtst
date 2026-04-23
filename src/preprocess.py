"""
preprocess.py
Normalisasi, pembuatan sliding window, dan split data untuk training PatchTST.
Disesuaikan untuk data bulanan bahan jadi dari parse_excel.py.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
LOOKBACK    = 6    # panjang input sekuens (bulan)
HORIZON     = 3    # panjang prediksi (bulan ke depan)
TARGET_COL  = "stok_akhir_kg"
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# sisa = 0.15 → test


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.sort_values(["produk", "bulan_tahun"], inplace=True)
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


def preprocess(csv_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    df = load_data(csv_path)
    products = df["produk"].unique()

    all_X_train, all_y_train = [], []
    all_X_val,   all_y_val   = [], []
    all_X_test,  all_y_test  = [], []

    scalers = {}

    print(f"Preprocessing {len(products)} produk (LOOKBACK={LOOKBACK}, HORIZON={HORIZON})...")

    skipped = 0
    for produk in products:
        subset = df[df["produk"] == produk][TARGET_COL].values.astype(float)

        # Skip produk dengan data kurang dari minimum requirement
        min_len = LOOKBACK + HORIZON + 3  # minimal bisa bikin beberapa window
        if len(subset) < min_len:
            skipped += 1
            continue

        # Normalisasi per-produk
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(subset.reshape(-1, 1)).flatten()
        scalers[produk] = scaler

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

        if len(X_tr) > 0:
            all_X_train.append(X_tr); all_y_train.append(y_tr)
        if len(X_va) > 0:
            all_X_val.append(X_va);   all_y_val.append(y_va)
        if len(X_te) > 0:
            all_X_test.append(X_te);  all_y_test.append(y_te)

        print(f"  {produk:40s}: total={n:3d} | train={len(X_tr)} | val={len(X_va)} | test={len(X_te)}")

    if skipped > 0:
        print(f"\n  ⚠ {skipped} produk diskip (data < {LOOKBACK + HORIZON + 3} bulan)")

    # Gabungkan semua produk
    def stack(lst):
        return np.concatenate(lst, axis=0) if lst else np.array([]).reshape(0, LOOKBACK)

    X_train, y_train = stack(all_X_train), stack(all_y_train) if all_y_train else np.array([]).reshape(0, HORIZON)
    X_val,   y_val   = stack(all_X_val),   stack(all_y_val) if all_y_val else np.array([]).reshape(0, HORIZON)
    X_test,  y_test  = stack(all_X_test),  stack(all_y_test) if all_y_test else np.array([]).reshape(0, HORIZON)

    # Tambah channel dimension → (N, 1, L) untuk kompatibilitas PatchTST
    if X_train.ndim == 2 and len(X_train) > 0:
        X_train = X_train[:, np.newaxis, :]
    if X_val.ndim == 2 and len(X_val) > 0:
        X_val = X_val[:, np.newaxis, :]
    if X_test.ndim == 2 and len(X_test) > 0:
        X_test = X_test[:, np.newaxis, :]

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
    print(f"   Scalers ({len(scalers)} produk) disimpan → {os.path.abspath(scaler_path)}")


if __name__ == "__main__":
    BASE = os.path.dirname(__file__)
    csv_path = os.path.join(BASE, "..", "data", "processed", "bahan_jadi_bulanan.csv")
    out_dir  = os.path.join(BASE, "..", "data", "processed")
    preprocess(csv_path, out_dir)
