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
# Rasio keseluruhan (Train+Val) : Test = 80 : 20 sesuai Bab III
# Di dalam 80%, kita bagi lagi untuk monitoring training
TRAIN_SPLIT = 0.80 


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
    return np.array(X), np.array(y)


def preprocess():
    base = os.path.dirname(__file__)
    csv_path = os.path.join(base, "..", "data", "processed", "bahan_jadi_bulanan.csv")
    out_dir  = os.path.join(base, "..", "data", "processed")
    model_dir = os.path.join(base, "..", "models")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        print(f"❌ File {csv_path} tidak ditemukan. Jalankan parse_excel.py dulu.")
        return

    df = load_data(csv_path)
    products = df["produk"].unique()

    all_X_train, all_y_train = [], []
    all_X_val,   all_y_val   = [], []
    all_X_test,  all_y_test  = [], []

    scalers = {}
    skipped = 0

    print(f"🔄 Memproses {len(products)} produk...")

    for produk in products:
        sub_df = df[df["produk"] == produk]
        if len(sub_df) < (LOOKBACK + HORIZON):
            skipped += 1
            continue

        subset = sub_df[TARGET_COL].values.astype(float)
        
        # Normalisasi (Min-Max Scaling 0-1) sesuai Bab III
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(subset.reshape(-1, 1)).flatten()
        scalers[produk] = scaler

        # Buat windows dulu, baru split biar lebih efisien buat data pendek
        X, y = make_windows(scaled, LOOKBACK, HORIZON)
        
        if len(X) == 0:
            skipped += 1
            continue

        # Split kronologis: 80% Latih, 20% Uji sesuai Bab III
        n = len(X)
        n_train_total = int(n * TRAIN_SPLIT)
        
        # Di dalam data Latih (80%), kita ambil sedikit buat Validasi (misal 10% dari total)
        # supaya proses training bisa dipantau early stopping-nya.
        n_val = max(1, int(n * 0.10))
        n_train = n_train_total - n_val
        
        if n_train <= 0: # jika data terlalu pendek, gabung ke train aja
            X_tr, y_tr = X[:n_train_total], y[:n_train_total]
            X_va, y_va = X[n_train_total-1:n_train_total], y[n_train_total-1:n_train_total]
        else:
            X_tr, y_tr = X[:n_train], y[:n_train]
            X_va, y_va = X[n_train:n_train_total], y[n_train:n_train_total]
            
        X_te, y_te = X[n_train_total:], y[n_train_total:]

        all_X_train.append(X_tr); all_y_train.append(y_tr)
        if len(X_va) > 0:
            all_X_val.append(X_va);   all_y_val.append(y_va)
        if len(X_te) > 0:
            all_X_test.append(X_te);  all_y_test.append(y_te)

    if skipped > 0:
        print(f"\n  ⚠ {skipped} produk diskip (data terlalu pendek untuk windowing)")

    # Gabungkan semua produk
    def stack(lst):
        return np.concatenate(lst, axis=0) if lst else np.array([]).reshape(0, LOOKBACK)

    X_train, y_train = stack(all_X_train), stack(all_y_train)
    X_val,   y_val   = stack(all_X_val),   stack(all_y_val)
    X_test,  y_test  = stack(all_X_test),  stack(all_y_test)

    # Tambah channel dimension → (N, 1, L) untuk kompatibilitas PatchTST
    if X_train.ndim == 2 and len(X_train) > 0:
        X_train = X_train[:, np.newaxis, :]
    if X_val.ndim == 2 and len(X_val) > 0:
        X_val = X_val[:, np.newaxis, :]
    if X_test.ndim == 2 and len(X_test) > 0:
        X_test = X_test[:, np.newaxis, :]

    # Simpan
    np.save(os.path.join(out_dir, "X_train.npy"), X_train.astype(np.float32))
    np.save(os.path.join(out_dir, "y_train.npy"), y_train.astype(np.float32))
    np.save(os.path.join(out_dir, "X_val.npy"),   X_val.astype(np.float32))
    np.save(os.path.join(out_dir, "y_val.npy"),   y_val.astype(np.float32))
    np.save(os.path.join(out_dir, "X_test.npy"),  X_test.astype(np.float32))
    np.save(os.path.join(out_dir, "y_test.npy"),  y_test.astype(np.float32))
    joblib.dump(scalers, os.path.join(model_dir, "scalers.pkl"))

    print(f"\n✅ Preprocessing selesai.")
    print(f"   X_train: {X_train.shape} | y_train: {y_train.shape}")
    print(f"   X_val:   {X_val.shape}   | y_val:   {y_val.shape}")
    print(f"   X_test:  {X_test.shape}  | y_test:  {y_test.shape}")
    print(f"   Scalers ({len(scalers)} produk) disimpan → {os.path.abspath(os.path.join(model_dir, 'scalers.pkl'))}")


if __name__ == "__main__":
    preprocess()
