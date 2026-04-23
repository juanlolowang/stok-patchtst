# 🏭 Prediksi Kebutuhan Stok Bahan Jadi — PatchTST

> Sistem cerdas berbasis **PatchTST** (Patch Time-Series Transformer) untuk memprediksi kebutuhan stok bahan jadi bulanan menggunakan data historis dari gudang.

---

## 📁 Struktur Project

```
stok-patchtst/
├── data/
│   ├── DATA BAHAN JADI.xlsx  ← Data asli (Excel)
│   └── processed/            ← bahan_jadi_bulanan.csv, X_train/val/test.npy
├── src/
│   ├── parse_excel.py        ← Parser Excel ke CSV bulanan
│   ├── preprocess.py         ← Normalisasi & sliding window (6 bulan lookback)
│   ├── model.py              ← Arsitektur PatchTST (PyTorch)
│   ├── train.py              ← Training pipeline
│   ├── evaluate.py           ← Evaluasi & plot
│   └── predict.py            ← Inferensi produk tertentu
├── models/
│   ├── best_model.pt         ← Checkpoint terbaik
│   ├── scalers.pkl           ← MinMaxScaler per produk
│   └── history.json          ← Training loss history
├── app/
│   └── streamlit_app.py      ← Dashboard web premium
└── requirements.txt
```

---

## 🚀 Cara Menjalankan

### 1. Setup Environment

```bash
cd stok-patchtst
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

### 2. Parse Data Excel
Konversi data Excel asli ke format CSV yang siap diproses.
```bash
python src/parse_excel.py
```
Output: `data/processed/bahan_jadi_bulanan.csv`

### 3. Preprocessing
Melakukan normalisasi dan pembuatan windowing (6 bulan input -> 3 bulan prediksi).
```bash
python src/preprocess.py
```

### 4. Training Model
```bash
# Training model PatchTST
python src/train.py
```

### 5. Evaluasi
```bash
python src/evaluate.py
```

### 6. Jalankan Dashboard Web
```bash
streamlit run app/streamlit_app.py
```
Buka browser → `http://localhost:8501`

---

## 🏗 Arsitektur & Hyperparameter

- **Input**: Sequence 6 bulan terakhir (`seq_len=6`)
- **Output**: Prediksi 3 bulan ke depan (`pred_len=3`)
- **Patching**: `patch_len=3`, `stride=1`
- **Model**: 2 Layer Transformer, 4 Attention Heads, `d_model=64`

---

## 📊 Fitur Dashboard

- **Beranda**: KPI Stok, Status Kritis, dan Tren Total Kirim.
- **EDA**: Analisis tren per produk, pola musiman bulanan, dan perbandingan Terima vs Kirim.
- **Prediksi**: Prediksi stok 3 bulan ke depan dengan confidence band.
- **Evaluasi**: Metrik performa model (MAE, RMSE, MAPE) dan Learning Curve.
