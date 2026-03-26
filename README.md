# 🏥 Prediksi Kebutuhan Stok Bahan Jadi Obat Kain — PatchTST

> Sistem cerdas berbasis **PatchTST** (Patch Time-Series Transformer) untuk memprediksi kebutuhan stok bahan jadi obat kain (kasa, perban, plester, dll.)

---

## 📁 Struktur Project

```
stok-patchtst/
├── data/
│   ├── raw/                  ← stok_obat_kain.csv (data sintetis)
│   └── processed/            ← X_train/val/test.npy, y_*.npy
├── src/
│   ├── generate_data.py      ← pembuat dataset sintetis
│   ├── preprocess.py         ← normalisasi & sliding window
│   ├── model.py              ← arsitektur PatchTST (PyTorch)
│   ├── train.py              ← training pipeline
│   ├── evaluate.py           ← evaluasi & plot
│   └── predict.py            ← inferensi produk tertentu
├── models/
│   ├── best_model.pt         ← checkpoint terbaik
│   ├── scalers.pkl           ← MinMaxScaler per produk
│   └── history.json          ← training loss history
├── outputs/
│   ├── evaluation.png        ← grafik prediksi vs aktual
│   └── metrics.json          ← MAE, RMSE, MAPE
├── app/
│   └── streamlit_app.py      ← dashboard web
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

### 2. Generate Data Sintetis

```bash
python src/generate_data.py
```

Output: `data/raw/stok_obat_kain.csv` (8 produk × 3 tahun = ~8.760 baris)

### 3. Preprocessing

```bash
python src/preprocess.py
```

Output: array `.npy` di `data/processed/` dan `models/scalers.pkl`

### 4. Training Model

```bash
# Full training (hingga 100 epoch + early stopping)
python src/train.py

# Debug cepat (5 epoch saja)
python src/train.py --debug
```

Output: `models/best_model.pt` + `models/history.json`

### 5. Evaluasi

```bash
python src/evaluate.py
```

Output: metrik MAE/RMSE/MAPE + `outputs/evaluation.png`

### 6. Prediksi CLI

```bash
python src/predict.py --kode KSA-001
```

### 7. Jalankan Dashboard Web

```bash
streamlit run app/streamlit_app.py
```

Buka browser → `http://localhost:8501`

---

## 🛠 Tech Stack

| Komponen        | Library/Framework |
|-----------------|-------------------|
| Model           | PyTorch 2.x       |
| Preprocessing   | scikit-learn, numpy |
| Data            | pandas            |
| Dashboard       | Streamlit + Plotly|
| Utilitas        | joblib, tqdm      |

---

## 📦 Produk yang Dimodelkan

| Kode     | Nama Produk              | Satuan |
|----------|--------------------------|--------|
| KSA-001  | Kasa Steril 16x16cm      | pcs    |
| KSA-002  | Kasa Non-Steril Roll     | roll   |
| PRB-001  | Perban Elastis 5cm       | roll   |
| PRB-002  | Perban Elastis 10cm      | roll   |
| PLS-001  | Plester Luka Kecil       | strip  |
| PLS-002  | Plester Luka Sedang      | strip  |
| KPT-001  | Kapas Medis 100gr        | pak    |
| VRP-001  | Verban Pembalut Gip      | roll   |

---

## 🏗 Arsitektur PatchTST

```
Input (B, 1, 60)
    ↓
Patching → Patch Embedding
    ↓  [num_patches × d_model]
Positional Encoding
    ↓
Transformer Encoder (3 layer, 8 head, Pre-LN)
    ↓
Flatten → Linear Head
    ↓
Output (B, 14)  ← prediksi 14 hari ke depan
```

**Hyperparameter default:**

- `seq_len=60`, `pred_len=14`, `patch_len=16`, `stride=8`
- `d_model=128`, `n_heads=8`, `n_layers=3`, `d_ff=256`
- Optimizer: AdamW + CosineAnnealingLR
- Early stopping: patience=10

---

## 📊 Halaman Dashboard

| Halaman        | Fitur |
|----------------|-------|
| 🏠 Beranda     | KPI cards, status stok terkini, tren pemakaian |
| 📊 EDA         | Tren stok, pola musiman, distribusi, pemakaian vs pengadaan |
| 🔮 Prediksi    | Pilih produk & horizon, grafik + confidence band + rekomendasi |
| 📈 Evaluasi    | Metrik MAE/RMSE/MAPE, learning curve, konfigurasi model |
