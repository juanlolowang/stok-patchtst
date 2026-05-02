"""
generate_thesis_figures.py
Membuat seluruh gambar/grafik untuk Bab 4 Tugas Akhir.

Output (disimpan ke outputs/thesis_figures/):
  - gambar_4_1_tren_stok_bulanan.png        → Sub-bab 4.2 Eksplorasi Data
  - gambar_4_1b_distribusi_stok.png          → Sub-bab 4.2 Eksplorasi Data (tambahan)
  - gambar_4_2_learning_curve.png            → Sub-bab 4.4 Evaluasi Model
  - gambar_4_3_prediksi_vs_aktual.png        → Sub-bab 4.4 Evaluasi Model
  - gambar_4_3b_scatter_pred_vs_actual.png   → Sub-bab 4.4 Evaluasi Model (tambahan)
  - gambar_4_3c_residual_plot.png            → Sub-bab 4.4 Evaluasi Model (tambahan)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings("ignore")

# Paths
BASE       = os.path.dirname(__file__)
DATA_DIR   = os.path.join(BASE, "..", "data", "processed")
MODELS_DIR = os.path.join(BASE, "..", "models")
OUT_DIR    = os.path.join(BASE, "..", "outputs", "thesis_figures")
os.makedirs(OUT_DIR, exist_ok=True)

# Style config — Clean, professional style for thesis (white background)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Color palette (professional)
C_PRIMARY   = "#2563EB"  # Blue
C_SECONDARY = "#DC2626"  # Red
C_TERTIARY  = "#059669"  # Green
C_ACCENT    = "#D97706"  # Amber
C_GRAY      = "#6B7280"
C_LIGHT     = "#E5E7EB"

CSV_PATH = os.path.join(DATA_DIR, "bahan_jadi_bulanan.csv")


def load_data():
    df = pd.read_csv(CSV_PATH)
    df.sort_values(["produk", "bulan_tahun"], inplace=True)
    return df


# ============================================================================
# GAMBAR 4.1 — Tren Stok Akhir Bulanan (Sub-bab 4.2 Eksplorasi Data)
# ============================================================================
def generate_gambar_4_1(df):
    """Grafik tren stok akhir bulanan untuk beberapa produk representatif."""
    print("📊 Generating Gambar 4.1: Tren Stok Akhir Bulanan...")

    # Pilih top 5 produk dengan rata-rata stok tertinggi (yang punya variasi)
    avg_stok = df.groupby("produk")["stok_akhir_kg"].mean()
    std_stok = df.groupby("produk")["stok_akhir_kg"].std()
    # Produk dengan variasi yang menarik (std > 0)
    active_products = std_stok[std_stok > 0].index
    top_products = avg_stok[avg_stok.index.isin(active_products)].nlargest(5).index.tolist()

    if not top_products:
        top_products = avg_stok.nlargest(5).index.tolist()

    fig, ax = plt.subplots(figsize=(12, 5.5))

    colors = [C_PRIMARY, C_SECONDARY, C_TERTIARY, C_ACCENT, "#7C3AED"]
    for i, produk in enumerate(top_products):
        sub = df[df["produk"] == produk].sort_values("bulan_tahun")
        ax.plot(sub["bulan_tahun"], sub["stok_akhir_kg"],
                marker="o", markersize=3, linewidth=1.5,
                color=colors[i % len(colors)], label=produk, alpha=0.85)

    ax.set_xlabel("Periode (Bulan-Tahun)")
    ax.set_ylabel("Stok Akhir (kg)")
    ax.set_title("Tren Stok Akhir Bulanan — 5 Produk Teratas", fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.9, edgecolor=C_LIGHT)

    # Rotate x labels
    tick_positions = range(0, len(sub), max(1, len(sub) // 10))
    ax.set_xticks([sub["bulan_tahun"].iloc[i] for i in tick_positions if i < len(sub)])
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "gambar_4_1_tren_stok_bulanan.png")
    plt.savefig(path, facecolor="white")
    plt.close()
    print(f"   ✅ Saved: {path}")


# ============================================================================
# GAMBAR 4.1b — Distribusi Stok Akhir (Sub-bab 4.2 Eksplorasi Data, tambahan)
# ============================================================================
def generate_gambar_4_1b(df):
    """Boxplot distribusi stok akhir untuk top produk."""
    print("📊 Generating Gambar 4.1b: Distribusi Stok Akhir...")

    avg_stok = df.groupby("produk")["stok_akhir_kg"].mean()
    top10 = avg_stok.nlargest(10).index.tolist()
    df_top = df[df["produk"].isin(top10)]

    fig, ax = plt.subplots(figsize=(12, 5.5))

    bp = ax.boxplot(
        [df_top[df_top["produk"] == p]["stok_akhir_kg"].values for p in top10],
        labels=top10, patch_artist=True, notch=True,
        boxprops=dict(linewidth=1.2),
        medianprops=dict(color=C_SECONDARY, linewidth=2),
        whiskerprops=dict(linewidth=1),
        capprops=dict(linewidth=1),
    )

    box_colors = [C_PRIMARY, "#3B82F6", "#60A5FA", "#93C5FD", "#BFDBFE",
                  C_TERTIARY, "#34D399", "#6EE7B7", "#A7F3D0", "#D1FAE5"]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_xlabel("Produk")
    ax.set_ylabel("Stok Akhir (kg)")
    ax.set_title("Distribusi Stok Akhir — 10 Produk Teratas", fontweight="bold", pad=12)
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "gambar_4_1b_distribusi_stok.png")
    plt.savefig(path, facecolor="white")
    plt.close()
    print(f"   ✅ Saved: {path}")


# ============================================================================
# GAMBAR 4.2 — Learning Curve (Sub-bab 4.4 Evaluasi Model)
# ============================================================================
def generate_gambar_4_2():
    """Learning Curve: Train Loss vs Validation Loss."""
    print("📊 Generating Gambar 4.2: Learning Curve...")

    hist_path = os.path.join(MODELS_DIR, "history.json")
    if not os.path.exists(hist_path):
        print("   ⚠ history.json tidak ditemukan. Jalankan train.py dulu.")
        return

    with open(hist_path) as f:
        history = json.load(f)

    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    epochs = range(1, len(train_loss) + 1)

    fig, ax = plt.subplots(figsize=(10, 5.5))

    ax.plot(epochs, train_loss, color=C_TERTIARY, linewidth=2, label="Training Loss", alpha=0.85)
    ax.plot(epochs, val_loss, color=C_SECONDARY, linewidth=2, label="Validation Loss", alpha=0.85)

    # Tandai epoch konvergensi (val loss stabil)
    min_val_idx = np.argmin(val_loss)
    min_val = val_loss[min_val_idx]
    ax.axvline(x=min_val_idx + 1, color=C_GRAY, linestyle=":", alpha=0.6, linewidth=1)
    ax.annotate(f"Best Val Loss: {min_val:.4f}\n(Epoch {min_val_idx + 1})",
                xy=(min_val_idx + 1, min_val),
                xytext=(min_val_idx + 10, min_val + 0.015),
                fontsize=9, color=C_GRAY,
                arrowprops=dict(arrowstyle="->", color=C_GRAY, lw=1.2))

    # Shaded area di bawah val loss
    ax.fill_between(epochs, val_loss, alpha=0.08, color=C_SECONDARY)
    ax.fill_between(epochs, train_loss, alpha=0.08, color=C_TERTIARY)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Learning Curve — Model PatchTST", fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.9, edgecolor=C_LIGHT)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "gambar_4_2_learning_curve.png")
    plt.savefig(path, facecolor="white")
    plt.close()
    print(f"   ✅ Saved: {path}")


# ============================================================================
# GAMBAR 4.3 — Prediksi vs Aktual (Sub-bab 4.4 Evaluasi Model)
# ============================================================================
def generate_gambar_4_3():
    """Plot Prediksi vs Aktual pada Test Set."""
    print("📊 Generating Gambar 4.3: Prediksi vs Aktual...")

    X_test_path = os.path.join(DATA_DIR, "X_test.npy")
    y_test_path = os.path.join(DATA_DIR, "y_test.npy")
    model_path = os.path.join(MODELS_DIR, "best_model.pt")

    if not all(os.path.exists(p) for p in [X_test_path, y_test_path, model_path]):
        print("   ⚠ File test data / model tidak lengkap.")
        return

    import torch
    sys.path.insert(0, os.path.dirname(__file__))
    from model import PatchTST

    device = torch.device("cpu")
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = PatchTST(
        seq_len=cfg["seq_len"], pred_len=cfg["pred_len"],
        patch_len=cfg["patch_len"], stride=cfg["stride"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], d_ff=cfg["d_ff"],
        dropout=cfg["dropout"], n_channels=cfg["n_channels"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    X_test = torch.from_numpy(np.load(X_test_path)).to(device)
    y_test = np.load(y_test_path)

    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()

    # Ambil langkah pertama horizon
    N = min(200, len(y_test))
    y_true_flat = y_test[:N, 0]
    y_pred_flat = y_pred[:N, 0]

    fig, ax = plt.subplots(figsize=(12, 5.5))

    ax.plot(range(N), y_true_flat, color=C_PRIMARY, linewidth=1.8,
            label="Aktual", alpha=0.9)
    ax.plot(range(N), y_pred_flat, color=C_SECONDARY, linewidth=1.8,
            linestyle="--", label="Prediksi", alpha=0.85)

    # Shaded error area
    ax.fill_between(range(N), y_true_flat, y_pred_flat,
                     alpha=0.12, color=C_ACCENT, label="Area Error")

    ax.set_xlabel("Indeks Sampel")
    ax.set_ylabel("Stok Akhir (Ternormalisasi)")
    ax.set_title("Prediksi vs Aktual — Test Set (Langkah Pertama Horizon)",
                  fontweight="bold", pad=12)
    ax.legend(loc="upper right", framealpha=0.9, edgecolor=C_LIGHT)

    # Tambahkan info metrik di pojok
    mae_v = np.mean(np.abs(y_true_flat - y_pred_flat))
    mse_v = np.mean((y_true_flat - y_pred_flat) ** 2)
    textstr = f"MAE = {mae_v:.6f}\nMSE = {mse_v:.6f}"
    props = dict(boxstyle="round,pad=0.5", facecolor="#F3F4F6", edgecolor=C_LIGHT, alpha=0.9)
    ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=props)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "gambar_4_3_prediksi_vs_aktual.png")
    plt.savefig(path, facecolor="white")
    plt.close()
    print(f"   ✅ Saved: {path}")

    return y_true_flat, y_pred_flat


# ============================================================================
# GAMBAR 4.3b — Scatter Plot Pred vs Actual (Sub-bab 4.4, tambahan)
# ============================================================================
def generate_gambar_4_3b(y_true, y_pred):
    """Scatter plot antara prediksi dan aktual."""
    print("📊 Generating Gambar 4.3b: Scatter Prediksi vs Aktual...")

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    ax.scatter(y_true, y_pred, color=C_PRIMARY, alpha=0.5, s=25, edgecolors="white", linewidth=0.5)

    # Perfect prediction line
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, color=C_SECONDARY, linestyle="--", linewidth=1.5,
            label="Garis Ideal (y = x)", alpha=0.7)

    # R² score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    textstr = f"R² = {r2:.4f}"
    props = dict(boxstyle="round,pad=0.5", facecolor="#F3F4F6", edgecolor=C_LIGHT, alpha=0.9)
    ax.text(0.05, 0.92, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=props, fontweight="bold")

    ax.set_xlabel("Nilai Aktual (Ternormalisasi)")
    ax.set_ylabel("Nilai Prediksi (Ternormalisasi)")
    ax.set_title("Scatter Plot — Prediksi vs Aktual", fontweight="bold", pad=12)
    ax.legend(loc="lower right", framealpha=0.9, edgecolor=C_LIGHT)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "gambar_4_3b_scatter_pred_vs_actual.png")
    plt.savefig(path, facecolor="white")
    plt.close()
    print(f"   ✅ Saved: {path}")


# ============================================================================
# GAMBAR 4.3c — Residual Plot (Sub-bab 4.4, tambahan)
# ============================================================================
def generate_gambar_4_3c(y_true, y_pred):
    """Residual plot (error distribution)."""
    print("📊 Generating Gambar 4.3c: Residual Plot...")

    residuals = y_true - y_pred

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [2, 1]})

    # Plot 1: Residual per sampel
    ax1 = axes[0]
    ax1.scatter(range(len(residuals)), residuals, color=C_PRIMARY, alpha=0.5, s=15, edgecolors="white", linewidth=0.3)
    ax1.axhline(y=0, color=C_SECONDARY, linestyle="--", linewidth=1.2, alpha=0.7)
    ax1.fill_between(range(len(residuals)), residuals, alpha=0.08, color=C_PRIMARY)
    ax1.set_xlabel("Indeks Sampel")
    ax1.set_ylabel("Residual (Aktual − Prediksi)")
    ax1.set_title("Distribusi Residual per Sampel", fontweight="bold", pad=10)

    # Plot 2: Histogram residual
    ax2 = axes[1]
    ax2.hist(residuals, bins=30, color=C_PRIMARY, alpha=0.6, edgecolor="white", linewidth=0.8, density=True)
    ax2.axvline(x=0, color=C_SECONDARY, linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.set_xlabel("Residual")
    ax2.set_ylabel("Densitas")
    ax2.set_title("Histogram Residual", fontweight="bold", pad=10)

    # Stats
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    textstr = f"μ = {mean_res:.5f}\nσ = {std_res:.5f}"
    props = dict(boxstyle="round,pad=0.5", facecolor="#F3F4F6", edgecolor=C_LIGHT, alpha=0.9)
    ax2.text(0.95, 0.92, textstr, transform=ax2.transAxes, fontsize=9,
             verticalalignment="top", horizontalalignment="right", bbox=props)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "gambar_4_3c_residual_plot.png")
    plt.savefig(path, facecolor="white")
    plt.close()
    print(f"   ✅ Saved: {path}")


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 60)
    print("  GENERATOR GAMBAR BAB 4 TUGAS AKHIR")
    print("  Prediksi Stok Bahan Jadi — PatchTST")
    print("  PT. Seikyo Indochem")
    print("=" * 60)
    print()

    # Load data
    if os.path.exists(CSV_PATH):
        df = load_data()
        print(f"📂 Data loaded: {len(df):,} baris, {df['produk'].nunique()} produk\n")
    else:
        print("⚠ CSV tidak ditemukan. Hanya gambar model yang akan digenerate.\n")
        df = None

    # Sub-bab 4.2 — Eksplorasi Data
    if df is not None:
        generate_gambar_4_1(df)
        generate_gambar_4_1b(df)

    # Sub-bab 4.4 — Evaluasi Model
    generate_gambar_4_2()
    result = generate_gambar_4_3()
    if result is not None:
        y_true, y_pred = result
        generate_gambar_4_3b(y_true, y_pred)
        generate_gambar_4_3c(y_true, y_pred)

    print()
    print("=" * 60)
    print("  ✅ SEMUA GAMBAR BERHASIL DIGENERATE!")
    print(f"  📁 Lokasi: {os.path.abspath(OUT_DIR)}")
    print("=" * 60)
    print()
    print("📝 Panduan Penempatan di Bab 4:")
    print("   Gambar 4.1  → Sub-bab 4.2 Eksplorasi Data (Tren Stok)")
    print("   Gambar 4.1b → Sub-bab 4.2 Eksplorasi Data (Distribusi)")
    print("   Gambar 4.2  → Sub-bab 4.4 Learning Curve")
    print("   Gambar 4.3  → Sub-bab 4.4 Prediksi vs Aktual")
    print("   Gambar 4.3b → Sub-bab 4.4 Scatter Plot")
    print("   Gambar 4.3c → Sub-bab 4.4 Residual Analysis")
    print("   Gambar 4.4  → Sub-bab 4.5 Screenshot Dashboard Streamlit")
    print()


if __name__ == "__main__":
    main()
