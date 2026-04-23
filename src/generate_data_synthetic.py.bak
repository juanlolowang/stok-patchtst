"""
generate_data.py
Membuat dataset sintetis stok bahan jadi obat kain (kasa, perban, plester, dll.)
selama 3 tahun dengan pola realistis: tren, seasonality, noise, lonjakan hari raya.
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta
import os

# ---------------------------------------------------------------------------
# Konfigurasi produk
# ---------------------------------------------------------------------------
PRODUCTS = [
    {"kode": "KSA-001", "nama": "Kasa Steril 16x16cm",    "satuan": "pcs",  "base_demand": 320, "trend": 0.08},
    {"kode": "KSA-002", "nama": "Kasa Non-Steril Roll",   "satuan": "roll", "base_demand": 150, "trend": 0.05},
    {"kode": "PRB-001", "nama": "Perban Elastis 5cm",     "satuan": "roll", "base_demand": 200, "trend": 0.06},
    {"kode": "PRB-002", "nama": "Perban Elastis 10cm",    "satuan": "roll", "base_demand": 130, "trend": 0.04},
    {"kode": "PLS-001", "nama": "Plester Luka Kecil",     "satuan": "strip","base_demand": 500, "trend": 0.10},
    {"kode": "PLS-002", "nama": "Plester Luka Sedang",    "satuan": "strip","base_demand": 280, "trend": 0.07},
    {"kode": "KPT-001", "nama": "Kapas Medis 100gr",      "satuan": "pak",  "base_demand": 180, "trend": 0.05},
    {"kode": "VRP-001", "nama": "Verban Pembalut Gip",    "satuan": "roll", "base_demand": 90,  "trend": 0.03},
]

# Hari libur nasional Indonesia (tanggal-bulan format MM-DD) — perkiraan
NATIONAL_HOLIDAYS_MD = {
    "01-01",  # Tahun Baru
    "02-09",  # Isra Mi'raj (perkiraan)
    "03-21",  # Hari Raya Nyepi
    "04-07",  # Wafat Isa Al-Masih
    "04-09",  # Hari Raya Idul Fitri (contoh)
    "04-10",  # Hari Raya Idul Fitri ke-2
    "05-01",  # Hari Buruh
    "05-09",  # Kenaikan Isa Al-Masih
    "05-29",  # Hari Raya Waisak
    "06-01",  # Hari Pancasila
    "06-16",  # Idul Adha (perkiraan)
    "07-07",  # Tahun Baru Hijriyah
    "08-17",  # HUT RI
    "09-15",  # Maulid Nabi (perkiraan)
    "12-25",  # Natal
    "12-26",  # Cuti bersama Natal
}

np.random.seed(42)


def is_holiday(d: date) -> bool:
    return d.strftime("%m-%d") in NATIONAL_HOLIDAYS_MD or d.weekday() >= 5  # weekend juga


def seasonal_factor(d: date) -> float:
    """
    Pola meningkat di bulan-bulan tertentu:
    - Januari (setelah liburan akhir tahun): spike ringan
    - Maret–April (Ramadhan / Idul Fitri): spike kuat
    - Juni–Juli (musim dingin / ajaran baru): spike sedang
    - Oktober–November: naik lagi
    """
    m = d.month
    season = {
        1: 1.15, 2: 1.05, 3: 1.30, 4: 1.40,
        5: 1.10, 6: 1.20, 7: 1.25, 8: 1.00,
        9: 1.05, 10: 1.15, 11: 1.20, 12: 1.10,
    }
    return season.get(m, 1.0)


def generate_product_series(product: dict, dates: list[date]) -> pd.DataFrame:
    records = []
    stok = int(product["base_demand"] * 30)  # stok awal = satu bulan penuh
    n = len(dates)

    for i, d in enumerate(dates):
        # Tren linear kecil
        trend_factor = 1.0 + product["trend"] * (i / 365)

        # Faktor musiman
        sf = seasonal_factor(d)

        # Hari libur: permintaan sangat rendah
        holiday_factor = 0.05 if is_holiday(d) else 1.0

        # Pemakaian harian (unit)
        mean_demand = product["base_demand"] * trend_factor * sf * holiday_factor
        noise = np.random.normal(0, mean_demand * 0.12)
        pemakaian = max(0, int(mean_demand + noise))

        # Pengadaan: dilakukan setiap 7 hari jika stok < threshold
        threshold = product["base_demand"] * 14  # 2 minggu
        if i % 7 == 0 and stok < threshold:
            order_qty = int(product["base_demand"] * 30 + np.random.normal(0, product["base_demand"] * 2))
            pengadaan = max(0, order_qty)
        else:
            pengadaan = 0

        stok_awal = stok
        stok = max(0, stok_awal + pengadaan - pemakaian)

        records.append({
            "tanggal":       d.strftime("%Y-%m-%d"),
            "kode_produk":   product["kode"],
            "nama_produk":   product["nama"],
            "satuan":        product["satuan"],
            "stok_awal":     stok_awal,
            "pemakaian":     pemakaian,
            "pengadaan":     pengadaan,
            "stok_akhir":    stok,
            "bulan":         d.month,
            "tahun":         d.year,
            "hari_minggu":   d.weekday() + 1,       # 1=Senin … 7=Minggu
            "is_libur":      int(is_holiday(d)),
            "faktor_musim":  round(sf, 4),
        })

    return pd.DataFrame(records)


def main():
    start = date(2022, 1, 1)
    end   = date(2024, 12, 31)
    dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]

    print(f"Generating data: {start} → {end} ({len(dates)} hari, {len(PRODUCTS)} produk)")

    all_dfs = []
    for product in PRODUCTS:
        df = generate_product_series(product, dates)
        all_dfs.append(df)
        print(f"  ✓ {product['kode']} {product['nama']:30s} | rows={len(df)}")

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.sort_values(["tanggal", "kode_produk"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    out_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "stok_obat_kain.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    combined.to_csv(out_path, index=False)

    print(f"\n✅ Dataset saved → {os.path.abspath(out_path)}")
    print(f"   Shape: {combined.shape}")
    print(combined.head(4).to_string(index=False))


if __name__ == "__main__":
    main()
