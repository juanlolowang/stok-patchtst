"""
parse_excel.py
Parsing file 'DATA BAHAN JADI.xlsx' menjadi CSV flat bulanan
untuk pipeline PatchTST.

Format input:
  Setiap section bulanan:
    Row 0: "Laporan Bahan Jadi"
    Row 1: "JANUARI 2019"
    Row 2: Header: Produk | Customer | Stock Awal (drum/tong/lain2/jmlh kg) | Terima | Kirim | Pakai | Stock Akhir
    Row 3: Sub-header satuan
    Row 4+: Data produk

Output CSV:
  bulan_tahun, produk, stok_awal_kg, terima_kg, kirim_kg, stok_akhir_kg
"""

import os
import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Mapping bulan Indonesia → angka
# ---------------------------------------------------------------------------
BULAN_MAP = {
    "JANUARI": 1, "FEBRUARI": 2, "MARET": 3, "APRIL": 4,
    "MEI": 5, "JUNI": 6, "JULI": 7, "AGUSTUS": 8,
    "SEPTEMBER": 9, "OKTOBER": 10, "NOVEMBER": 11, "DESEMBER": 12,
}

# Kolom index dalam raw Excel (0-based)
COL_PRODUK        = 0
COL_CUSTOMER      = 1
COL_STOK_AWAL_KG  = 5   # jmlh kg kolom Stock Awal
COL_TERIMA_KG     = 9   # jmlh kg kolom Terima
COL_KIRIM_KG      = 13  # jmlh kg kolom Kirim
COL_STOK_AKHIR_KG = 19  # jmlh kg kolom Stock Akhir

# Minimum jumlah bulan aktif (stok_akhir > 0) agar produk dimasukkan
MIN_ACTIVE_MONTHS = 12


def detect_month_label(text: str):
    """
    Detect apakah text adalah label bulan, misal 'JANUARI 2019'.
    Returns (bulan_int, tahun_int) atau None.
    """
    if not isinstance(text, str):
        return None
    text = text.strip().upper()
    for nama, num in BULAN_MAP.items():
        pattern = rf"^{nama}\s+(\d{{4}})$"
        m = re.match(pattern, text)
        if m:
            return (num, int(m.group(1)))
    return None


def normalize_product_name(name: str) -> str:
    """Normalisasi nama produk: strip, lowercase-ish, collapse spaces."""
    if not isinstance(name, str):
        return ""
    name = name.strip()
    name = re.sub(r"\s+", " ", name)  # collapse multiple spaces
    return name


def safe_float(val, default=0.0) -> float:
    """Convert value ke float, handle NaN/None."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def parse_excel(xlsx_path: str, min_active_months: int = MIN_ACTIVE_MONTHS) -> pd.DataFrame:
    """
    Parse DATA BAHAN JADI.xlsx → DataFrame flat bulanan.
    
    Returns DataFrame dengan kolom:
        bulan_tahun (str, format 'YYYY-MM')
        produk (str)
        stok_awal_kg (float)
        terima_kg (float)
        kirim_kg (float)
        stok_akhir_kg (float)
    """
    print(f"📖 Membaca {xlsx_path} ...")
    df_raw = pd.read_excel(xlsx_path, sheet_name=0, header=None)
    total_rows = len(df_raw)
    print(f"   Total baris raw: {total_rows:,}")

    # -----------------------------------------------------------------------
    # Step 1: Deteksi posisi setiap section bulanan
    # -----------------------------------------------------------------------
    sections = []  # list of (row_idx, bulan, tahun)
    for i in range(total_rows):
        val = df_raw.iloc[i, 0]
        result = detect_month_label(str(val) if pd.notna(val) else "")
        if result:
            sections.append((i, result[0], result[1]))

    print(f"   Section bulanan ditemukan: {len(sections)}")

    # -----------------------------------------------------------------------
    # Step 2: Extract data dari setiap section
    # -----------------------------------------------------------------------
    records = []

    for sec_idx, (sec_row, bulan, tahun) in enumerate(sections):
        bulan_tahun = f"{tahun}-{bulan:02d}"
        
        # Data dimulai 3 baris setelah label bulan (skip header + sub-header)
        data_start = sec_row + 3
        
        # Data berakhir sebelum section berikutnya atau akhir file
        if sec_idx + 1 < len(sections):
            data_end = sections[sec_idx + 1][0] - 2  # skip "Laporan Bahan Jadi" row
        else:
            data_end = total_rows

        for i in range(data_start, data_end):
            produk_raw = df_raw.iloc[i, COL_PRODUK]
            if pd.isna(produk_raw):
                continue

            produk = normalize_product_name(str(produk_raw))
            if not produk or produk.upper() in ("PRODUK", "JUMLAH", "TOTAL", ""):
                continue

            # Extract values (jmlh kg columns)
            stok_awal  = safe_float(df_raw.iloc[i, COL_STOK_AWAL_KG] if df_raw.shape[1] > COL_STOK_AWAL_KG else None)
            terima     = safe_float(df_raw.iloc[i, COL_TERIMA_KG] if df_raw.shape[1] > COL_TERIMA_KG else None)
            kirim      = safe_float(df_raw.iloc[i, COL_KIRIM_KG] if df_raw.shape[1] > COL_KIRIM_KG else None)
            stok_akhir = safe_float(df_raw.iloc[i, COL_STOK_AKHIR_KG] if df_raw.shape[1] > COL_STOK_AKHIR_KG else None)

            records.append({
                "bulan_tahun":   bulan_tahun,
                "produk":        produk,
                "stok_awal_kg":  stok_awal,
                "terima_kg":     terima,
                "kirim_kg":      kirim,
                "stok_akhir_kg": stok_akhir,
            })

    df = pd.DataFrame(records)
    print(f"   Total baris (raw parse): {len(df):,}")

    # -----------------------------------------------------------------------
    # Step 3: Agregasi per produk per bulan (gabung customer yang sama)
    # -----------------------------------------------------------------------
    df_agg = df.groupby(["bulan_tahun", "produk"], as_index=False).agg({
        "stok_awal_kg":  "sum",
        "terima_kg":     "sum",
        "kirim_kg":      "sum",
        "stok_akhir_kg": "sum",
    })
    print(f"   Setelah agregasi per produk: {len(df_agg):,}")
    print(f"   Produk unik: {df_agg['produk'].nunique()}")

    # -----------------------------------------------------------------------
    # Step 4: Filter produk aktif
    # -----------------------------------------------------------------------
    if min_active_months > 0:
        activity = df_agg[df_agg["stok_akhir_kg"] > 0].groupby("produk").size()
        active_products = activity[activity >= min_active_months].index.tolist()
        df_agg = df_agg[df_agg["produk"].isin(active_products)].copy()
        print(f"   Produk aktif (≥{min_active_months} bulan non-zero): {len(active_products)}")
        print(f"   Baris setelah filter: {len(df_agg):,}")

    # Sort
    df_agg.sort_values(["produk", "bulan_tahun"], inplace=True)
    df_agg.reset_index(drop=True, inplace=True)

    return df_agg


def main():
    base = os.path.dirname(__file__)
    xlsx_path = os.path.join(base, "..", "data", "DATA BAHAN JADI.xlsx")
    out_dir = os.path.join(base, "..", "data", "processed")
    os.makedirs(out_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, "bahan_jadi_bulanan.csv")

    df = parse_excel(xlsx_path)

    df.to_csv(out_csv, index=False)
    print(f"\n✅ CSV disimpan → {os.path.abspath(out_csv)}")
    print(f"   Shape: {df.shape}")
    print(f"   Periode: {df['bulan_tahun'].min()} → {df['bulan_tahun'].max()}")
    print(f"   Produk: {df['produk'].nunique()}")
    print(f"\nSample data:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
