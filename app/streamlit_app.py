"""
streamlit_app.py
Dashboard web premium untuk Prediksi Kebutuhan Stok Bahan Jadi Obat Kain
menggunakan model PatchTST.
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Tambahkan src/ ke path
SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC_DIR)

RAW_CSV   = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "stok_obat_kain.csv")
MODELS_DIR= os.path.join(os.path.dirname(__file__), "..", "models")
OUT_DIR   = os.path.join(os.path.dirname(__file__), "..", "outputs")

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="StokAI — Prediksi Obat Kain",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — Dark Premium Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-dark:     #0d0d1a;
    --bg-card:     #13131f;
    --bg-card2:    #1a1a2e;
    --accent:      #7c5cfc;
    --accent2:     #4fc3f7;
    --accent3:     #ff8a65;
    --success:     #66bb6a;
    --warning:     #ffca28;
    --text:        #e8e8f0;
    --text-muted:  #8888aa;
    --border:      rgba(124,92,252,0.2);
}

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }
.stApp { background: var(--bg-dark); }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0f1e 0%, #161628 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--text); }

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-card2) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 32px rgba(124,92,252,0.15); }
.metric-val  { font-size: 2.2rem; font-weight: 700; color: var(--accent2); line-height: 1; }
.metric-label{ font-size: 0.82rem; color: var(--text-muted); margin-top: 6px; letter-spacing: 0.5px; }
.metric-unit { font-size: 0.75rem; color: var(--accent); margin-top: 2px; }

/* Section headers */
.section-title {
    font-size: 1.25rem; font-weight: 600; color: var(--text);
    border-left: 4px solid var(--accent); padding-left: 12px; margin: 16px 0;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #16013a 0%, #0a1628 50%, #001a0a 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero-title { font-size: 2rem; font-weight: 700; color: white; margin: 0; }
.hero-sub   { font-size: 0.95rem; color: #aab0cc; margin-top: 8px; }
.hero-badge {
    display: inline-block; background: rgba(124,92,252,0.2);
    border: 1px solid var(--accent); border-radius: 20px;
    padding: 4px 14px; font-size: 0.78rem; color: var(--accent);
    margin-top: 12px; font-weight: 500;
}

/* Tag chips */
.chip {
    display: inline-block; padding: 3px 10px; border-radius: 99px; font-size: 0.72rem;
    font-weight: 600; margin: 2px;
}
.chip-purple { background: rgba(124,92,252,0.15); color: var(--accent); border: 1px solid rgba(124,92,252,0.3); }
.chip-blue   { background: rgba(79,195,247,0.15); color: var(--accent2); border: 1px solid rgba(79,195,247,0.3); }
.chip-orange { background: rgba(255,138,101,0.15); color: var(--accent3); border: 1px solid rgba(255,138,101,0.3); }

/* Status badges */
.badge-ok    { background: rgba(102,187,106,0.15); color: var(--success); border:1px solid rgba(102,187,106,0.3); border-radius: 6px; padding: 2px 8px; font-size:0.75rem; }
.badge-warn  { background: rgba(255,202,40,0.15);  color: var(--warning);  border:1px solid rgba(255,202,40,0.3);  border-radius: 6px; padding: 2px 8px; font-size:0.75rem; }
.badge-err   { background: rgba(239,83,80,0.15);   color: #ef5350;         border:1px solid rgba(239,83,80,0.3);   border-radius: 6px; padding: 2px 8px; font-size:0.75rem; }

/* Streamlit overrides */
div[data-testid="stMetric"]           { background: var(--bg-card2); border-radius: 12px; padding: 12px 16px; border: 1px solid var(--border); }
div[data-testid="stMetric"] label     { color: var(--text-muted) !important; font-size: 0.8rem; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--accent2) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#13131f",
    plot_bgcolor="#0d0d1a",
    font=dict(family="Inter", color="#e8e8f0"),
    margin=dict(t=40, b=30, l=40, r=20),
)

PRODUCT_COLORS = [
    "#7c5cfc","#4fc3f7","#ff8a65","#66bb6a",
    "#ffca28","#f06292","#80cbc4","#bcaaa4",
]

@st.cache_data
def load_raw_data():
    if not os.path.exists(RAW_CSV):
        return None
    df = pd.read_csv(RAW_CSV, parse_dates=["tanggal"])
    df.sort_values(["kode_produk","tanggal"], inplace=True)
    return df

@st.cache_resource
def load_model_resources():
    try:
        import torch
        import joblib
        from predict import load_model_and_scalers
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, scalers, cfg = load_model_and_scalers(device)
        return model, scalers, cfg, device
    except Exception as e:
        return None, None, None, str(e)


def fmt_number(n):
    return f"{n:,.0f}"

def status_badge(stok, threshold):
    if stok > threshold * 1.2:
        return '<span class="badge-ok">🟢 Aman</span>'
    elif stok > threshold * 0.5:
        return '<span class="badge-warn">🟡 Perlu Pantau</span>'
    else:
        return '<span class="badge-err">🔴 Kritis</span>'


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar Navigation
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 16px'>
        <div style='font-size:2.5rem'>🏥</div>
        <div style='font-size:1.1rem; font-weight:700; color:white;'>StokAI</div>
        <div style='font-size:0.75rem; color:#8888aa;'>Prediksi Stok Obat Kain</div>
        <div style='margin-top:8px' class='chip chip-purple'>PatchTST · v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigasi",
        ["🏠  Beranda", "📊  Analisis EDA", "🔮  Prediksi", "📈  Evaluasi Model"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    df_all = load_raw_data()
    if df_all is not None:
        products = df_all["kode_produk"].unique().tolist()
        prod_names = df_all.drop_duplicates("kode_produk")[["kode_produk","nama_produk"]].set_index("kode_produk")["nama_produk"].to_dict()
    else:
        products = []
        prod_names = {}

    st.markdown('<div style="color:#8888aa; font-size:0.75rem;">Status Data</div>', unsafe_allow_html=True)
    if df_all is not None:
        st.markdown(f'<span class="badge-ok">✓ {len(df_all):,} baris dimuat</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-err">✗ Data belum ada</span><br><span style="font-size:0.7rem;color:#888">Jalankan generate_data.py</span>', unsafe_allow_html=True)

    model_exists = os.path.exists(os.path.join(MODELS_DIR, "best_model.pt"))
    st.markdown('<div style="color:#8888aa; font-size:0.75rem; margin-top:8px">Status Model</div>', unsafe_allow_html=True)
    if model_exists:
        st.markdown('<span class="badge-ok">✓ Model tersedia</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-warn">⚠ Belum ditraining</span>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — Beranda
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠  Beranda":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🏥 Prediksi Stok Bahan Jadi Obat Kain</div>
        <div class="hero-sub">Sistem cerdas berbasis <strong>PatchTST</strong> — Patch Time-Series Transformer untuk perencanaan stok bahan medis tekstil secara akurat.</div>
        <div>
            <span class="chip chip-purple">PatchTST</span>
            <span class="chip chip-blue">Deep Learning</span>
            <span class="chip chip-orange">Time Series</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df_all is None:
        st.warning("⚠ Data belum ada. Jalankan `python src/generate_data.py` terlebih dahulu.")
    else:
        # KPI Row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_jenis = df_all["kode_produk"].nunique()
            st.markdown(f'<div class="metric-card"><div class="metric-val">{total_jenis}</div><div class="metric-label">Jenis Produk</div><div class="metric-unit">kode unik</div></div>', unsafe_allow_html=True)
        with col2:
            hari = df_all["tanggal"].nunique()
            st.markdown(f'<div class="metric-card"><div class="metric-val">{hari:,}</div><div class="metric-label">Total Hari Data</div><div class="metric-unit">{df_all["tanggal"].min().strftime("%b %Y")} – {df_all["tanggal"].max().strftime("%b %Y")}</div></div>', unsafe_allow_html=True)
        with col3:
            total_pakai = int(df_all["pemakaian"].sum())
            st.markdown(f'<div class="metric-card"><div class="metric-val">{total_pakai/1e6:.2f}M</div><div class="metric-label">Total Pemakaian</div><div class="metric-unit">unit (semua produk)</div></div>', unsafe_allow_html=True)
        with col4:
            avg_stok = int(df_all["stok_akhir"].mean())
            st.markdown(f'<div class="metric-card"><div class="metric-val">{avg_stok:,}</div><div class="metric-label">Rata-rata Stok Akhir</div><div class="metric-unit">unit/hari</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Stok Terkini per Produk
        st.markdown('<div class="section-title">📦 Status Stok Terkini</div>', unsafe_allow_html=True)
        latest = df_all.sort_values("tanggal").groupby("kode_produk").last().reset_index()

        status_rows = []
        for _, row in latest.iterrows():
            mean_pakai = df_all[df_all["kode_produk"] == row["kode_produk"]]["pemakaian"].mean()
            threshold  = mean_pakai * 14  # 2 minggu buffer
            pct_safe   = min(100, row["stok_akhir"] / threshold * 100)
            badge_html = status_badge(row["stok_akhir"], threshold)
            status_rows.append({
                "Kode":     row["kode_produk"],
                "Produk":   row["nama_produk"],
                "Stok Akhir": f"{row['stok_akhir']:,.0f} {row['satuan']}",
                "Pemakaian/hari": f"{mean_pakai:.0f}",
                "Cukup (hari)": f"{int(row['stok_akhir']/(mean_pakai+1))}",
                "Status": badge_html,
            })

        df_status = pd.DataFrame(status_rows)
        st.markdown(df_status.to_html(escape=False, index=False), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Mini chart: tren pemakaian total
        st.markdown('<div class="section-title">📉 Tren Pemakaian Harian (Semua Produk)</div>', unsafe_allow_html=True)
        daily_total = df_all.groupby("tanggal")["pemakaian"].sum().reset_index()
        daily_total["MA7"] = daily_total["pemakaian"].rolling(7).mean()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily_total["tanggal"], y=daily_total["pemakaian"],
                                  mode="lines", name="Pemakaian Harian",
                                  line=dict(color="#7c5cfc", width=1), opacity=0.5))
        fig.add_trace(go.Scatter(x=daily_total["tanggal"], y=daily_total["MA7"],
                                  mode="lines", name="MA 7-hari",
                                  line=dict(color="#4fc3f7", width=2)))
        fig.update_layout(**PLOTLY_LAYOUT, title="Total Pemakaian Harian (Semua Produk)")
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — EDA
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊  Analisis EDA":
    st.markdown('<div class="section-title">📊 Analisis Eksplorasi Data (EDA)</div>', unsafe_allow_html=True)

    if df_all is None:
        st.warning("Data belum ada. Jalankan `python src/generate_data.py`.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Tren Stok", "🗓 Pola Musiman", "📦 Distribusi", "🔄 Pemakaian vs Pengadaan"])

    with tab1:
        selected_prods = st.multiselect("Pilih Produk:", products,
                                         default=products[:3],
                                         format_func=lambda x: f"{x} — {prod_names.get(x,'')}")
        if selected_prods:
            fig = go.Figure()
            for i, kode in enumerate(selected_prods):
                sub = df_all[df_all["kode_produk"] == kode]
                sub_w = sub.set_index("tanggal")["stok_akhir"].resample("W").mean().reset_index()
                fig.add_trace(go.Scatter(x=sub_w["tanggal"], y=sub_w["stok_akhir"],
                                          mode="lines", name=f"{kode}",
                                          line=dict(color=PRODUCT_COLORS[i % len(PRODUCT_COLORS)], width=2)))
            fig.update_layout(**PLOTLY_LAYOUT, title="Tren Stok Akhir (rata-rata mingguan)")
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        kode_sel = st.selectbox("Pilih Produk:", products, format_func=lambda x: f"{x} — {prod_names.get(x,'')}")
        sub = df_all[df_all["kode_produk"] == kode_sel]

        col1, col2 = st.columns(2)
        with col1:
            monthly = sub.groupby("bulan")["pemakaian"].mean().reset_index()
            bulan_label = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"]
            monthly["bulan_str"] = monthly["bulan"].apply(lambda x: bulan_label[x-1])
            fig2 = px.bar(monthly, x="bulan_str", y="pemakaian",
                          color="pemakaian", color_continuous_scale="Purples",
                          title="Rata-rata Pemakaian per Bulan")
            fig2.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            dow = sub.groupby("hari_minggu")["pemakaian"].mean().reset_index()
            hari_label = ["Sen","Sel","Rab","Kam","Jum","Sab","Min"]
            dow["hari_str"] = dow["hari_minggu"].apply(lambda x: hari_label[x-1])
            fig3 = px.bar(dow, x="hari_str", y="pemakaian",
                          color="pemakaian", color_continuous_scale="Blues",
                          title="Rata-rata Pemakaian per Hari")
            fig3.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        fig4 = px.box(df_all, x="kode_produk", y="pemakaian",
                       color="kode_produk", color_discrete_sequence=PRODUCT_COLORS,
                       title="Distribusi Pemakaian Harian per Produk")
        fig4.update_layout(**PLOTLY_LAYOUT, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    with tab4:
        kode_sel2 = st.selectbox("Produk:", products, key="eda4", format_func=lambda x: f"{x} — {prod_names.get(x,'')}")
        sub2 = df_all[df_all["kode_produk"] == kode_sel2].copy()
        sub2_m = sub2.set_index("tanggal")[["pemakaian","pengadaan"]].resample("W").sum().reset_index()

        fig5 = make_subplots(specs=[[{"secondary_y": True}]])
        fig5.add_trace(go.Bar(x=sub2_m["tanggal"], y=sub2_m["pemakaian"],
                               name="Pemakaian", marker_color="#7c5cfc", opacity=0.8), secondary_y=False)
        fig5.add_trace(go.Scatter(x=sub2_m["tanggal"], y=sub2_m["pengadaan"],
                                   name="Pengadaan", line=dict(color="#ff8a65", width=2)), secondary_y=True)
        fig5.update_layout(**PLOTLY_LAYOUT, title=f"Pemakaian vs Pengadaan — {kode_sel2}")
        st.plotly_chart(fig5, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — Prediksi
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔮  Prediksi":
    st.markdown('<div class="section-title">🔮 Prediksi Kebutuhan Stok</div>', unsafe_allow_html=True)

    if df_all is None:
        st.warning("Data belum ada."); st.stop()
    if not model_exists:
        st.error("❌ Model belum ditraining. Jalankan `python src/train.py` terlebih dahulu.")
        st.stop()

    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.markdown("**Pengaturan Prediksi**")
        sel_kode = st.selectbox("Produk:", products, format_func=lambda x: f"{x} — {prod_names.get(x,'')}")
        horizon  = st.slider("Horizon prediksi (hari):", 7, 30, 14)
        run_btn  = st.button("🚀 Jalankan Prediksi", use_container_width=True)

    with col_r:
        if run_btn:
            with st.spinner("Memproses prediksi..."):
                try:
                    import torch
                    from predict import load_model_and_scalers, predict_product
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model, scalers, cfg = load_model_and_scalers(device)

                    # Override pred_len jika horizon berbeda (re-run head secara manual tidak dimungkinkan tanpa retrain)
                    result = predict_product(sel_kode, model, scalers, cfg, device, df_all)

                    preds = result["prediksi_stok"][:horizon]
                    dates = result["tanggal_prediksi"][:horizon]
                    hist  = result["last_history"]

                    # Plot
                    fig = go.Figure()

                    # History
                    hist_dates = pd.date_range(
                        end=pd.to_datetime(dates[0]) - pd.Timedelta(days=1),
                        periods=len(hist), freq="D"
                    )
                    fig.add_trace(go.Scatter(
                        x=list(hist_dates.strftime("%Y-%m-%d")), y=hist,
                        mode="lines", name="Historis (60 hari)",
                        line=dict(color="#4fc3f7", width=2)
                    ))

                    # Prediksi + confidence band
                    fig.add_trace(go.Scatter(
                        x=dates, y=[p * 1.1 for p in preds],
                        mode="lines", name="Upper Band", line=dict(width=0),
                        showlegend=False
                    ))
                    fig.add_trace(go.Scatter(
                        x=dates, y=[p * 0.9 for p in preds],
                        fill="tonexty", mode="lines", name="Confidence Band",
                        line=dict(width=0), fillcolor="rgba(124,92,252,0.15)"
                    ))
                    fig.add_trace(go.Scatter(
                        x=dates, y=preds,
                        mode="lines+markers", name="Prediksi",
                        line=dict(color="#ff8a65", width=2.5),
                        marker=dict(size=6, color="#ff8a65")
                    ))

                    # Vertical divider
                    fig.add_vline(x=dates[0], line_dash="dash", line_color="#888888", opacity=0.6)
                    fig.add_annotation(x=dates[0], y=max(hist), text="Mulai Prediksi",
                                       showarrow=False, font=dict(color="#888888", size=11), yshift=15)

                    fig.update_layout(**PLOTLY_LAYOUT,
                                       title=f"Prediksi Stok — {sel_kode} ({prod_names.get(sel_kode, '')})",
                                       yaxis_title="Stok (unit)")
                    st.plotly_chart(fig, use_container_width=True)

                    # Tabel Prediksi
                    st.markdown("**📋 Tabel Prediksi**")
                    df_pred = pd.DataFrame({
                        "Tanggal":    dates,
                        "Prediksi Stok": [fmt_number(p) for p in preds],
                        "Min (−10%)": [fmt_number(p * 0.9) for p in preds],
                        "Max (+10%)": [fmt_number(p * 1.1) for p in preds],
                    })
                    st.dataframe(df_pred, use_container_width=True, hide_index=True)

                    # Rekomendasi
                    avg_pred = np.mean(preds)
                    if avg_pred < np.mean(hist) * 0.7:
                        st.error("🔴 **Peringatan!** Stok diperkirakan turun signifikan. Segera rencanakan pengadaan.")
                    elif avg_pred < np.mean(hist) * 0.9:
                        st.warning("🟡 Stok cenderung menurun. Pantau kondisi dan siapkan order pengadaan.")
                    else:
                        st.success("🟢 Stok diperkirakan aman dalam rentang prediksi.")

                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.info("Pastikan model sudah ditraining dengan menjalankan `python src/train.py`.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — Evaluasi
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📈  Evaluasi Model":
    st.markdown('<div class="section-title">📈 Evaluasi Model PatchTST</div>', unsafe_allow_html=True)

    metrics_path = os.path.join(OUT_DIR, "metrics.json")
    eval_img     = os.path.join(OUT_DIR, "evaluation.png")
    hist_path    = os.path.join(MODELS_DIR, "history.json")

    if not model_exists:
        st.warning("Model belum ditraining. Jalankan `python src/train.py` lalu `python src/evaluate.py`.")
        st.stop()

    # Metrics
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{metrics["MAE"]:.4f}</div><div class="metric-label">MAE</div><div class="metric-unit">Mean Absolute Error (norm.)</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{metrics["RMSE"]:.4f}</div><div class="metric-label">RMSE</div><div class="metric-unit">Root Mean Squared Error (norm.)</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{metrics["MAPE"]:.2f}%</div><div class="metric-label">MAPE</div><div class="metric-unit">Mean Absolute Percentage Error</div></div>', unsafe_allow_html=True)
    else:
        st.info("Metrik belum tersedia. Jalankan `python src/evaluate.py`.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Learning Curve (Plotly)
    if os.path.exists(hist_path):
        st.markdown('<div class="section-title">📉 Learning Curve</div>', unsafe_allow_html=True)
        with open(hist_path) as f:
            history = json.load(f)
        epochs = list(range(1, len(history["train_loss"]) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], mode="lines",
                                  name="Train Loss", line=dict(color="#81c784", width=2)))
        fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], mode="lines",
                                  name="Val Loss", line=dict(color="#e57373", width=2)))
        fig.update_layout(**PLOTLY_LAYOUT, title="Training vs Validation Loss",
                           xaxis_title="Epoch", yaxis_title="MSE Loss")
        st.plotly_chart(fig, use_container_width=True)

    # Static eval plot
    if os.path.exists(eval_img):
        st.markdown('<div class="section-title">🖼 Plot Prediksi vs Aktual</div>', unsafe_allow_html=True)
        st.image(eval_img, use_container_width=True)
    else:
        st.info("Plot evaluasi belum ada. Jalankan `python src/evaluate.py`.")

    # Model info
    ckpt_path = os.path.join(MODELS_DIR, "best_model.pt")
    if os.path.exists(ckpt_path):
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg  = ckpt.get("config", {})
        with st.expander("ℹ️ Detail Konfigurasi Model"):
            st.json({
                "Epoch terbaik":  ckpt.get("epoch", "-"),
                "Val Loss terbaik": f"{ckpt.get('val_loss', '-'):.6f}",
                "seq_len":   cfg.get("seq_len"),
                "pred_len":  cfg.get("pred_len"),
                "patch_len": cfg.get("patch_len"),
                "stride":    cfg.get("stride"),
                "d_model":   cfg.get("d_model"),
                "n_heads":   cfg.get("n_heads"),
                "n_layers":  cfg.get("n_layers"),
                "dropout":   cfg.get("dropout"),
            })
