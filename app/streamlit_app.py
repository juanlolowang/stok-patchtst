"""
streamlit_app.py
Dashboard web premium untuk Prediksi Stok Bahan Jadi menggunakan model PatchTST.
"""

import os, sys, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC_DIR)

CSV_PATH   = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "bahan_jadi_bulanan.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "..", "outputs")

st.set_page_config(page_title="StokAI — Prediksi Bahan Jadi", page_icon="🏭", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
:root {
    --bg-dark:#0d0d1a;--bg-card:#13131f;--bg-card2:#1a1a2e;
    --accent:#7c5cfc;--accent2:#4fc3f7;--accent3:#ff8a65;
    --success:#66bb6a;--warning:#ffca28;--text:#e8e8f0;--text-muted:#8888aa;
    --border:rgba(124,92,252,0.2);
}
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:var(--bg-dark);}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0f0f1e 0%,#161628 100%);border-right:1px solid var(--border);}
[data-testid="stSidebar"] .stSelectbox label,[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] p,[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:var(--text);}
.metric-card{background:linear-gradient(135deg,var(--bg-card) 0%,var(--bg-card2) 100%);border:1px solid var(--border);border-radius:16px;padding:20px 24px;text-align:center;transition:transform 0.2s,box-shadow 0.2s;}
.metric-card:hover{transform:translateY(-3px);box-shadow:0 8px 32px rgba(124,92,252,0.15);}
.metric-val{font-size:2.2rem;font-weight:700;color:var(--accent2);line-height:1;}
.metric-label{font-size:0.82rem;color:var(--text-muted);margin-top:6px;letter-spacing:0.5px;}
.metric-unit{font-size:0.75rem;color:var(--accent);margin-top:2px;}
.section-title{font-size:1.25rem;font-weight:600;color:var(--text);border-left:4px solid var(--accent);padding-left:12px;margin:16px 0;}
.hero-banner{background:linear-gradient(135deg,#16013a 0%,#0a1628 50%,#001a0a 100%);border:1px solid var(--border);border-radius:20px;padding:36px 40px;margin-bottom:24px;overflow:hidden;}
.hero-title{font-size:2rem;font-weight:700;color:white;margin:0;}
.hero-sub{font-size:0.95rem;color:#aab0cc;margin-top:8px;}
.chip{display:inline-block;padding:3px 10px;border-radius:99px;font-size:0.72rem;font-weight:600;margin:2px;}
.chip-purple{background:rgba(124,92,252,0.15);color:var(--accent);border:1px solid rgba(124,92,252,0.3);}
.chip-blue{background:rgba(79,195,247,0.15);color:var(--accent2);border:1px solid rgba(79,195,247,0.3);}
.chip-orange{background:rgba(255,138,101,0.15);color:var(--accent3);border:1px solid rgba(255,138,101,0.3);}
.badge-ok{background:rgba(102,187,106,0.15);color:var(--success);border:1px solid rgba(102,187,106,0.3);border-radius:6px;padding:2px 8px;font-size:0.75rem;}
.badge-warn{background:rgba(255,202,40,0.15);color:var(--warning);border:1px solid rgba(255,202,40,0.3);border-radius:6px;padding:2px 8px;font-size:0.75rem;}
.badge-err{background:rgba(239,83,80,0.15);color:#ef5350;border:1px solid rgba(239,83,80,0.3);border-radius:6px;padding:2px 8px;font-size:0.75rem;}
div[data-testid="stMetric"]{background:var(--bg-card2);border-radius:12px;padding:12px 16px;border:1px solid var(--border);}
div[data-testid="stMetric"] label{color:var(--text-muted)!important;font-size:0.8rem;}
div[data-testid="stMetric"] [data-testid="stMetricValue"]{color:var(--accent2)!important;}
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(template="plotly_dark", paper_bgcolor="#13131f", plot_bgcolor="#0d0d1a",
                     font=dict(family="Inter", color="#e8e8f0"), margin=dict(t=40, b=30, l=40, r=20))
PRODUCT_COLORS = ["#7c5cfc","#4fc3f7","#ff8a65","#66bb6a","#ffca28","#f06292","#80cbc4","#bcaaa4",
                  "#ce93d8","#a5d6a7","#ef9a9a","#90caf9","#fff59d","#b0bec5","#ffab91","#81d4fa"]

@st.cache_data
def load_raw_data():
    if not os.path.exists(CSV_PATH):
        return None
    df = pd.read_csv(CSV_PATH)
    df.sort_values(["produk","bulan_tahun"], inplace=True)
    return df

@st.cache_resource
def load_model_resources():
    try:
        import torch, joblib
        from predict import load_model_and_scalers
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, scalers, cfg = load_model_and_scalers(device)
        return model, scalers, cfg, device
    except Exception as e:
        return None, None, None, str(e)

def fmt_number(n):
    return f"{n:,.1f}"

def status_badge(stok, threshold):
    if stok > threshold * 1.2:
        return '<span class="badge-ok">🟢 Aman</span>'
    elif stok > threshold * 0.5:
        return '<span class="badge-warn">🟡 Perlu Pantau</span>'
    else:
        return '<span class="badge-err">🔴 Kritis</span>'

# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 16px'>
        <div style='font-size:2.5rem'>🏭</div>
        <div style='font-size:1.1rem; font-weight:700; color:white;'>StokAI</div>
        <div style='font-size:0.75rem; color:#8888aa;'>Bahan Jadi Obat Kain</div>
        <div style='font-size:0.85rem; font-weight:600; color:var(--accent2); margin-top:4px;'>PT. Seikyo Indochem</div>
        <div style='margin-top:8px' class='chip chip-purple'>PatchTST · v2.1</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigasi", ["🏠  Beranda", "📊  Analisis EDA", "🔮  Prediksi", "📈  Evaluasi Model"], label_visibility="collapsed")
    st.markdown("---")

    df_all = load_raw_data()
    if df_all is not None:
        products = sorted(df_all["produk"].unique().tolist())
    else:
        products = []

    st.markdown('<div style="color:#8888aa; font-size:0.75rem;">Status Data</div>', unsafe_allow_html=True)
    if df_all is not None:
        st.markdown(f'<span class="badge-ok">✓ {len(df_all):,} baris · {len(products)} produk</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-err">✗ Data belum ada</span><br><span style="font-size:0.7rem;color:#888">Jalankan parse_excel.py</span>', unsafe_allow_html=True)

    model_exists = os.path.exists(os.path.join(MODELS_DIR, "best_model.pt"))
    st.markdown('<div style="color:#8888aa; font-size:0.75rem; margin-top:8px">Status Model</div>', unsafe_allow_html=True)
    if model_exists:
        st.markdown('<span class="badge-ok">✓ Model tersedia</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-warn">⚠ Belum ditraining</span>', unsafe_allow_html=True)

# ── PAGE 1: Beranda ──
if page == "🏠  Beranda":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">🏭 Prediksi Stok Bahan Jadi Obat Kain</div>
        <div class="hero-sub">Sistem cerdas berbasis <strong>PatchTST</strong> untuk optimasi inventaris di <strong>PT. Seikyo Indochem</strong>.</div>
        <div>
            <span class="chip chip-purple">PatchTST</span>
            <span class="chip chip-blue">Deep Learning</span>
            <span class="chip chip-orange">Time Series</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df_all is None:
        st.warning("⚠ Data belum ada. Jalankan `python src/parse_excel.py` terlebih dahulu.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        n_produk = df_all["produk"].nunique()
        n_bulan = df_all["bulan_tahun"].nunique()
        total_kirim = df_all["kirim_kg"].sum()
        avg_stok = df_all["stok_akhir_kg"].mean()
        periode_min = df_all["bulan_tahun"].min()
        periode_max = df_all["bulan_tahun"].max()

        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{n_produk}</div><div class="metric-label">Jenis Produk</div><div class="metric-unit">produk aktif</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{n_bulan}</div><div class="metric-label">Total Bulan Data</div><div class="metric-unit">{periode_min} – {periode_max}</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{total_kirim/1e3:.1f}T</div><div class="metric-label">Total Kirim</div><div class="metric-unit">ton (semua produk)</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{avg_stok:,.0f}</div><div class="metric-label">Rata-rata Stok Akhir</div><div class="metric-unit">kg/bulan</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">📦 Status Stok Terkini (Bulan Terakhir)</div>', unsafe_allow_html=True)

        latest = df_all.sort_values("bulan_tahun").groupby("produk").last().reset_index()
        top_latest = latest.nlargest(20, "stok_akhir_kg")

        status_rows = []
        for _, row in top_latest.iterrows():
            mean_kirim = df_all[df_all["produk"] == row["produk"]]["kirim_kg"].mean()
            threshold = mean_kirim * 3
            badge_html = status_badge(row["stok_akhir_kg"], threshold) if threshold > 0 else '<span class="badge-ok">🟢 Aman</span>'
            cukup = f"{int(row['stok_akhir_kg']/(mean_kirim+0.01))}" if mean_kirim > 0 else "∞"
            status_rows.append({
                "Produk": row["produk"],
                "Stok Akhir (kg)": f"{row['stok_akhir_kg']:,.1f}",
                "Kirim/bulan (avg)": f"{mean_kirim:,.1f}",
                "Cukup (bulan)": cukup,
                "Status": badge_html,
            })
        df_status = pd.DataFrame(status_rows)
        st.markdown(df_status.to_html(escape=False, index=False), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">📉 Tren Total Kirim Bulanan</div>', unsafe_allow_html=True)
        monthly_total = df_all.groupby("bulan_tahun")["kirim_kg"].sum().reset_index()
        monthly_total = monthly_total.sort_values("bulan_tahun")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly_total["bulan_tahun"], y=monthly_total["kirim_kg"],
                                  mode="lines+markers", name="Total Kirim (kg)",
                                  line=dict(color="#7c5cfc", width=2), marker=dict(size=4)))
        fig.update_layout(**PLOTLY_LAYOUT, title="Total Kirim Bulanan (Semua Produk)", yaxis_title="kg")
        st.plotly_chart(fig, width="stretch")

# ── PAGE 2: EDA ──
elif page == "📊  Analisis EDA":
    st.markdown('<div class="section-title">📊 Analisis Eksplorasi Data (EDA)</div>', unsafe_allow_html=True)
    if df_all is None:
        st.warning("Data belum ada. Jalankan `python src/parse_excel.py`."); st.stop()

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Tren Stok", "🗓 Pola Musiman", "📦 Distribusi", "🔄 Terima vs Kirim"])

    with tab1:
        selected_prods = st.multiselect("Pilih Produk:", products, default=products[:3])
        if selected_prods:
            fig = go.Figure()
            for i, produk in enumerate(selected_prods):
                sub = df_all[df_all["produk"] == produk].sort_values("bulan_tahun")
                fig.add_trace(go.Scatter(x=sub["bulan_tahun"], y=sub["stok_akhir_kg"],
                                          mode="lines+markers", name=produk,
                                          line=dict(color=PRODUCT_COLORS[i % len(PRODUCT_COLORS)], width=2),
                                          marker=dict(size=4)))
            fig.update_layout(**PLOTLY_LAYOUT, title="Tren Stok Akhir Bulanan (kg)")
            st.plotly_chart(fig, width="stretch")

    with tab2:
        kode_sel = st.selectbox("Pilih Produk:", products, key="eda2")
        sub = df_all[df_all["produk"] == kode_sel].copy()
        sub["bulan"] = sub["bulan_tahun"].str[5:7].astype(int)
        bulan_label = ["Jan","Feb","Mar","Apr","Mei","Jun","Jul","Agu","Sep","Okt","Nov","Des"]

        col1, col2 = st.columns(2)
        with col1:
            monthly = sub.groupby("bulan")["kirim_kg"].mean().reset_index()
            monthly["bulan_str"] = monthly["bulan"].apply(lambda x: bulan_label[x-1])
            fig2 = px.bar(monthly, x="bulan_str", y="kirim_kg", color="kirim_kg", color_continuous_scale="Purples", title="Rata-rata Kirim per Bulan (kg)")
            fig2.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig2, width="stretch")
        with col2:
            monthly2 = sub.groupby("bulan")["terima_kg"].mean().reset_index()
            monthly2["bulan_str"] = monthly2["bulan"].apply(lambda x: bulan_label[x-1])
            fig3 = px.bar(monthly2, x="bulan_str", y="terima_kg", color="terima_kg", color_continuous_scale="Blues", title="Rata-rata Terima per Bulan (kg)")
            fig3.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig3, width="stretch")

    with tab3:
        top_prods = df_all.groupby("produk")["stok_akhir_kg"].mean().nlargest(15).index.tolist()
        df_top = df_all[df_all["produk"].isin(top_prods)]
        fig4 = px.box(df_top, x="produk", y="stok_akhir_kg", color="produk", color_discrete_sequence=PRODUCT_COLORS, title="Distribusi Stok Akhir (Top 15 Produk)")
        fig4.update_layout(**PLOTLY_LAYOUT, showlegend=False, xaxis_tickangle=-45)
        st.plotly_chart(fig4, width="stretch")

    with tab4:
        kode_sel2 = st.selectbox("Produk:", products, key="eda4")
        sub2 = df_all[df_all["produk"] == kode_sel2].sort_values("bulan_tahun")
        fig5 = make_subplots(specs=[[{"secondary_y": True}]])
        fig5.add_trace(go.Bar(x=sub2["bulan_tahun"], y=sub2["kirim_kg"], name="Kirim (kg)", marker_color="#7c5cfc", opacity=0.8), secondary_y=False)
        fig5.add_trace(go.Scatter(x=sub2["bulan_tahun"], y=sub2["terima_kg"], name="Terima (kg)", line=dict(color="#ff8a65", width=2)), secondary_y=True)
        fig5.update_layout(**PLOTLY_LAYOUT, title=f"Terima vs Kirim — {kode_sel2}")
        st.plotly_chart(fig5, width="stretch")

# ── PAGE 3: Prediksi ──
elif page == "🔮  Prediksi":
    st.markdown('<div class="section-title">🔮 Prediksi Kebutuhan Stok</div>', unsafe_allow_html=True)
    if df_all is None:
        st.warning("Data belum ada."); st.stop()
    if not model_exists:
        st.error("❌ Model belum ditraining. Jalankan `python src/train.py` terlebih dahulu."); st.stop()

    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.markdown("**Pengaturan Prediksi**")
        sel_produk = st.selectbox("Produk:", products)
        run_btn = st.button("🚀 Jalankan Prediksi", width="stretch")

    with col_r:
        if run_btn:
            with st.spinner("Memproses prediksi..."):
                try:
                    import torch
                    from predict import load_model_and_scalers, predict_product
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model, scalers, cfg = load_model_and_scalers(device)
                    result = predict_product(sel_produk, model, scalers, cfg, device, df_all)

                    preds = result["prediksi_stok"]
                    dates = result["bulan_prediksi"]
                    hist = result["last_history"]
                    hist_months = result["history_months"]

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=hist_months, y=hist, mode="lines+markers", name=f"Historis ({len(hist)} bulan)",
                                              line=dict(color="#4fc3f7", width=2), marker=dict(size=5)))
                    fig.add_trace(go.Scatter(x=dates, y=[p*1.1 for p in preds], mode="lines", name="Upper", line=dict(width=0), showlegend=False))
                    fig.add_trace(go.Scatter(x=dates, y=[p*0.9 for p in preds], fill="tonexty", mode="lines", name="Confidence Band",
                                              line=dict(width=0), fillcolor="rgba(124,92,252,0.15)"))
                    fig.add_trace(go.Scatter(x=dates, y=preds, mode="lines+markers", name="Prediksi",
                                              line=dict(color="#ff8a65", width=2.5), marker=dict(size=6, color="#ff8a65")))
                    fig.add_vline(x=dates[0], line_dash="dash", line_color="#888888", opacity=0.6)
                    fig.update_layout(**PLOTLY_LAYOUT, title=f"Prediksi Stok — {sel_produk}", yaxis_title="Stok (kg)")
                    st.plotly_chart(fig, width="stretch")

                    st.markdown("**📋 Tabel Prediksi**")
                    df_pred = pd.DataFrame({"Bulan": dates, "Prediksi (kg)": [fmt_number(p) for p in preds],
                                            "Min (−10%)": [fmt_number(p*0.9) for p in preds], "Max (+10%)": [fmt_number(p*1.1) for p in preds]})
                    st.dataframe(df_pred, width="stretch", hide_index=True)

                    avg_pred = np.mean(preds)
                    if avg_pred < np.mean(hist) * 0.7:
                        st.error("🔴 **Peringatan!** Stok diperkirakan turun signifikan. Segera rencanakan pengadaan.")
                    elif avg_pred < np.mean(hist) * 0.9:
                        st.warning("🟡 Stok cenderung menurun. Pantau kondisi dan siapkan order.")
                    else:
                        st.success("🟢 Stok diperkirakan aman dalam rentang prediksi.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.info("Pastikan model sudah ditraining dengan menjalankan `python src/train.py`.")

# ── PAGE 4: Evaluasi ──
elif page == "📈  Evaluasi Model":
    st.markdown('<div class="section-title">📈 Evaluasi Model PatchTST</div>', unsafe_allow_html=True)
    metrics_path = os.path.join(OUT_DIR, "metrics.json")
    hist_path = os.path.join(MODELS_DIR, "history.json")

    if not model_exists:
        st.warning("Model belum ditraining."); st.stop()

    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{metrics["MSE"]:.6f}</div><div class="metric-label">MSE</div><div class="metric-unit">Mean Squared Error</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{metrics["MAE"]:.6f}</div><div class="metric-label">MAE</div><div class="metric-unit">Mean Absolute Error</div></div>', unsafe_allow_html=True)
    else:
        st.info("Metrik belum tersedia. Jalankan `python src/evaluate.py`.")

    st.markdown("<br>", unsafe_allow_html=True)

    if os.path.exists(hist_path):
        st.markdown('<div class="section-title">📉 Learning Curve</div>', unsafe_allow_html=True)
        with open(hist_path) as f:
            history = json.load(f)
        epochs = list(range(1, len(history["train_loss"]) + 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], mode="lines", name="Train Loss", line=dict(color="#81c784", width=2)))
        fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], mode="lines", name="Val Loss", line=dict(color="#e57373", width=2)))
        fig.update_layout(**PLOTLY_LAYOUT, title="Training vs Validation Loss", xaxis_title="Epoch", yaxis_title="MSE Loss")
        st.plotly_chart(fig, width="stretch")

    eval_img = os.path.join(OUT_DIR, "evaluation.png")
    if os.path.exists(eval_img):
        st.markdown('<div class="section-title">🖼 Plot Prediksi vs Aktual</div>', unsafe_allow_html=True)
        st.image(eval_img, width="stretch")

    ckpt_path = os.path.join(MODELS_DIR, "best_model.pt")
    if os.path.exists(ckpt_path):
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        with st.expander("ℹ️ Detail Konfigurasi Model"):
            st.json({"Epoch terbaik": ckpt.get("epoch", "-"), "Val Loss terbaik": f"{ckpt.get('val_loss', '-'):.6f}",
                      "seq_len": cfg.get("seq_len"), "pred_len": cfg.get("pred_len"), "patch_len": cfg.get("patch_len"),
                      "stride": cfg.get("stride"), "d_model": cfg.get("d_model"), "n_heads": cfg.get("n_heads"),
                      "n_layers": cfg.get("n_layers"), "dropout": cfg.get("dropout")})
