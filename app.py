"""
app.py — BESTARIL BOO Streamlit Application
Non-Invasive Bladder Outlet Obstruction Analysis System
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import sys
from pathlib import Path

st.set_page_config(
    page_title="BESTARIL BOO | AI Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

sys.path.insert(0, str(Path(__file__).parent))
from utils.waveform_extractor import extract_from_dta, extract_from_pdf, get_waveform_source_label
from utils.predictor import predict, determine_model_route

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
:root {
    --bg-primary:#0a0f1a; --bg-card:#111827; --bg-input:#1e293b;
    --border:#1e3a5f; --border-light:#2d4a6e; --accent-cyan:#06b6d4;
    --text-primary:#f0f6ff; --text-secondary:#94a3b8; --text-muted:#64748b;
    --font-display:'DM Serif Display',serif; --font-body:'DM Sans',sans-serif; --font-mono:'JetBrains Mono',monospace;
}
.stApp { background:var(--bg-primary); font-family:var(--font-body); color:var(--text-primary); }
.stApp > header { background:transparent !important; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0d1425 0%,#0a1020 100%); border-right:1px solid var(--border); }
[data-testid="stSidebar"] .block-container { padding:1.5rem 1rem; }
.sidebar-brand { text-align:center; padding:1.5rem 0 2rem; border-bottom:1px solid var(--border); margin-bottom:1.5rem; }
.sidebar-brand-title { font-family:var(--font-display); font-size:1.6rem; color:var(--accent-cyan); letter-spacing:0.05em; margin:0; line-height:1.1; }
.sidebar-brand-sub { font-size:0.65rem; letter-spacing:0.2em; text-transform:uppercase; color:var(--text-muted); margin-top:0.3rem; }
.sidebar-engine-badge { display:inline-block; margin-top:0.8rem; padding:0.2rem 0.7rem; background:rgba(6,182,212,0.12); border:1px solid rgba(6,182,212,0.3); border-radius:20px; font-size:0.6rem; letter-spacing:0.15em; text-transform:uppercase; color:var(--accent-cyan); font-family:var(--font-mono); }
.section-header { font-family:var(--font-mono); font-size:0.6rem; letter-spacing:0.25em; text-transform:uppercase; color:var(--accent-cyan); margin:1.5rem 0 0.8rem; padding-bottom:0.4rem; border-bottom:1px solid rgba(6,182,212,0.2); }
.stTextInput > div > div > input, .stNumberInput > div > div > input { background:var(--bg-input) !important; border:1px solid var(--border) !important; border-radius:8px !important; color:var(--text-primary) !important; font-family:var(--font-body) !important; font-size:0.875rem !important; }
label { font-family:var(--font-body) !important; font-size:0.8rem !important; font-weight:500 !important; color:var(--text-secondary) !important; }
[data-testid="stFileUploader"] { border:1.5px dashed var(--border-light); border-radius:12px; background:rgba(59,130,246,0.04); transition:all 0.2s; }
[data-testid="stFileUploader"]:hover { border-color:var(--accent-cyan); background:rgba(6,182,212,0.06); }
.stButton > button { font-family:var(--font-body) !important; font-weight:600 !important; letter-spacing:0.05em !important; border-radius:10px !important; transition:all 0.2s ease !important; }
.stButton > button[kind="primary"] { background:linear-gradient(135deg,#1d4ed8,#0891b2) !important; border:none !important; color:white !important; padding:0.75rem 2rem !important; }
.stButton > button[kind="primary"]:hover { background:linear-gradient(135deg,#2563eb,#06b6d4) !important; transform:translateY(-1px) !important; box-shadow:0 4px 20px rgba(6,182,212,0.3) !important; }
.upload-zone-title { font-family:var(--font-mono); font-size:0.65rem; letter-spacing:0.2em; text-transform:uppercase; color:var(--accent-cyan); margin-bottom:0.5rem; }
.badge-req { display:inline-block; padding:0.1rem 0.5rem; background:rgba(239,68,68,0.15); border:1px solid rgba(239,68,68,0.3); border-radius:4px; font-size:0.6rem; color:#ef4444; letter-spacing:0.1em; text-transform:uppercase; font-family:var(--font-mono); margin-left:0.5rem; }
.badge-opt { display:inline-block; padding:0.1rem 0.5rem; background:rgba(59,130,246,0.15); border:1px solid rgba(59,130,246,0.3); border-radius:4px; font-size:0.6rem; color:#60a5fa; letter-spacing:0.1em; text-transform:uppercase; font-family:var(--font-mono); margin-left:0.5rem; }
.model-route-card { background:rgba(30,58,95,0.3); border:1px solid var(--border); border-radius:10px; padding:0.8rem 1rem; margin:0.8rem 0; }
.model-route-label { font-family:var(--font-mono); font-size:0.7rem; letter-spacing:0.1em; text-transform:uppercase; color:var(--text-muted); margin-bottom:0.2rem; }
.model-route-value { font-size:0.875rem; font-weight:600; color:var(--accent-cyan); }
.metric-box { background:var(--bg-input); border:1px solid var(--border); border-radius:10px; padding:0.8rem 1rem; margin:0.4rem 0; }
.metric-label { font-size:0.7rem; color:var(--text-muted); letter-spacing:0.05em; font-family:var(--font-mono); text-transform:uppercase; }
.warning-banner { background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.3); border-radius:10px; padding:0.8rem 1rem; font-size:0.8rem; color:#fca5a5; margin:0.5rem 0; }
.info-banner { background:rgba(59,130,246,0.08); border:1px solid rgba(59,130,246,0.25); border-radius:10px; padding:0.8rem 1rem; font-size:0.8rem; color:#93c5fd; margin:0.5rem 0; }
.result-label-boo { font-family:var(--font-display); font-size:2.5rem; letter-spacing:0.1em; color:#f97316; line-height:1; margin:0.5rem 0; }
.result-label-noboo { font-family:var(--font-display); font-size:2.5rem; letter-spacing:0.1em; color:#22c55e; line-height:1; margin:0.5rem 0; }
hr { border-color:var(--border) !important; margin:1.5rem 0 !important; }
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:var(--bg-primary); }
::-webkit-scrollbar-thumb { background:var(--border-light); border-radius:3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div class="sidebar-brand">
    <div class="sidebar-brand-title">BESTARIL</div>
    <div class="sidebar-brand-sub">BOO Detection System</div>
    <div class="sidebar-engine-badge">&#9889; Engine v2.0</div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Profil Pasien</div>', unsafe_allow_html=True)
    mrn  = st.text_input("No. Rekam Medis", placeholder="MRN-2025-XXXX")
    usia = st.number_input("Usia (Tahun)", min_value=18, max_value=100, value=65, step=1)

    st.markdown('<div class="section-header">Faktor Komorbid</div>', unsafe_allow_html=True)
    diabetes   = st.checkbox("Diabetes Melitus")
    hipertensi = st.checkbox("Hipertensi")
    stroke     = st.checkbox("Riwayat Stroke / CVD")
    neurologi  = st.checkbox("Kelainan Neurologis")
    op_prostat = st.checkbox("Riwayat Operasi Prostat")

    st.markdown('<div class="section-header">Model Override</div>', unsafe_allow_html=True)
    model_override = st.selectbox(
        "Paksa pakai model tertentu",
        ["Auto (Rekomendasi)", "Model C — Klinis Lengkap", "Model D — Klinis Ringkas", "Model A — Waveform Only"],
    )
    model_key_override = None
    if   "Model C" in model_override: model_key_override = 'C'
    elif "Model D" in model_override: model_key_override = 'D'
    elif "Model A" in model_override: model_key_override = 'A'

    st.markdown("---")
    st.markdown("""
<div style="font-size:0.65rem; color:#475569; line-height:1.6;">
<b style="color:#64748b;">DISCLAIMER KLINIS</b><br>
Sistem ini adalah alat bantu skrining penelitian.<br>
Bukan pengganti penilaian klinis spesialis urologi.<br>
Diagnosis definitif memerlukan pemeriksaan lengkap.
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,rgba(13,20,37,0.9) 0%,rgba(10,15,26,0.95) 100%);
     border:1px solid #1e3a5f; border-radius:16px; padding:2rem 2.5rem; margin-bottom:2rem;">
    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
        <div>
            <div style="font-family:'DM Serif Display',serif; font-size:1.75rem; color:#f0f6ff; margin:0 0 0.3rem;">
                Skrining BOO Non-Invasif
            </div>
            <div style="font-size:0.8rem; color:#64748b; letter-spacing:0.05em;">
                Bladder Outlet Obstruction &middot; AI-Powered Uroflowmetry Analysis
            </div>
            <div style="display:inline-flex; align-items:center; gap:0.4rem; padding:0.25rem 0.8rem;
                 background:rgba(34,197,94,0.12); border:1px solid rgba(34,197,94,0.3); border-radius:20px;
                 font-size:0.65rem; letter-spacing:0.1em; text-transform:uppercase; color:#22c55e;
                 font-family:'JetBrains Mono',monospace; margin-top:0.8rem;">
                &#9679; BESTARIL Engine v2.0 &middot; Model C (AUC 0.856) Primary
            </div>
        </div>
        <div style="text-align:right; font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#475569;">
            <div>RS Mitra Keluarga</div>
            <div>Dep. Urologi</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# INPUT COLUMNS
# ─────────────────────────────────────────────────────────────────────────────
col_upload, col_ufm, col_ipss = st.columns([1.2, 1, 1])

# ── Kolom 1: Upload ───────────────────────────────────────────────────────────
with col_upload:
    st.markdown('<div class="upload-zone-title">01 &middot; File Waveform <span class="badge-req">WAJIB</span></div>', unsafe_allow_html=True)
    pdf_file = st.file_uploader("PDF Uroflowmetry Report", type=['pdf'])

    st.markdown('<div class="upload-zone-title" style="margin-top:0.8rem;">File .DTA Binary <span class="badge-opt">OPSIONAL</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="info-banner">&#128161; Jika file .DTA tersedia, ekstraksi waveform lebih akurat dibanding dari PDF.</div>', unsafe_allow_html=True)
    dta_file = st.file_uploader("File .DTA (Laborie format)", type=['dta', 'DTA'])

    waveform_ready  = False
    waveform_data   = None
    waveform_source = "IDLE"
    ufm_from_pdf    = {}

    if dta_file is not None or pdf_file is not None:
        with st.spinner("Mengekstrak waveform..."):
            if dta_file is not None:
                with tempfile.NamedTemporaryFile(suffix='.dta', delete=False) as tmp:
                    tmp.write(dta_file.read())
                    tmp_path = tmp.name
                wave = extract_from_dta(tmp_path)
                os.unlink(tmp_path)
                if wave is not None:
                    waveform_data   = wave
                    waveform_source = "DTA"
                    waveform_ready  = True

            if pdf_file is not None:
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                    tmp.write(pdf_file.read())
                    tmp_path = tmp.name
                wave_pdf, ufm_from_pdf = extract_from_pdf(tmp_path)
                os.unlink(tmp_path)
                if waveform_data is None and wave_pdf is not None:
                    waveform_data   = wave_pdf
                    waveform_source = "PDF"
                    waveform_ready  = True
                elif waveform_data is None:
                    st.markdown('<div class="warning-banner">&#9888; Gagal mengekstrak waveform dari PDF. Coba upload file .DTA.</div>', unsafe_allow_html=True)

    if waveform_ready and waveform_data is not None:
        dot_cls   = "status-dot-ok" if waveform_source == "DTA" else "status-dot-warn"
        dot_color = "#22c55e" if waveform_source == "DTA" else "#f59e0b"
        src_label = get_waveform_source_label(waveform_source == "DTA")
        st.markdown(f"""
<div class="model-route-card" style="margin-top:0.8rem; display:flex; align-items:center; gap:0.8rem;">
    <div style="width:8px; height:8px; background:{dot_color}; border-radius:50%; flex-shrink:0;"></div>
    <div>
        <div class="model-route-label">Waveform Status</div>
        <div class="model-route-value">{src_label}</div>
    </div>
</div>
""", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(4.5, 1.8))
        fig.patch.set_facecolor('#111827')
        ax.set_facecolor('#0d1425')
        ax.plot(waveform_data, color='#06b6d4', linewidth=1.5, alpha=0.9)
        ax.fill_between(range(len(waveform_data)), waveform_data, alpha=0.15, color='#06b6d4')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_xlim(0, len(waveform_data)); ax.set_ylim(-0.05, 1.05)
        plt.tight_layout(pad=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    else:
        st.markdown("""
<div style="background:#0d1425; border:1.5px dashed #1e3a5f; border-radius:10px;
     height:90px; display:flex; align-items:center; justify-content:center;
     color:#334155; font-size:0.75rem; font-family:'JetBrains Mono',monospace;
     letter-spacing:0.1em; margin-top:0.8rem;">
    AWAITING FILE UPLOAD
</div>
""", unsafe_allow_html=True)


# ── Kolom 2: UFM ──────────────────────────────────────────────────────────────
with col_ufm:
    st.markdown('<div class="upload-zone-title">02 &middot; Metrik Aliran &amp; Prostat <span class="badge-opt">OPSIONAL</span></div>', unsafe_allow_html=True)

    def get_pdf_val(key, default):
        v = ufm_from_pdf.get(key, default)
        try:
            fv = float(v)
            return default if (fv != fv) else fv
        except:
            return default

    qmax  = st.number_input("Qmax (ml/s)",          min_value=0.0, max_value=100.0, value=get_pdf_val('Qmax_ml_s', 0.0),            step=0.1, format="%.1f")
    qave  = st.number_input("Qave (ml/s)",           min_value=0.0, max_value=50.0,  value=get_pdf_val('Qave_ml_s', 0.0),            step=0.1, format="%.1f")
    vvoid = st.number_input("Voided Volume (ml)",    min_value=0,   max_value=2000,  value=int(get_pdf_val('Voided_Volume_ml', 0)),  step=5)
    pvr   = st.number_input("PVR (ml)",              min_value=0,   max_value=1000,  value=int(get_pdf_val('PVR_ml', 0)),            step=5)
    ft    = st.number_input("Flow Time (s)",          min_value=0,   max_value=300,   value=int(get_pdf_val('Flow_Time_s', 0)),       step=1)
    ttmf  = st.number_input("Time to Max Flow (s)",  min_value=0,   max_value=120,   value=int(get_pdf_val('Time_to_Max_Flow_s', 0)),step=1)

    st.markdown("---")
    st.markdown('<div class="upload-zone-title" style="margin-top:0;">Volume Prostat &amp; IPP</div>', unsafe_allow_html=True)
    vol_prostat = st.number_input("Volume Prostat (ml/USG)", min_value=0, max_value=300, value=0, step=1)

    ipp_present = st.checkbox("IPP Terukur", value=False, help="Intravesical Prostatic Protrusion")
    ipp_value   = None
    if ipp_present:
        ipp_value = st.number_input("Nilai IPP (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.1, format="%.1f")

    if qmax > 0:
        qmax_flag  = "RENDAH" if qmax < 10 else ("NORMAL" if qmax <= 25 else "BAIK")
        qmax_color = "#f97316" if qmax < 10 else "#22c55e"
        st.markdown(f"""
<div class="metric-box">
    <div class="metric-label">Qmax Status</div>
    <div style="font-size:1.1rem; font-weight:600; color:{qmax_color};">{qmax:.1f} ml/s &mdash; {qmax_flag}</div>
</div>
""", unsafe_allow_html=True)


# ── Kolom 3: IPSS ─────────────────────────────────────────────────────────────
with col_ipss:
    st.markdown('<div class="upload-zone-title">03 &middot; Kategorisasi IPSS <span class="badge-opt">OPSIONAL</span></div>', unsafe_allow_html=True)

    sc_storage = st.number_input("Skor Storage (0-15)",          min_value=0, max_value=15, value=0, step=1)
    sc_voiding = st.number_input("Skor Voiding (0-20)",          min_value=0, max_value=20, value=0, step=1)
    sc_qol     = st.number_input("Post-Micturition / QoL (0-5)", min_value=0, max_value=5,  value=0, step=1)

    total_ipss = sc_storage + sc_voiding + sc_qol

    if total_ipss <= 7:
        ipss_cat   = "RINGAN"; ipss_color = "#22c55e"
    elif total_ipss <= 19:
        ipss_cat   = "SEDANG"; ipss_color = "#f59e0b"
    else:
        ipss_cat   = "BERAT";  ipss_color = "#ef4444"

    st.markdown(f"""
<div style="background:rgba(30,58,95,0.4); border:1px solid #1e3a5f; border-radius:12px;
     padding:1.2rem; text-align:center; margin:0.8rem 0;">
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.6rem; letter-spacing:0.2em;
         text-transform:uppercase; color:#64748b; margin-bottom:0.4rem;">TOTAL IPSS</div>
    <div style="font-family:'DM Serif Display',serif; font-size:2.5rem; color:{ipss_color}; line-height:1;">{total_ipss}</div>
    <div style="font-size:0.7rem; letter-spacing:0.15em; text-transform:uppercase;
         color:{ipss_color}; font-family:'JetBrains Mono',monospace; margin-top:0.3rem;">SYMPTOM: {ipss_cat}</div>
</div>
""", unsafe_allow_html=True)

    ufm_preview = {
        'Qmax_ml_s': qmax, 'Qave_ml_s': qave, 'Voided_Volume_ml': float(vvoid),
        'PVR_ml': float(pvr), 'Flow_Time_s': float(ft), 'Time_to_Max_Flow_s': float(ttmf)
    }
    cli_preview = {
        "Total skor IPSS": float(total_ipss), "Skor Storage": float(sc_storage),
        "Skor Voiding": float(sc_voiding), "QoL": float(sc_qol),
        "Usia": float(usia), "Volume Prostat": float(vol_prostat),
        "IPP": float(ipp_value) if ipp_present and ipp_value is not None else np.nan,
        "Diabetes": 1.0 if diabetes else 0.0,
        "Stroke": 1.0 if stroke else 0.0,
        "Neurology abnormalities": 1.0 if neurologi else 0.0
    }

    route = model_key_override if model_key_override else determine_model_route(ufm_preview, cli_preview, {})

    route_desc   = {'A': "Model A &mdash; Waveform Only", 'C': "Model C &mdash; Klinis Lengkap", 'D': "Model D &mdash; Klinis Ringkas"}
    route_detail = {'A': "Akurasi rendah. Gunakan jika tidak ada data klinis.",
                    'C': "AUC 0.856 CV &middot; 0.812 External. Rekomendasi utama.",
                    'D': "AUC 0.838 CV. Fallback jika data klinis tidak lengkap."}

    st.markdown(f"""
<div class="model-route-card">
    <div class="model-route-label">Model yang akan digunakan</div>
    <div class="model-route-value">{route_desc[route]}</div>
    <div style="font-size:0.7rem; color:#64748b; margin-top:0.2rem;">{route_detail[route]}</div>
</div>
""", unsafe_allow_html=True)

    if route == 'A':
        st.markdown('<div class="warning-banner">&#9888; <b>Akurasi Terbatas.</b> Lengkapi data IPSS dan klinis untuk hasil lebih akurat.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# BUTTON
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_, col_btn, _ = st.columns([1, 2, 1])
with col_btn:
    run_analysis = st.button("&#9889; PROSES ANALISIS BESTARIL-BOO", type="primary",
                              use_container_width=True, disabled=not waveform_ready)
    if not waveform_ready:
        st.markdown('<div style="text-align:center; font-size:0.72rem; color:#475569; margin-top:0.3rem;">Upload PDF Uroflowmetry untuk mengaktifkan analisis</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────
if run_analysis and waveform_ready:
    st.markdown("---")

    ufm_params = {
        'Qmax_ml_s': float(qmax), 'Qave_ml_s': float(qave),
        'Voided_Volume_ml': float(vvoid), 'PVR_ml': float(pvr),
        'Flow_Time_s': float(ft), 'Time_to_Max_Flow_s': float(ttmf)
    }
    clinical_full = {
        "Total skor IPSS": float(total_ipss), "Skor Storage": float(sc_storage),
        "Skor Voiding": float(sc_voiding), "QoL": float(sc_qol),
        "Usia": float(usia), "Volume Prostat": float(vol_prostat),
        "IPP": float(ipp_value) if ipp_present and ipp_value is not None else np.nan,
        "Diabetes": 1.0 if diabetes else 0.0,
        "Stroke": 1.0 if stroke else 0.0,
        "Neurology abnormalities": 1.0 if neurologi else 0.0
    }

    with st.spinner("Menganalisis pola waveform..."):
        try:
            result = predict(waveform_data, ufm_params, clinical_full, model_key_override)
        except FileNotFoundError as e:
            st.error(f"Model tidak ditemukan: {e}\n\nPastikan file .pt, .pkl, .json ada di folder models/")
            st.stop()
        except Exception as e:
            st.error(f"Error saat analisis: {e}")
            st.stop()

    # ── HITUNG SEMUA VARIABEL DI SINI, SEBELUM HTML APAPUN ──────────────────
    prob_pct     = result['prob_boo'] * 100
    pred_boo     = result['pred_boo']
    conf         = result['confidence']
    conf_col     = result['confidence_color']
    model_key    = result['model_key']

    display_pct   = prob_pct if pred_boo else (100.0 - prob_pct)
    bar_width_str = f"{display_pct:.1f}"
    bar_gradient  = "linear-gradient(90deg,#f97316,#ef4444)" if pred_boo else "linear-gradient(90deg,#14b8a6,#22c55e)"
    conf_display  = f"{display_pct:.1f}"

    pred_label    = "BOO" if pred_boo else "Non-BOO"
    pred_label_up = "BOO" if pred_boo else "NON-BOO"
    result_color  = "#f97316" if pred_boo else "#22c55e"
    result_text   = "BOO TERDETEKSI" if pred_boo else "NON-BOO"
    boo_color     = "#f97316" if pred_boo else "#22c55e"
    bg_color      = "rgba(249,115,22,0.06)" if pred_boo else "rgba(34,197,94,0.06)"
    border_color  = "rgba(249,115,22,0.3)"  if pred_boo else "rgba(34,197,94,0.3)"
    label_css     = "result-label-boo" if pred_boo else "result-label-noboo"

    conf_bg_map   = {"TINGGI": "rgba(34,197,94,0.15)",  "SEDANG": "rgba(245,158,11,0.15)", "RENDAH": "rgba(239,68,68,0.15)"}
    conf_bd_map   = {"TINGGI": "rgba(34,197,94,0.4)",   "SEDANG": "rgba(245,158,11,0.4)",  "RENDAH": "rgba(239,68,68,0.4)"}
    conf_bg       = conf_bg_map[conf]
    conf_bd       = conf_bd_map[conf]

    interp_title  = "Bladder Outlet Obstruction" if pred_boo else "Tidak Ditemukan Obstruksi"
    interp_body   = (
        f"Pola waveform mengindikasikan obstruksi dengan probabilitas {prob_pct:.1f}%. "
        "Skor IPSS Voiding dan volume prostat mendukung temuan ini."
        if pred_boo else
        "Pola waveform tidak menunjukkan obstruksi signifikan. "
        "Kemungkinan Detrusor Underactivity (DU) atau kondisi lainnya perlu dievaluasi."
    )

    ipss_warn_col  = "#f59e0b" if total_ipss == 0 else "#f0f6ff"
    ipss_warn_icon = " &#9888;" if total_ipss == 0 else ""
    qmax_color_rep = "#f97316" if qmax < 10 else "#f0f6ff"
    src_label      = get_waveform_source_label(waveform_source == "DTA")

    if result['needs_urodynamics']:
        uds_bg    = "rgba(245,158,11,0.08)"; uds_bd     = "rgba(245,158,11,0.3)"
        uds_icon  = "&#128302;";             uds_lbl_cl = "#f59e0b"
        uds_title = "Pemeriksaan UDS Direkomendasikan"; uds_title_cl = "#fcd34d"
        uds_body  = ("Confidence rendah atau model terbatas. UDS diperlukan untuk konfirmasi diagnostik."
                     if conf == 'RENDAH' or model_key == 'A' else
                     "Temuan borderline. UDS dapat memperkuat keputusan terapi.")
    else:
        uds_bg    = "rgba(34,197,94,0.08)";  uds_bd     = "rgba(34,197,94,0.3)"
        uds_icon  = "&#9989;";               uds_lbl_cl = "#22c55e"
        uds_title = "Data BESTARIL Cukup Meyakinkan"; uds_title_cl = "#4ade80"
        uds_body  = "Untuk saat ini prosedur invasif dapat ditunda berdasarkan confidence analisis."

    # ── Warning jika data kosong ──────────────────────────────────────────────
    if model_key == 'C' and sc_storage == 0 and sc_voiding == 0 and sc_qol == 0 and vol_prostat == 0:
        st.markdown('<div class="warning-banner">&#9888; <b>Data klinis belum diisi.</b> Skor IPSS dan volume prostat masih 0. Lengkapi data klinis untuk prediksi bermakna.</div>', unsafe_allow_html=True)

    # ── Header hasil ─────────────────────────────────────────────────────────
    st.markdown(f"""
<div style="background:#111827; border:1px solid #1e3a5f; border-radius:16px;
     padding:1.2rem 2rem; margin-bottom:1.5rem;">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.6rem;
                 letter-spacing:0.2em; text-transform:uppercase; color:#64748b;">
                Hasil Analisis &middot; {mrn or 'Pasien Baru'}
            </div>
            <div style="font-family:'DM Serif Display',serif; font-size:1.3rem; margin-top:0.2rem;">
                {result['model_used']} &middot; {result['model_description']}
            </div>
        </div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem; color:#475569; text-align:right;">
            <div>Threshold (Youden): {result['threshold']:.3f}</div>
            <div>CV AUC: {result['model_auc_cv']} &middot; Ext AUC: {result['model_auc_ext']}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

    rc1, rc2, rc3 = st.columns([1, 1.2, 1.2])

    # ── RC1: Probabilitas ─────────────────────────────────────────────────────
    with rc1:
        st.markdown(f"""
<div style="background:{bg_color}; border:1px solid {border_color}; border-radius:16px; padding:2rem; text-align:center;">
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.6rem; letter-spacing:0.2em; text-transform:uppercase; color:#64748b;">
        Probabilitas NIVA-BOO
    </div>
    <div class="{label_css}">{pred_label}</div>
    <div style="font-family:'JetBrains Mono',monospace; font-size:1.5rem; font-weight:500; color:{conf_col};">
        {prob_pct:.1f}%
    </div>
    <div style="margin:1rem 0; background:#1e293b; border-radius:8px; height:6px; overflow:hidden;">
        <div style="width:{bar_width_str}%; height:100%; background:{bar_gradient}; border-radius:8px;"></div>
    </div>
    <div style="display:inline-block; padding:0.35rem 1.2rem; border-radius:20px; font-family:'JetBrains Mono',monospace; font-size:0.7rem; letter-spacing:0.15em; text-transform:uppercase; margin-top:0.5rem; background:{conf_bg}; border:1px solid {conf_bd}; color:{conf_col};">
        {pred_label_up} &middot; CONFIDENCE: {conf}
    </div>
</div>
""", unsafe_allow_html=True)

    # ── RC2: Interpretasi ─────────────────────────────────────────────────────
    with rc2:
        st.markdown(f"""
<div style="background:#111827; border:1px solid {border_color}; border-radius:14px; padding:1.2rem 1.5rem; margin-bottom:0.8rem;">
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.6rem; letter-spacing:0.2em; text-transform:uppercase; color:#64748b; margin-bottom:0.5rem;">
        Interpretasi BESTARIL
    </div>
    <div style="font-family:'DM Serif Display',serif; font-size:1.2rem; color:{boo_color}; margin-bottom:0.5rem;">
        {interp_title}
    </div>
    <div style="font-size:0.82rem; color:#94a3b8; line-height:1.6;">{interp_body}</div>
</div>
<div style="background:#111827; border:1px solid {border_color}; border-radius:14px; padding:1.2rem 1.5rem;">
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.6rem; letter-spacing:0.2em; text-transform:uppercase; color:#64748b; margin-bottom:0.5rem;">
        Saran Klinis
    </div>
    <div style="font-family:'DM Serif Display',serif; font-size:1rem; margin-bottom:0.5rem;">
        {result['clinical_recommendation']}
    </div>
    <div style="font-size:0.82rem; color:#94a3b8; line-height:1.6;">{result['clinical_detail']}</div>
</div>
""", unsafe_allow_html=True)

    # ── RC3: UDS + Waveform ───────────────────────────────────────────────────
    with rc3:
        st.markdown(f"""
<div style="background:{uds_bg}; border:1px solid {uds_bd}; border-radius:12px; padding:1rem 1.2rem; display:flex; align-items:flex-start; gap:1rem; margin-bottom:0.8rem;">
    <div style="font-size:1.5rem; flex-shrink:0;">{uds_icon}</div>
    <div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.6rem; letter-spacing:0.15em; text-transform:uppercase; color:{uds_lbl_cl}; margin-bottom:0.3rem;">
            Kebutuhan Urodinamika (UDS)
        </div>
        <div style="font-size:0.82rem; color:{uds_title_cl}; font-weight:500;">{uds_title}</div>
        <div style="font-size:0.75rem; color:#94a3b8; margin-top:0.3rem;">{uds_body}</div>
    </div>
</div>
""", unsafe_allow_html=True)

        if waveform_data is not None:
            wave_color = '#f97316' if pred_boo else '#22c55e'
            fig2, ax2  = plt.subplots(figsize=(4, 1.5))
            fig2.patch.set_facecolor('#111827')
            ax2.set_facecolor('#0d1425')
            ax2.plot(waveform_data, color=wave_color, linewidth=1.5, alpha=0.9)
            ax2.fill_between(range(len(waveform_data)), waveform_data, alpha=0.12, color=wave_color)
            ax2.set_xticks([]); ax2.set_yticks([])
            for sp in ax2.spines.values(): sp.set_visible(False)
            ax2.set_xlim(0, len(waveform_data)); ax2.set_ylim(-0.05, 1.05)
            plt.tight_layout(pad=0.3)
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        st.markdown(f"""
<div style="font-family:'JetBrains Mono',monospace; font-size:0.6rem; color:#475569; margin-top:0.4rem; text-align:center;">
    Sumber: {src_label}
</div>
""", unsafe_allow_html=True)

    # ── Laporan ───────────────────────────────────────────────────────────────
    st.markdown(f"""
<div style="background:#111827; border:1px solid #1e3a5f; border-radius:16px; padding:2rem 2.5rem; margin-top:1.5rem;">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.6rem; letter-spacing:0.2em; text-transform:uppercase; color:#64748b; margin-bottom:0.3rem;">
                Laporan Analisis BESTARIL-BOO
            </div>
            <div style="font-family:'DM Serif Display',serif; font-size:1.3rem;">
                Non-Invasive Analysis for Bladder Outlet Obstruction
            </div>
        </div>
        <div style="text-align:right; font-size:0.8rem; color:#64748b;">
            <div style="font-weight:600; color:#94a3b8;">{mrn or '&mdash;'}</div>
            <div>Usia {usia} tahun</div>
        </div>
    </div>
    <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:1.5rem; margin-top:1.5rem; padding-top:1.5rem; border-top:1px solid #1e3a5f;">
        <div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.6rem; letter-spacing:0.2em; text-transform:uppercase; color:#64748b; margin-bottom:0.3rem;">No. RM Pasien</div>
            <div style="font-size:1rem; font-weight:600; color:#f0f6ff;">{mrn or '&mdash;'}</div>
            <div style="font-size:0.7rem; color:#64748b; margin-top:0.2rem;">Usia: {usia} thn</div>
        </div>
        <div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.6rem; letter-spacing:0.2em; text-transform:uppercase; color:#64748b; margin-bottom:0.3rem;">Profil LUTS</div>
            <div style="font-size:1rem; font-weight:600; color:{ipss_warn_col};">IPSS: {total_ipss} ({ipss_cat}){ipss_warn_icon}</div>
            <div style="font-size:0.7rem; color:#64748b; margin-top:0.2rem;">Voiding: {sc_voiding} &middot; Storage: {sc_storage}</div>
        </div>
        <div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.6rem; letter-spacing:0.2em; text-transform:uppercase; color:#64748b; margin-bottom:0.3rem;">Uroflowmetry</div>
            <div style="font-size:1rem; font-weight:600; color:{qmax_color_rep};">Qmax: {qmax:.1f} mL/s</div>
            <div style="font-size:0.7rem; color:#64748b; margin-top:0.2rem;">PVR: {pvr} mL</div>
        </div>
        <div>
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.6rem; letter-spacing:0.2em; text-transform:uppercase; color:#64748b; margin-bottom:0.3rem;">Analisis BESTARIL</div>
            <div style="font-size:1rem; font-weight:600; color:{result_color};">{result_text}</div>
            <div style="font-size:0.7rem; color:#64748b; margin-top:0.2rem;">Confidence: {conf_display}%</div>
        </div>
    </div>
    <div style="margin-top:1.2rem; padding-top:1rem; border-top:1px solid #1e3a5f; font-size:0.7rem; color:#475569; font-family:'JetBrains Mono',monospace;">
        Model: {result['model_used']} &middot; Threshold: {result['threshold']:.3f} (Youden's J) &middot;
        Waveform: {src_label} &middot;
        &#9888; Alat bantu skrining penelitian &mdash; bukan pengganti penilaian spesialis
    </div>
</div>
""", unsafe_allow_html=True)

    if model_key == 'A':
        st.markdown("""
<div class="warning-banner" style="margin-top:1rem;">
&#9888; <b>PERHATIAN:</b> Analisis menggunakan Model A (Waveform Only) dengan AUC 0.52 &mdash;
hampir setara random. Hasil ini <b>tidak boleh digunakan</b> sebagai dasar keputusan klinis.
Lengkapi data IPSS dan parameter klinis untuk akurasi yang bermakna.
</div>
""", unsafe_allow_html=True)

elif not waveform_ready:
    st.markdown("""
<div style="text-align:center; padding:4rem 2rem; opacity:0.4;">
    <div style="font-size:3rem; margin-bottom:1rem;">&#128302;</div>
    <div style="font-family:'DM Serif Display',serif; font-size:1.2rem; color:#94a3b8;">
        Upload PDF Uroflowmetry untuk memulai analisis
    </div>
    <div style="font-size:0.8rem; color:#64748b; margin-top:0.5rem;">
        File .DTA opsional untuk akurasi lebih tinggi
    </div>
</div>
""", unsafe_allow_html=True)