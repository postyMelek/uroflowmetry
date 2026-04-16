"""
app.py — NIVA BOO Streamlit Application
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
    page_title="NIVA-BOO | AI Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

sys.path.insert(0, str(Path(__file__).parent))
from utils.waveform_extractor import extract_from_dta, extract_from_pdf, get_waveform_source_label
from utils.predictor import predict, determine_model_route

# ─────────────────────────────────────────────────────────────────────────────
# CSS — Light Theme, Clean Medical Design
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap');

:root {
    --bg-primary: #f5f7fa;
    --bg-white: #ffffff;
    --bg-soft: #eef1f6;
    --bg-input: #f9fafb;
    --border: #d1d9e6;
    --border-strong: #b0bdd0;
    --accent: #1a5fa8;
    --accent-light: #e8f1fb;
    --accent-mid: #3b82c4;
    --teal: #0d8a74;
    --teal-light: #e6f6f3;
    --amber: #b45309;
    --amber-light: #fef3c7;
    --red: #c0392b;
    --red-light: #fdecea;
    --green: #166534;
    --green-light: #dcfce7;
    --orange: #c2410c;
    --orange-light: #fff7ed;
    --text-primary: #1a2233;
    --text-secondary: #4a5568;
    --text-muted: #8a9bb0;
    --font-display: 'Libre Baskerville', Georgia, serif;
    --font-body: 'Plus Jakarta Sans', sans-serif;
    --font-mono: 'Fira Code', monospace;
    --shadow-sm: 0 1px 3px rgba(26,42,67,0.08), 0 1px 2px rgba(26,42,67,0.05);
    --shadow-md: 0 4px 16px rgba(26,42,67,0.10), 0 1px 4px rgba(26,42,67,0.06);
    --shadow-lg: 0 8px 32px rgba(26,42,67,0.12), 0 2px 8px rgba(26,42,67,0.07);
    --radius: 12px;
    --radius-sm: 8px;
    --radius-lg: 18px;
}

/* ── Global ── */
.stApp {
    background: var(--bg-primary);
    font-family: var(--font-body);
    color: var(--text-primary);
}
.stApp > header { background: transparent !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a3a6b 0%, #122a52 100%);
    border-right: none;
    box-shadow: 4px 0 20px rgba(26,42,67,0.15);
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem; }

/* Sidebar text overrides */
[data-testid="stSidebar"] label {
    color: #c8d8f0 !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    font-family: var(--font-body) !important;
    letter-spacing: 0.02em !important;
}
[data-testid="stSidebar"] .stCheckbox label { color: #c8d8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label { color: #c8d8f0 !important; }
[data-testid="stSidebar"] .stNumberInput label { color: #c8d8f0 !important; }
[data-testid="stSidebar"] .stTextInput label { color: #c8d8f0 !important; }

[data-testid="stSidebar"] .stTextInput > div > div > input,
[data-testid="stSidebar"] .stNumberInput > div > div > input {
    background: rgba(255,255,255,0.10) !important;
    border: 1px solid rgba(255,255,255,0.20) !important;
    border-radius: var(--radius-sm) !important;
    color: #f0f6ff !important;
    font-family: var(--font-body) !important;
    font-size: 0.875rem !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.10) !important;
    border: 1px solid rgba(255,255,255,0.20) !important;
    border-radius: var(--radius-sm) !important;
    color: #f0f6ff !important;
}

/* ── Main area inputs ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: var(--bg-input) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    font-size: 0.875rem !important;
    transition: border-color 0.2s;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--accent-mid) !important;
    box-shadow: 0 0 0 3px rgba(59,130,196,0.12) !important;
}

label {
    font-family: var(--font-body) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.01em !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--border-strong);
    border-radius: var(--radius);
    background: var(--bg-white);
    transition: all 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-mid);
    background: var(--accent-light);
}

/* ── Button ── */
.stButton > button {
    font-family: var(--font-body) !important;
    font-weight: 700 !important;
    letter-spacing: 0.04em !important;
    border-radius: var(--radius) !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #1a5fa8, #2176c7) !important;
    border: none !important;
    color: white !important;
    padding: 0.75rem 2rem !important;
    box-shadow: 0 4px 14px rgba(26,95,168,0.30) !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #2176c7, #3b93e0) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(26,95,168,0.38) !important;
}
.stButton > button[kind="primary"]:disabled {
    background: #b0bdd0 !important;
    box-shadow: none !important;
    transform: none !important;
}

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-soft); }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 3px; }

hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="text-align:center; padding:1.8rem 0 2rem; border-bottom:1px solid rgba(255,255,255,0.12); margin-bottom:1.8rem;">
    <div style="display:inline-flex; align-items:center; justify-content:center; width:52px; height:52px;
         background:rgba(255,255,255,0.12); border-radius:14px; margin-bottom:0.8rem; border:1px solid rgba(255,255,255,0.2);">
        <span style="font-size:1.5rem;">🔬</span>
    </div>
    <div style="font-family:'Libre Baskerville',serif; font-size:1.5rem; color:#ffffff; font-weight:700; letter-spacing:0.04em; line-height:1;">NIVA-BOO</div>
    <div style="font-size:0.62rem; letter-spacing:0.22em; text-transform:uppercase; color:#90aad0; margin-top:0.3rem; font-family:'Plus Jakarta Sans',sans-serif;">BOO Detection System</div>
    <div style="display:inline-block; margin-top:0.8rem; padding:0.2rem 0.8rem;
         background:rgba(59,147,224,0.25); border:1px solid rgba(59,147,224,0.4); border-radius:20px;
         font-size:0.6rem; letter-spacing:0.15em; text-transform:uppercase; color:#90c8f0; font-family:'Fira Code',monospace;">
        ⚡ Engine v2.0
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div style="font-family:\'Fira Code\',monospace; font-size:0.58rem; letter-spacing:0.22em; text-transform:uppercase; color:#7090b8; margin:0 0 0.7rem; padding-bottom:0.4rem; border-bottom:1px solid rgba(255,255,255,0.08);">Profil Pasien</div>', unsafe_allow_html=True)
    mrn  = st.text_input("No. Rekam Medis", placeholder="MRN-2025-XXXX")
    usia = st.number_input("Usia (Tahun)", min_value=18, max_value=100, value=65, step=1)

    st.markdown(
        """
        <style>
        /* Mengubah warna teks semua label checkbox menjadi putih */
        .stCheckbox label p {
            color: white !important;
            font-weight: 500;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    diabetes   = st.checkbox("Diabetes Melitus")
    hipertensi = st.checkbox("Hipertensi")
    stroke     = st.checkbox("Riwayat Stroke / CVD")
    neurologi  = st.checkbox("Kelainan Neurologis")
    op_prostat = st.checkbox("Riwayat Operasi Prostat")

    st.markdown('<div style="font-family:\'Fira Code\',monospace; font-size:0.58rem; letter-spacing:0.22em; text-transform:uppercase; color:#7090b8; margin:1.5rem 0 0.7rem; padding-bottom:0.4rem; border-bottom:1px solid rgba(255,255,255,0.08);">Model Override</div>', unsafe_allow_html=True)
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
<div style="font-size:0.65rem; color:#5878a0; line-height:1.7; font-family:'Plus Jakarta Sans',sans-serif;">
<b style="color:#7090b8; font-size:0.62rem; letter-spacing:0.12em; text-transform:uppercase;">⚠ Disclaimer Klinis</b><br>
Sistem ini adalah alat bantu skrining penelitian.<br>
Bukan pengganti penilaian klinis spesialis urologi.<br>
Diagnosis definitif memerlukan pemeriksaan lengkap.
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#ffffff; border:1.5px solid #d1d9e6; border-radius:18px;
     padding:2rem 2.5rem; margin-bottom:1.8rem;
     box-shadow:0 4px 16px rgba(26,42,67,0.08);">
    <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap; gap:1rem;">
        <div>
            <div style="display:flex; align-items:center; gap:0.6rem; margin-bottom:0.4rem;">
                <div style="width:4px; height:32px; background:linear-gradient(180deg,#1a5fa8,#0d8a74); border-radius:4px;"></div>
                <div style="font-family:'Libre Baskerville',serif; font-size:1.7rem; color:#1a2233; font-weight:700; line-height:1.1;">
                    Skrining BOO Non-Invasif
                </div>
            </div>
            <div style="font-size:0.82rem; color:#8a9bb0; letter-spacing:0.03em; margin-left:1rem; font-family:'Plus Jakarta Sans',sans-serif;">
                Bladder Outlet Obstruction &middot; AI-Powered Uroflowmetry Analysis
            </div>
            <div style="margin-left:1rem; margin-top:0.8rem;">
                <span style="display:inline-flex; align-items:center; gap:0.4rem; padding:0.28rem 0.9rem;
                     background:#e6f6f3; border:1.5px solid #0d8a74; border-radius:20px;
                     font-size:0.65rem; letter-spacing:0.08em; text-transform:uppercase; color:#0d8a74;
                     font-family:'Fira Code',monospace; font-weight:500;">
                    ● NIVA Engine v2.0 &middot; Model C (AUC 0.856) Primary
                </span>
            </div>
        </div>
        <div style="text-align:right; font-family:'Fira Code',monospace; font-size:0.7rem; color:#8a9bb0;
             background:#f5f7fa; border:1px solid #d1d9e6; border-radius:10px; padding:0.8rem 1.1rem;">
            <div style="font-weight:600; color:#4a5568; font-size:0.75rem;">RSUP Persahabatan</div>
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
    st.markdown("""
<div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.7rem;">
    <div style="background:#1a5fa8; color:white; font-family:'Fira Code',monospace; font-size:0.6rem;
         padding:0.2rem 0.5rem; border-radius:6px; font-weight:600; letter-spacing:0.06em;">01</div>
    <span style="font-family:'Fira Code',monospace; font-size:0.65rem; letter-spacing:0.18em; text-transform:uppercase; color:#4a5568; font-weight:500;">File Waveform</span>
    <span style="background:#fdecea; color:#c0392b; border:1px solid #f5c6c2; border-radius:5px;
         font-size:0.58rem; padding:0.1rem 0.5rem; font-family:'Fira Code',monospace; letter-spacing:0.08em; text-transform:uppercase;">WAJIB</span>
</div>
""", unsafe_allow_html=True)
    pdf_file = st.file_uploader("PDF Uroflowmetry Report", type=['pdf'])

    st.markdown("""
<div style="display:flex; align-items:center; gap:0.5rem; margin:0.9rem 0 0.5rem;">
    <span style="font-family:'Fira Code',monospace; font-size:0.65rem; letter-spacing:0.18em; text-transform:uppercase; color:#4a5568; font-weight:500;">File .DTA Binary</span>
    <span style="background:#e8f1fb; color:#1a5fa8; border:1px solid #b8d2f0; border-radius:5px;
         font-size:0.58rem; padding:0.1rem 0.5rem; font-family:'Fira Code',monospace; letter-spacing:0.08em; text-transform:uppercase;">OPSIONAL</span>
</div>
<div style="background:#e8f1fb; border:1.5px solid #b8d2f0; border-radius:10px; padding:0.7rem 0.9rem;
     font-size:0.78rem; color:#1a5fa8; margin-bottom:0.5rem; font-family:'Plus Jakarta Sans',sans-serif;">
    💡 Jika file .DTA tersedia, ekstraksi waveform lebih akurat dibanding dari PDF.
</div>
""", unsafe_allow_html=True)
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
                    st.markdown('<div style="background:#fff7ed; border:1.5px solid #fed7aa; border-radius:10px; padding:0.7rem 0.9rem; font-size:0.78rem; color:#b45309;">⚠️ Gagal mengekstrak waveform dari PDF. Coba upload file .DTA.</div>', unsafe_allow_html=True)

    if waveform_ready and waveform_data is not None:
        dot_color = "#0d8a74" if waveform_source == "DTA" else "#b45309"
        dot_bg    = "#e6f6f3" if waveform_source == "DTA" else "#fef3c7"
        dot_bd    = "#0d8a74" if waveform_source == "DTA" else "#b45309"
        src_label = get_waveform_source_label(waveform_source == "DTA")
        st.markdown(f"""
<div style="background:{dot_bg}; border:1.5px solid {dot_bd}; border-radius:10px; padding:0.6rem 0.9rem;
     margin-top:0.7rem; display:flex; align-items:center; gap:0.7rem;">
    <div style="width:9px; height:9px; background:{dot_color}; border-radius:50%; flex-shrink:0;"></div>
    <div>
        <div style="font-family:'Fira Code',monospace; font-size:0.58rem; letter-spacing:0.12em; text-transform:uppercase; color:{dot_color}; margin-bottom:0.1rem;">Waveform Status</div>
        <div style="font-size:0.82rem; font-weight:600; color:{dot_color};">{src_label}</div>
    </div>
</div>
""", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(4.5, 1.8))
        fig.patch.set_facecolor('#f5f7fa')
        ax.set_facecolor('#eef1f6')
        ax.plot(waveform_data, color='#1a5fa8', linewidth=1.8, alpha=0.9)
        ax.fill_between(range(len(waveform_data)), waveform_data, alpha=0.18, color='#1a5fa8')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_xlim(0, len(waveform_data)); ax.set_ylim(-0.05, 1.05)
        plt.tight_layout(pad=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close()
    else:
        st.markdown("""
<div style="background:#f5f7fa; border:2px dashed #d1d9e6; border-radius:10px;
     height:90px; display:flex; align-items:center; justify-content:center;
     color:#b0bdd0; font-size:0.75rem; font-family:'Fira Code',monospace;
     letter-spacing:0.1em; margin-top:0.8rem;">
    AWAITING FILE UPLOAD
</div>
""", unsafe_allow_html=True)


# ── Kolom 2: UFM ──────────────────────────────────────────────────────────────
with col_ufm:
    st.markdown("""
<div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.7rem;">
    <div style="background:#0d8a74; color:white; font-family:'Fira Code',monospace; font-size:0.6rem;
         padding:0.2rem 0.5rem; border-radius:6px; font-weight:600; letter-spacing:0.06em;">02</div>
    <span style="font-family:'Fira Code',monospace; font-size:0.65rem; letter-spacing:0.18em; text-transform:uppercase; color:#4a5568; font-weight:500;">Metrik Aliran &amp; Prostat</span>
    <span style="background:#e8f1fb; color:#1a5fa8; border:1px solid #b8d2f0; border-radius:5px;
         font-size:0.58rem; padding:0.1rem 0.5rem; font-family:'Fira Code',monospace; letter-spacing:0.08em; text-transform:uppercase;">OPSIONAL</span>
</div>
""", unsafe_allow_html=True)

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

    st.markdown("<hr style='margin:0.8rem 0;'>", unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'Fira Code\',monospace; font-size:0.62rem; letter-spacing:0.15em; text-transform:uppercase; color:#8a9bb0; margin-bottom:0.5rem;">Volume Prostat &amp; IPP</div>', unsafe_allow_html=True)
    vol_prostat = st.number_input("Volume Prostat (ml/USG)", min_value=0, max_value=300, value=0, step=1)

    ipp_present = st.checkbox("IPP Terukur", value=False, help="Intravesical Prostatic Protrusion")
    ipp_value   = None
    if ipp_present:
        ipp_value = st.number_input("Nilai IPP (mm)", min_value=0.0, max_value=50.0, value=0.0, step=0.1, format="%.1f")

    if qmax > 0:
        qmax_flag  = "RENDAH" if qmax < 10 else ("NORMAL" if qmax <= 25 else "BAIK")
        qmax_bg    = "#fdecea" if qmax < 10 else "#dcfce7"
        qmax_col   = "#c0392b" if qmax < 10 else "#166534"
        qmax_bd    = "#f5c6c2" if qmax < 10 else "#86efac"
        st.markdown(f"""
<div style="background:{qmax_bg}; border:1.5px solid {qmax_bd}; border-radius:10px; padding:0.7rem 0.9rem; margin-top:0.5rem;">
    <div style="font-family:'Fira Code',monospace; font-size:0.6rem; letter-spacing:0.12em; text-transform:uppercase; color:{qmax_col}; margin-bottom:0.2rem;">Qmax Status</div>
    <div style="font-size:1rem; font-weight:700; color:{qmax_col};">{qmax:.1f} ml/s — {qmax_flag}</div>
</div>
""", unsafe_allow_html=True)


# ── Kolom 3: IPSS ─────────────────────────────────────────────────────────────
with col_ipss:
    st.markdown("""
<div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.7rem;">
    <div style="background:#6d28d9; color:white; font-family:'Fira Code',monospace; font-size:0.6rem;
         padding:0.2rem 0.5rem; border-radius:6px; font-weight:600; letter-spacing:0.06em;">03</div>
    <span style="font-family:'Fira Code',monospace; font-size:0.65rem; letter-spacing:0.18em; text-transform:uppercase; color:#4a5568; font-weight:500;">Kategorisasi IPSS</span>
    <span style="background:#e8f1fb; color:#1a5fa8; border:1px solid #b8d2f0; border-radius:5px;
         font-size:0.58rem; padding:0.1rem 0.5rem; font-family:'Fira Code',monospace; letter-spacing:0.08em; text-transform:uppercase;">OPSIONAL</span>
</div>
""", unsafe_allow_html=True)

    sc_storage = st.number_input("Skor Storage (0-15)",          min_value=0, max_value=15, value=0, step=1)
    sc_voiding = st.number_input("Skor Voiding (0-20)",          min_value=0, max_value=20, value=0, step=1)
    sc_qol     = st.number_input("Post-Micturition / QoL (0-5)", min_value=0, max_value=5,  value=0, step=1)

    total_ipss = sc_storage + sc_voiding + sc_qol

    if total_ipss <= 7:
        ipss_cat   = "RINGAN"; ipss_col = "#166534"; ipss_bg = "#dcfce7"; ipss_bd = "#86efac"
    elif total_ipss <= 19:
        ipss_cat   = "SEDANG"; ipss_col = "#b45309"; ipss_bg = "#fef3c7"; ipss_bd = "#fcd34d"
    else:
        ipss_cat   = "BERAT";  ipss_col = "#c0392b"; ipss_bg = "#fdecea"; ipss_bd = "#f5c6c2"

    st.markdown(f"""
<div style="background:{ipss_bg}; border:2px solid {ipss_bd}; border-radius:14px;
     padding:1.3rem; text-align:center; margin:0.8rem 0;">
    <div style="font-family:'Fira Code',monospace; font-size:0.6rem; letter-spacing:0.2em;
         text-transform:uppercase; color:{ipss_col}; opacity:0.7; margin-bottom:0.3rem;">TOTAL IPSS</div>
    <div style="font-family:'Libre Baskerville',serif; font-size:2.8rem; color:{ipss_col}; line-height:1; font-weight:700;">{total_ipss}</div>
    <div style="font-size:0.7rem; letter-spacing:0.15em; text-transform:uppercase;
         color:{ipss_col}; font-family:'Fira Code',monospace; margin-top:0.4rem; font-weight:600;">SYMPTOM: {ipss_cat}</div>
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

    route_desc   = {'A': "Model A — Waveform Only", 'C': "Model C — Klinis Lengkap", 'D': "Model D — Klinis Ringkas"}
    route_detail = {'A': "Akurasi rendah. Gunakan jika tidak ada data klinis.",
                    'C': "AUC 0.856 CV · 0.812 External. Rekomendasi utama.",
                    'D': "AUC 0.838 CV. Fallback jika data klinis tidak lengkap."}
    route_color  = {'A': "#c0392b", 'C': "#1a5fa8", 'D': "#0d8a74"}
    route_bg     = {'A': "#fdecea", 'C': "#e8f1fb", 'D': "#e6f6f3"}
    route_bd     = {'A': "#f5c6c2", 'C': "#b8d2f0", 'D': "#86d4c6"}

    st.markdown(f"""
<div style="background:{route_bg[route]}; border:1.5px solid {route_bd[route]}; border-radius:10px; padding:0.8rem 1rem;">
    <div style="font-family:'Fira Code',monospace; font-size:0.58rem; letter-spacing:0.15em; text-transform:uppercase; color:{route_color[route]}; margin-bottom:0.2rem;">Model yang akan digunakan</div>
    <div style="font-size:0.88rem; font-weight:700; color:{route_color[route]};">{route_desc[route]}</div>
    <div style="font-size:0.72rem; color:#4a5568; margin-top:0.2rem;">{route_detail[route]}</div>
</div>
""", unsafe_allow_html=True)

    if route == 'A':
        st.markdown('<div style="background:#fff7ed; border:1.5px solid #fed7aa; border-radius:10px; padding:0.7rem 0.9rem; font-size:0.78rem; color:#b45309; margin-top:0.5rem;">⚠️ <b>Akurasi Terbatas.</b> Lengkapi data IPSS dan klinis untuk hasil lebih akurat.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# BUTTON
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_, col_btn, _ = st.columns([1, 2, 1])
with col_btn:
    run_analysis = st.button("⚡ PROSES ANALISIS NIVA-BOO", type="primary",
                              use_container_width=True, disabled=not waveform_ready)
    if not waveform_ready:
        st.markdown('<div style="text-align:center; font-size:0.72rem; color:#8a9bb0; margin-top:0.3rem;">Upload PDF Uroflowmetry untuk mengaktifkan analisis</div>', unsafe_allow_html=True)


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

    # ── HITUNG SEMUA VARIABEL ────────────────────────────────────────────────
    prob_pct     = result['prob_boo'] * 100
    pred_boo     = result['pred_boo']
    conf         = result['confidence']
    conf_col     = result['confidence_color']
    model_key    = result['model_key']

    display_pct   = prob_pct if pred_boo else (100.0 - prob_pct)
    bar_width_str = f"{display_pct:.1f}"
    bar_gradient  = "linear-gradient(90deg,#c2410c,#ef4444)" if pred_boo else "linear-gradient(90deg,#0d8a74,#22c55e)"
    conf_display  = f"{display_pct:.1f}"

    pred_label    = "BOO" if pred_boo else "Non-BOO"
    pred_label_up = "BOO" if pred_boo else "NON-BOO"
    result_color  = "#c2410c" if pred_boo else "#166534"
    result_text   = "BOO TERDETEKSI" if pred_boo else "NON-BOO"
    boo_color     = "#c2410c" if pred_boo else "#166534"
    bg_color      = "#fff7ed" if pred_boo else "#dcfce7"
    border_color  = "#fed7aa" if pred_boo else "#86efac"
    border_strong = "#f97316" if pred_boo else "#22c55e"

    conf_bg_map   = {"TINGGI": "#dcfce7", "SEDANG": "#fef3c7", "RENDAH": "#fdecea"}
    conf_bd_map   = {"TINGGI": "#86efac", "SEDANG": "#fcd34d", "RENDAH": "#f5c6c2"}
    conf_cl_map   = {"TINGGI": "#166534", "SEDANG": "#b45309", "RENDAH": "#c0392b"}
    conf_bg       = conf_bg_map[conf]
    conf_bd       = conf_bd_map[conf]
    conf_cl       = conf_cl_map[conf]

    interp_title  = "Bladder Outlet Obstruction" if pred_boo else "Tidak Ditemukan Obstruksi"
    interp_body   = (
        f"Pola waveform mengindikasikan obstruksi dengan probabilitas {prob_pct:.1f}%. "
        "Skor IPSS Voiding dan volume prostat mendukung temuan ini."
        if pred_boo else
        "Pola waveform tidak menunjukkan obstruksi signifikan. "
        "Kemungkinan Detrusor Underactivity (DU) atau kondisi lainnya perlu dievaluasi."
    )

    ipss_warn_col  = "#b45309" if total_ipss == 0 else "#1a2233"
    ipss_warn_icon = " ⚠️" if total_ipss == 0 else ""
    qmax_color_rep = "#c0392b" if qmax < 10 else "#1a2233"
    src_label      = get_waveform_source_label(waveform_source == "DTA")

    if result['needs_urodynamics']:
        uds_bg    = "#fef3c7"; uds_bd = "#fcd34d"; uds_cl = "#b45309"
        uds_icon  = "🔄"
        uds_title = "Pemeriksaan UDS Direkomendasikan"; uds_title_cl = "#92400e"
        uds_body  = ("Confidence rendah atau model terbatas. UDS diperlukan untuk konfirmasi diagnostik."
                     if conf == 'RENDAH' or model_key == 'A' else
                     "Temuan borderline. UDS dapat memperkuat keputusan terapi.")
    else:
        uds_bg    = "#dcfce7"; uds_bd = "#86efac"; uds_cl = "#166534"
        uds_icon  = "✅"
        uds_title = "Data NIVA-BOO Cukup Meyakinkan"; uds_title_cl = "#14532d"
        uds_body  = "Untuk saat ini prosedur invasif dapat ditunda berdasarkan confidence analisis."

    # ── Warning data kosong ──────────────────────────────────────────────────
    if model_key == 'C' and sc_storage == 0 and sc_voiding == 0 and sc_qol == 0 and vol_prostat == 0:
        st.markdown('<div style="background:#fff7ed; border:1.5px solid #fed7aa; border-radius:10px; padding:0.8rem 1rem; font-size:0.8rem; color:#b45309; margin-bottom:1rem;">⚠️ <b>Data klinis belum diisi.</b> Skor IPSS dan volume prostat masih 0. Lengkapi data klinis untuk prediksi bermakna.</div>', unsafe_allow_html=True)

    # ── Header Hasil ─────────────────────────────────────────────────────────
    st.markdown(f"""
<div style="background:#ffffff; border:1.5px solid #d1d9e6; border-radius:16px;
     padding:1.2rem 2rem; margin-bottom:1.5rem; box-shadow:0 2px 8px rgba(26,42,67,0.07);">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">
        <div>
            <div style="font-family:'Fira Code',monospace; font-size:0.6rem; letter-spacing:0.2em;
                 text-transform:uppercase; color:#8a9bb0; margin-bottom:0.2rem;">
                Hasil Analisis &middot; {mrn or 'Pasien Baru'}
            </div>
            <div style="font-family:'Libre Baskerville',serif; font-size:1.2rem; color:#1a2233; font-weight:700;">
                {result['model_used']} &middot; {result['model_description']}
            </div>
        </div>
        <div style="font-family:'Fira Code',monospace; font-size:0.65rem; color:#8a9bb0; text-align:right;
             background:#f5f7fa; border:1px solid #d1d9e6; border-radius:8px; padding:0.6rem 0.9rem;">
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
<div style="background:{bg_color}; border:2px solid {border_strong}; border-radius:16px; padding:2rem; text-align:center; box-shadow:0 2px 10px rgba(26,42,67,0.07);">
    <div style="font-family:'Fira Code',monospace; font-size:0.6rem; letter-spacing:0.2em; text-transform:uppercase; color:#8a9bb0; margin-bottom:0.5rem;">
        Probabilitas NIVA-BOO
    </div>
    <div style="font-family:'Libre Baskerville',serif; font-size:2.4rem; color:{boo_color}; font-weight:700; line-height:1; margin:0.4rem 0;">{pred_label}</div>
    <div style="font-family:'Fira Code',monospace; font-size:1.4rem; font-weight:600; color:{boo_color};">
        {prob_pct:.1f}%
    </div>
    <div style="margin:1rem 0; background:#e2e8f0; border-radius:8px; height:8px; overflow:hidden;">
        <div style="width:{bar_width_str}%; height:100%; background:{bar_gradient}; border-radius:8px;"></div>
    </div>
    <div style="display:inline-block; padding:0.35rem 1.1rem; border-radius:20px; font-family:'Fira Code',monospace; font-size:0.68rem; letter-spacing:0.12em; text-transform:uppercase; margin-top:0.4rem; background:{conf_bg}; border:1.5px solid {conf_bd}; color:{conf_cl}; font-weight:600;">
        {pred_label_up} &middot; CONFIDENCE: {conf}
    </div>
</div>
""", unsafe_allow_html=True)

    # ── RC2: Interpretasi ─────────────────────────────────────────────────────
    with rc2:
        st.markdown(f"""
<div style="background:#ffffff; border:1.5px solid {border_color}; border-radius:14px; padding:1.3rem 1.5rem; margin-bottom:0.8rem; box-shadow:0 2px 8px rgba(26,42,67,0.06);">
    <div style="font-family:'Fira Code',monospace; font-size:0.58rem; letter-spacing:0.2em; text-transform:uppercase; color:#8a9bb0; margin-bottom:0.5rem;">
        Interpretasi NIVA-BOO
    </div>
    <div style="font-family:'Libre Baskerville',serif; font-size:1.15rem; color:{boo_color}; margin-bottom:0.5rem; font-weight:700;">
        {interp_title}
    </div>
    <div style="font-size:0.82rem; color:#4a5568; line-height:1.65;">{interp_body}</div>
</div>
<div style="background:#ffffff; border:1.5px solid {border_color}; border-radius:14px; padding:1.3rem 1.5rem; box-shadow:0 2px 8px rgba(26,42,67,0.06);">
    <div style="font-family:'Fira Code',monospace; font-size:0.58rem; letter-spacing:0.2em; text-transform:uppercase; color:#8a9bb0; margin-bottom:0.5rem;">
        Saran Klinis
    </div>
    <div style="font-family:'Libre Baskerville',serif; font-size:1rem; margin-bottom:0.5rem; font-weight:700; color:#1a2233;">
        {result['clinical_recommendation']}
    </div>
    <div style="font-size:0.82rem; color:#4a5568; line-height:1.65;">{result['clinical_detail']}</div>
</div>
""", unsafe_allow_html=True)

    # ── RC3: UDS + Waveform ───────────────────────────────────────────────────
    with rc3:
        st.markdown(f"""
<div style="background:{uds_bg}; border:1.5px solid {uds_bd}; border-radius:12px; padding:1rem 1.2rem; display:flex; align-items:flex-start; gap:0.9rem; margin-bottom:0.8rem; box-shadow:0 2px 8px rgba(26,42,67,0.06);">
    <div style="font-size:1.4rem; flex-shrink:0; margin-top:0.1rem;">{uds_icon}</div>
    <div>
        <div style="font-family:'Fira Code',monospace; font-size:0.58rem; letter-spacing:0.15em; text-transform:uppercase; color:{uds_cl}; margin-bottom:0.3rem;">
            Kebutuhan Urodinamika (UDS)
        </div>
        <div style="font-size:0.84rem; color:{uds_title_cl}; font-weight:700;">{uds_title}</div>
        <div style="font-size:0.75rem; color:#4a5568; margin-top:0.3rem;">{uds_body}</div>
    </div>
</div>
""", unsafe_allow_html=True)

        if waveform_data is not None:
            wave_color = '#c2410c' if pred_boo else '#0d8a74'
            fig2, ax2  = plt.subplots(figsize=(4, 1.5))
            fig2.patch.set_facecolor('#ffffff')
            ax2.set_facecolor('#f5f7fa')
            ax2.plot(waveform_data, color=wave_color, linewidth=1.8, alpha=0.9)
            ax2.fill_between(range(len(waveform_data)), waveform_data, alpha=0.14, color=wave_color)
            ax2.set_xticks([]); ax2.set_yticks([])
            for sp in ax2.spines.values(): sp.set_visible(False)
            ax2.set_xlim(0, len(waveform_data)); ax2.set_ylim(-0.05, 1.05)
            plt.tight_layout(pad=0.3)
            st.pyplot(fig2, use_container_width=True)
            plt.close()

        st.markdown(f"""
<div style="font-family:'Fira Code',monospace; font-size:0.6rem; color:#8a9bb0; margin-top:0.4rem; text-align:center;">
    Sumber: {src_label}
</div>
""", unsafe_allow_html=True)

    # ── Laporan ───────────────────────────────────────────────────────────────
    st.markdown(f"""
<div style="background:#ffffff; border:1.5px solid #d1d9e6; border-radius:16px; padding:2rem 2.5rem; margin-top:1.5rem; box-shadow:0 4px 16px rgba(26,42,67,0.08);">
    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">
        <div>
            <div style="font-family:'Fira Code',monospace; font-size:0.6rem; letter-spacing:0.2em; text-transform:uppercase; color:#8a9bb0; margin-bottom:0.3rem;">
                Laporan Analisis NIVA-BOO
            </div>
            <div style="font-family:'Libre Baskerville',serif; font-size:1.25rem; color:#1a2233; font-weight:700;">
                Non-Invasive Analysis for Bladder Outlet Obstruction
            </div>
        </div>
        <div style="text-align:right; font-size:0.8rem; color:#8a9bb0;
             background:#f5f7fa; border:1px solid #d1d9e6; border-radius:8px; padding:0.6rem 0.9rem;">
            <div style="font-weight:700; color:#4a5568;">{mrn or '—'}</div>
            <div>Usia {usia} tahun</div>
        </div>
    </div>
    <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:1.2rem; margin-top:1.5rem; padding-top:1.5rem; border-top:1.5px solid #eef1f6;">
        <div style="background:#f5f7fa; border:1px solid #d1d9e6; border-radius:10px; padding:1rem;">
            <div style="font-family:'Fira Code',monospace; font-size:0.58rem; letter-spacing:0.2em; text-transform:uppercase; color:#8a9bb0; margin-bottom:0.4rem;">No. RM Pasien</div>
            <div style="font-size:1rem; font-weight:700; color:#1a2233;">{mrn or '—'}</div>
            <div style="font-size:0.7rem; color:#8a9bb0; margin-top:0.2rem;">Usia: {usia} thn</div>
        </div>
        <div style="background:#f5f7fa; border:1px solid #d1d9e6; border-radius:10px; padding:1rem;">
            <div style="font-family:'Fira Code',monospace; font-size:0.58rem; letter-spacing:0.2em; text-transform:uppercase; color:#8a9bb0; margin-bottom:0.4rem;">Profil LUTS</div>
            <div style="font-size:1rem; font-weight:700; color:{ipss_warn_col};">IPSS: {total_ipss} ({ipss_cat}){ipss_warn_icon}</div>
            <div style="font-size:0.7rem; color:#8a9bb0; margin-top:0.2rem;">Voiding: {sc_voiding} · Storage: {sc_storage}</div>
        </div>
        <div style="background:#f5f7fa; border:1px solid #d1d9e6; border-radius:10px; padding:1rem;">
            <div style="font-family:'Fira Code',monospace; font-size:0.58rem; letter-spacing:0.2em; text-transform:uppercase; color:#8a9bb0; margin-bottom:0.4rem;">Uroflowmetry</div>
            <div style="font-size:1rem; font-weight:700; color:{qmax_color_rep};">Qmax: {qmax:.1f} mL/s</div>
            <div style="font-size:0.7rem; color:#8a9bb0; margin-top:0.2rem;">PVR: {pvr} mL</div>
        </div>
        <div style="background:{bg_color}; border:1.5px solid {border_color}; border-radius:10px; padding:1rem;">
            <div style="font-family:'Fira Code',monospace; font-size:0.58rem; letter-spacing:0.2em; text-transform:uppercase; color:#8a9bb0; margin-bottom:0.4rem;">Analisis NIVA-BOO</div>
            <div style="font-size:1rem; font-weight:700; color:{result_color};">{result_text}</div>
            <div style="font-size:0.7rem; color:#8a9bb0; margin-top:0.2rem;">Confidence: {conf_display}%</div>
        </div>
    </div>
    <div style="margin-top:1.2rem; padding-top:1rem; border-top:1px solid #eef1f6; font-size:0.68rem; color:#b0bdd0; font-family:'Fira Code',monospace;">
        Model: {result['model_used']} &middot; Threshold: {result['threshold']:.3f} (Youden's J) &middot;
        Waveform: {src_label} &middot;
        ⚠️ Alat bantu skrining penelitian — bukan pengganti penilaian spesialis
    </div>
</div>
""", unsafe_allow_html=True)

    if model_key == 'A':
        st.markdown("""
<div style="background:#fdecea; border:1.5px solid #f5c6c2; border-radius:10px; padding:0.9rem 1rem; margin-top:1rem; font-size:0.8rem; color:#c0392b;">
⚠️ <b>PERHATIAN:</b> Analisis menggunakan Model A (Waveform Only) dengan AUC 0.52 —
hampir setara random. Hasil ini <b>tidak boleh digunakan</b> sebagai dasar keputusan klinis.
Lengkapi data IPSS dan parameter klinis untuk akurasi yang bermakna.
</div>
""", unsafe_allow_html=True)

elif not waveform_ready:
    st.markdown("""
<div style="text-align:center; padding:4rem 2rem; background:#ffffff; border:1.5px dashed #d1d9e6; border-radius:18px; margin-top:1rem;">
    <div style="font-size:3rem; margin-bottom:1rem;">🔬</div>
    <div style="font-family:'Libre Baskerville',serif; font-size:1.3rem; color:#4a5568; font-weight:700;">
        Upload PDF Uroflowmetry untuk memulai analisis
    </div>
    <div style="font-size:0.82rem; color:#8a9bb0; margin-top:0.5rem;">
        File .DTA opsional untuk akurasi lebih tinggi
    </div>
</div>
""", unsafe_allow_html=True)