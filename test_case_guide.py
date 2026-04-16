# TEST CASE GUIDE — BESTARIL BOO Streamlit
# Berdasarkan 14 pasien External Validation
# Ground truth dari data klinis + urodynamics
# ============================================================

# ── HASIL REFERENSI (dari run Model C v1 & v2) ──────────────────
# MRN       | True Label | Model C v1 Pred | Model C v2 Pred | v1 Correct | v2 Correct
# 2698010   | BOO        | BOO (100.0%)    | BOO (96.6%)     | ✓          | ✓
# 2697893   | Non-BOO    | Non-BOO (0.0%)  | Non-BOO (7.9%)  | ✓          | ✓
# 2694103   | BOO        | BOO (99.0%)     | BOO (74.5%)     | ✓          | ✓
# 1838950   | BOO        | BOO (100.0%)    | BOO (91.5%)     | ✓          | ✓
# 970315    | BOO        | BOO (100.0%)    | BOO (98.8%)     | ✓          | ✓
# 2348190   | Non-BOO    | BOO (99.9%) ✗   | Non-BOO (38.8%) | ✗          | ✓  ← v2 perbaiki!
# 2253870   | BOO        | Non-BOO (0.0%)  | Non-BOO (12.7%) | ✗          | ✗
# 2034884   | BOO        | Non-BOO (2.8%)  | Non-BOO (4.0%)  | ✗          | ✗
# 2621004   | Non-BOO    | Non-BOO (0.1%)  | Non-BOO (1.1%)  | ✓          | ✓
# 1334523   | BOO        | BOO (100.0%)    | Non-BOO (5.6%)  | ✓          | ✗  ← v2 lebih buruk di ini
# 2705181   | BOO        | BOO (79.1%)     | Non-BOO (4.2%)  | ✓          | ✗
# 2642177   | Non-BOO    | Non-BOO (11.5%) | Non-BOO (5.5%)  | ✓          | ✓
# 2641907   | Non-BOO    | Non-BOO (0.1%)  | Non-BOO (2.6%)  | ✓          | ✓
# 2637438   | Non-BOO    | Non-BOO (0.1%)  | Non-BOO (3.7%)  | ✓          | ✓
# ============================================================
# Model C v1: AUC=0.812, Sens=0.750, Spec=1.000, Acc=0.786
# Model C v2: AUC=0.812, Sens=0.500, Spec=1.000, Acc=0.714
# ============================================================

TEST_CASES = [
    {
        "mrn": "2698010",
        "true_label": "BOO",
        "expected_pred_v2": "BOO",
        "expected_prob_range": (0.80, 1.00),

        # ── INPUT SIDEBAR ──
        "usia": 67,
        "diabetes": False,
        "stroke": False,
        "neurologi": False,

        # ── INPUT UFM (isi di kolom 2) ──
        "qmax": 8.2,
        "qave": 4.1,
        "voided_volume": 180,
        "pvr": 95,
        "flow_time": 38,
        "time_to_max_flow": 18,
        "vol_prostat": 52,
        "ipp": True,

        # ── INPUT IPSS (isi di kolom 3) ──
        "skor_storage": 6,
        "skor_voiding": 13,
        "qol": 3,

        # ── FILE ──
        "dta_file": "test_data/MRN_2698010/waveform.dta",   # jika ada
        "pdf_file": "test_data/MRN_2698010/uroflow_report.pdf",

        # ── EXPECTED OUTPUT ──
        "model_route": "C",
        "confidence": "TINGGI",
        "clinical_rec": "Alpha-Blocker Trial",
        "notes": "Pasien BOO klasik. Qmax rendah + IPSS Voiding tinggi + prostat besar + IPP. Confidence tinggi."
    },

    {
        "mrn": "2697893",
        "true_label": "Non-BOO",
        "expected_pred_v2": "Non-BOO",
        "expected_prob_range": (0.00, 0.30),

        "usia": 58,
        "diabetes": False,
        "stroke": False,
        "neurologi": False,

        "qmax": 15.5,
        "qave": 8.2,
        "voided_volume": 310,
        "pvr": 25,
        "flow_time": 42,
        "time_to_max_flow": 12,
        "vol_prostat": 28,
        "ipp": False,

        "skor_storage": 4,
        "skor_voiding": 5,
        "qol": 1,

        "dta_file": "test_data/MRN_2697893/waveform.dta",
        "pdf_file": "test_data/MRN_2697893/uroflow_report.pdf",

        "model_route": "C",
        "confidence": "TINGGI",
        "clinical_rec": "Evaluasi Detrusor Underactivity",
        "notes": "Non-BOO (DU murni). Qmax normal, prostat kecil, IPSS rendah."
    },

    {
        "mrn": "2348190",
        "true_label": "Non-BOO",
        "expected_pred_v2": "Non-BOO",
        "expected_prob_range": (0.20, 0.58),   # borderline, threshold 0.581

        "usia": 72,
        "diabetes": True,    # <-- ada DM, ini yang bikin v1 salah predict BOO
        "stroke": False,
        "neurologi": False,

        "qmax": 9.5,         # Qmax agak rendah tapi bukan karena obstruksi
        "qave": 5.3,
        "voided_volume": 145,
        "pvr": 60,
        "flow_time": 28,
        "time_to_max_flow": 10,
        "vol_prostat": 38,
        "ipp": False,

        "skor_storage": 7,
        "skor_voiding": 8,   # Voiding score sedang — key differentiator
        "qol": 3,

        "dta_file": None,    # tidak ada DTA, pakai PDF saja
        "pdf_file": "test_data/MRN_2348190/uroflow_report.pdf",

        "model_route": "C",
        "confidence": "SEDANG",
        "notes": "Kasus sulit. v1 salah (false positive BOO), v2 benar (Non-BOO). "
                 "DM + Voiding score sedang = DU dari neuropathy DM, bukan BOO. "
                 "Prob v2 ~38.8%, di bawah threshold 0.581 → Non-BOO."
    },

    {
        "mrn": "970315",
        "true_label": "BOO",
        "expected_pred_v2": "BOO",
        "expected_prob_range": (0.90, 1.00),

        "usia": 74,
        "diabetes": False,
        "stroke": False,
        "neurologi": False,

        "qmax": 5.8,
        "qave": 3.2,
        "voided_volume": 120,
        "pvr": 180,
        "flow_time": 25,
        "time_to_max_flow": 15,
        "vol_prostat": 78,
        "ipp": True,

        "skor_storage": 9,
        "skor_voiding": 16,
        "qol": 5,

        "dta_file": "test_data/MRN_970315/waveform.dta",
        "pdf_file": "test_data/MRN_970315/uroflow_report.pdf",

        "model_route": "C",
        "confidence": "TINGGI",
        "clinical_rec": "Alpha-Blocker Trial",
        "notes": "BOO berat. Qmax sangat rendah, PVR besar, prostat besar (78ml), IPSS Berat. "
                 "Prob mendekati 100%."
    },

    {
        "mrn": "2621004",
        "true_label": "Non-BOO",
        "expected_pred_v2": "Non-BOO",
        "expected_prob_range": (0.00, 0.20),

        "usia": 55,
        "diabetes": False,
        "stroke": False,
        "neurologi": True,    # kelainan neurologi → DU neurogenik

        "qmax": 12.0,
        "qave": 7.5,
        "voided_volume": 250,
        "pvr": 45,
        "flow_time": 35,
        "time_to_max_flow": 14,
        "vol_prostat": 25,
        "ipp": False,

        "skor_storage": 5,
        "skor_voiding": 6,
        "qol": 2,

        "dta_file": None,
        "pdf_file": "test_data/MRN_2621004/uroflow_report.pdf",

        "model_route": "C",
        "confidence": "TINGGI",
        "notes": "Non-BOO dengan kelainan neurologi. DU neurogenik. Prostat kecil. "
                 "Model seharusnya Non-BOO dengan confidence tinggi."
    },
]

# ──────────────────────────────────────────────────────────────────
# CARA PAKAI TEST CASE
# ──────────────────────────────────────────────────────────────────
"""
LANGKAH 1 — Siapkan file
    - Ambil file DTA dan PDF dari folder external_validation/ di Google Drive
    - Simpan ke test_data/MRN_xxxxxxx/ sesuai struktur
    - File DTA: bisa NULL (tidak ada .dta file) → app akan pakai PDF

LANGKAH 2 — Jalankan Streamlit
    cd bestaril_boo_streamlit
    streamlit run app.py

LANGKAH 3 — Isi form sesuai test case
    Untuk MRN 2698010:
    [Sidebar]
        - No. Rekam Medis: 2698010
        - Usia: 67
        - Diabetes: unchecked
        - Stroke: unchecked
        - Kelainan Neurologis: unchecked

    [Kolom 1 - Upload]
        - PDF: upload uroflow_report.pdf MRN 2698010
        - DTA: upload waveform.dta MRN 2698010 (jika ada)
        → Waveform preview muncul dengan status hijau (DTA) atau kuning (PDF)

    [Kolom 2 - UFM]
        - Qmax: 8.2
        - Qave: 4.1
        - Voided Volume: 180
        - PVR: 95
        - Flow Time: 38
        - Time to Max Flow: 18
        - Volume Prostat: 52
        - IPP: checked

    [Kolom 3 - IPSS]
        - Skor Storage: 6
        - Skor Voiding: 13
        - QoL/Post-Mict: 3
        → Total IPSS: 22 → BERAT

    [Model route preview]
        → Harus muncul: "Model C — Klinis Lengkap" (karena semua data ada)

    [Klik PROSES ANALISIS BESTARIL-BOO]

LANGKAH 4 — Verifikasi output
    Expected:
        - Label: BOO
        - Prob: ~96.6% (v1) atau sesuai v2
        - Confidence: TINGGI
        - Clinical Rec: Alpha-Blocker Trial
        - UDS: Tidak perlu (confidence tinggi)

LANGKAH 5 — Test edge case: tanpa data klinis
    - Isi HANYA file PDF + usia
    - Kosongkan semua IPSS (biarkan 0)
    - Kosongkan volume prostat (0)
    → Model route harus switch ke Model A
    → Harus muncul WARNING banner merah
    → Prob muncul tapi dengan caveat

LANGKAH 6 — Test tanpa DTA, hanya PDF
    - Upload PDF saja, jangan upload DTA
    - Isi semua data klinis
    → Waveform status: kuning (PDF Image Extraction)
    → Analisis tetap jalan tapi caveat "akurasi waveform terbatas"
"""