"""
utils/waveform_extractor.py — v2.3
Ekstrak waveform uroflowmetry dari file .DTA atau PDF Laborie.

Perbaikan v2.3:
  - Boundary detection via RED PIXEL SPIKE: cari axis horizontal merah
    (row dengan >> rata-rata pixel merah = zero-flow line)
  - y_top  = baris pertama dengan pixel merah (top of flow plot)
  - y_zero = baris spike merah (= sumbu 0 ml/s)
  - Abaikan pixel merah di dekat y_zero (axis line sendiri, bukan flow)
  - Best-run selection + fill small gaps
  - UFM params dari teks PDF (Qmax, Qave, VV, PVR, Flow Time, TtMF)
"""

import struct, re, io
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter

TARGET_LENGTH = 280


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def extract_from_dta(filepath):
    """Baca file .DTA binary Laborie. Return waveform [280] atau None."""
    try:
        with open(filepath, 'rb') as f:
            raw = f.read()

        if b'UDS' not in raw[:10]:
            return None

        all_labels = [b'Flow\x00', b'Volume\x00', b'CALQ\x00',
                      b'Pdet\x00', b'Pves\x00', b'Pabd\x00']
        all_pos = sorted(set(
            m.start() for lb in all_labels
            for m in re.finditer(re.escape(lb), raw)
        ))

        best_flow = None; best_score = -1
        for fp in [m.start() for m in re.finditer(re.escape(b'Flow\x00'), raw)]:
            next_ch = next((p for p in all_pos if p > fp + 10), len(raw))
            pos = fp + 5
            while pos < len(raw) and raw[pos] != 0: pos += 1
            pos += 1
            data = raw[pos:next_ch]; n = len(data) // 2
            if n < 10: continue
            vals = np.array(struct.unpack(f'<{n}H', data[:n*2]), dtype=float)
            vals[vals > 600] = 0; flow_mls = vals * 0.1
            runs = []; in_run = False
            for i, v in enumerate(flow_mls):
                if v > 0.1 and not in_run: in_run = True; s = i
                elif v <= 0.1 and in_run: in_run = False; runs.append((s, i))
            if in_run: runs.append((s, len(flow_mls)))
            valid = [(s,e) for s,e in runs if (e-s)>=10 and 1.0<=flow_mls[s:e].max()<=60.0]
            if not valid: continue
            best_run = max(valid, key=lambda x: x[1]-x[0])
            score = best_run[1] - best_run[0]
            if score > best_score:
                best_score = score
                best_flow = flow_mls[best_run[0]:best_run[1]]

        return _preprocess_wave(best_flow) if best_flow is not None else None

    except Exception as e:
        print(f"[DTA] Error: {e}"); return None


def extract_from_pdf(filepath):
    """
    Ekstrak waveform + UFM params dari PDF Laborie Uroflow Report.
    Return: (waveform [280] or None, ufm_params dict)
    """
    import fitz
    ufm_params = {
        'Qmax_ml_s': np.nan, 'Qave_ml_s': np.nan,
        'Voided_Volume_ml': np.nan, 'PVR_ml': np.nan,
        'Flow_Time_s': np.nan, 'Time_to_Max_Flow_s': np.nan,
    }
    try:
        doc       = fitz.open(str(filepath))
        full_text = "".join(p.get_text() for p in doc)
        _extract_ufm_params(full_text, ufm_params)
        waveform  = _extract_waveform_from_pdf(doc)
        doc.close()
        return waveform, ufm_params
    except Exception as e:
        print(f"[PDF] Error: {e}"); return None, ufm_params


def get_waveform_source_label(has_dta: bool) -> str:
    return "DTA Binary (Akurasi Tinggi)" if has_dta else "PDF Image Extraction (Estimasi)"


# ─────────────────────────────────────────────────────────────────────────────
# UFM PARAMETER EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_ufm_params(text, params):
    patterns = {
        'Qmax_ml_s':          [r'Maximum\s+[Ff]low[^:=\n]*[:\s=]+(\d+\.?\d*)',
                                r'Qmax[^:=\n]*[:\s=]+(\d+\.?\d*)'],
        'Qave_ml_s':          [r'Average\s+[Ff]low[^:=\n]*[:\s=]+(\d+\.?\d*)',
                                r'Qave[^:=\n]*[:\s=]+(\d+\.?\d*)'],
        'Voided_Volume_ml':   [r'Voided\s+volume[^:=\n]*[:\s=]+(\d+\.?\d*)',
                                r'VV[^:=\n]*[:\s=]+(\d+\.?\d*)'],
        'PVR_ml':             [r'PVR[^:=\n]*[:\s=]+(\d+\.?\d*)',
                                r'[Rr]esidual[^:=\n]*[:\s=]+(\d+\.?\d*)'],
        'Flow_Time_s':        [r'[Ff]low\s*time[^:=\n]*[:\s=]+(\d+:?\d*\.?\d*)'],
        'Time_to_Max_Flow_s': [r'[Tt]ime\s*to\s*max[^:=\n]*[:\s=]+(\d+:?\d*\.?\d*)'],
    }
    for key, pats in patterns.items():
        for pat in pats:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                raw = m.group(1).strip()
                try:
                    params[key] = (float(raw.split(':')[0])*60 + float(raw.split(':')[1])
                                   if ':' in raw else float(raw))
                    break
                except ValueError:
                    pass


# ─────────────────────────────────────────────────────────────────────────────
# WAVEFORM EXTRACTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def _red_mask(R, G, B):
    """True untuk pixel merah Laborie (flow curve)."""
    return (
        (R.astype(np.int32) > 130) &
        (G.astype(np.int32) < 120) &
        (B.astype(np.int32) < 120) &
        (R.astype(np.int32) - G.astype(np.int32) > 40)
    )


def _detect_plot_bounds(arr):
    """
    Deteksi batas area plot flow dari pixel merah.

    Key insight: baris zero-flow (sumbu X bawah) di Laborie adalah garis
    merah horizontal tipis — row dengan JAUH lebih banyak pixel merah
    dibanding rata-rata (spike dalam distribusi row_red).

    Return: y_top, y_zero, x_left, x_right
    """
    h, w = arr.shape[:2]
    R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    rm = _red_mask(R, G, B)
    row_red = rm.sum(axis=1)

    # Baris dengan pixel merah (minimal) — area plot flow
    HEADER_SKIP = int(h * 0.05)   # skip header
    row_red_clipped = row_red.copy()
    row_red_clipped[:HEADER_SKIP] = 0

    flow_rows = np.where(row_red_clipped > 3)[0]

    if len(flow_rows) < 10:
        # Fallback
        return int(h*0.10), int(h*0.55), int(w*0.05), int(w*0.95)

    # Deteksi spike = axis line merah horizontal (zero-flow line)
    nz_vals = row_red[flow_rows]
    mean_nz  = nz_vals.mean()
    spike_th = max(mean_nz * 3.5, mean_nz + 50)
    spike_rows = flow_rows[nz_vals > spike_th]

    if len(spike_rows) > 0:
        # y_zero = spike pertama dalam area flow
        y_zero = int(spike_rows[0])
        # y_top  = baris pertama dengan pixel merah, di atas y_zero
        above_zero = flow_rows[flow_rows < y_zero]
        y_top = max(0, int(above_zero.min()) - 20) if len(above_zero) > 0 else int(h*0.10)
    else:
        # Tidak ada spike — gunakan min/max dari flow rows
        y_top  = max(0, int(flow_rows.min()) - 10)
        y_zero = min(h-1, int(flow_rows.max()) + 10)

    # x boundaries dari kolom yang punya pixel merah dalam area plot
    col_red = rm[y_top:y_zero, :].sum(axis=0)
    red_cols = np.where(col_red > 2)[0]
    x_left  = int(red_cols.min()) if len(red_cols) > 5 else int(w*0.05)
    x_right = int(red_cols.max()) if len(red_cols) > 5 else int(w*0.95)

    return y_top, y_zero, x_left, x_right


def _extract_flow_signal(arr, y_top, y_zero, x_left, x_right):
    """
    Per kolom: cari y-minimum dari pixel merah dalam area [y_top, y_zero].
    Abaikan pixel di dekat y_zero (itu adalah axis line sendiri).
    Return: flow_vals array (0–1, unnormalized)
    """
    plot_h = y_zero - y_top
    if plot_h < 20:
        return np.zeros(x_right - x_left, dtype=np.float32)

    R = arr[y_top:y_zero, x_left:x_right, 0]
    G = arr[y_top:y_zero, x_left:x_right, 1]
    B = arr[y_top:y_zero, x_left:x_right, 2]

    # Batas abaikan: 90% ke bawah dari area plot = area axis line
    ignore_below = int(plot_h * 0.88)
    n_cols = x_right - x_left
    flow_vals = np.zeros(n_cols, dtype=np.float32)

    for xi in range(n_cols):
        rm = _red_mask(R[:, xi], G[:, xi], B[:, xi])
        # Abaikan area dekat axis line
        rm[ignore_below:] = False
        red_rows = np.where(rm)[0]
        if len(red_rows) >= 2:
            y_top_red = int(red_rows.min())
            flow_vals[xi] = max(0.0, min(1.0, (plot_h - y_top_red) / plot_h))

    return flow_vals


def _process_flow_signal(flow_raw):
    """
    1. Median filter (hapus spike dari label teks)
    2. Savitzky-Golay smooth
    3. Ambil continuous run terpanjang
    4. Normalize 0–1 + resample ke 280
    """
    flow_med = median_filter(flow_raw.astype(np.float32), size=7)
    win = min(31, max(5, (len(flow_med) // 10) | 1))
    flow_smooth = np.clip(savgol_filter(flow_med, win, 3), 0, 1).astype(np.float32)

    seg = _best_run(flow_smooth, threshold=0.04, min_gap=20)
    if seg is None or len(seg) < 15:
        return None

    mx = seg.max()
    if mx <= 0:
        return None

    return _resample(seg / mx, TARGET_LENGTH)


def _best_run(signal, threshold=0.04, min_gap=10):
    """Ambil segmen kontinu terpanjang di atas threshold (isi gap kecil)."""
    n = len(signal)
    above = (signal > threshold).copy()
    i = 0
    while i < n:
        if not above[i]:
            j = i
            while j < n and not above[j]: j += 1
            if (j - i) < min_gap and i > 0 and j < n:
                above[i:j] = True
            i = j + 1
        else:
            i += 1

    runs = []; in_run = False
    for i, v in enumerate(above):
        if v and not in_run: in_run = True; s = i
        elif not v and in_run: in_run = False; runs.append((s, i))
    if in_run: runs.append((s, n))
    if not runs: return None

    bs, be = max(runs, key=lambda x: x[1] - x[0])
    return signal[max(0,bs-5):min(n,be+5)].copy()


def _resample(signal, target):
    if len(signal) == target: return signal.astype(np.float32)
    f = interp1d(np.linspace(0,1,len(signal)), signal,
                 kind='linear', fill_value=0.0, bounds_error=False)
    return f(np.linspace(0,1,target)).astype(np.float32)


def _preprocess_wave(flow_mls, target=TARGET_LENGTH):
    """Smooth + normalize + resample untuk input DTA."""
    if flow_mls is None or len(flow_mls) < 10: return None
    win = min(11, len(flow_mls) if len(flow_mls)%2==1 else len(flow_mls)-1)
    flow_s = np.clip(savgol_filter(flow_mls, win, 3), 0, None) if win >= 5 else flow_mls.copy()
    mx = flow_s.max()
    if mx <= 0: return None
    return _resample(flow_s / mx, target)


# ─────────────────────────────────────────────────────────────────────────────
# PDF → IMAGE → WAVEFORM
# ─────────────────────────────────────────────────────────────────────────────

def _extract_waveform_from_pdf(doc):
    """Coba embedded image dulu, fallback ke rendered page."""
    from PIL import Image

    # Cari embedded PNG terbesar
    best_bytes = None; best_area = 0
    for page in doc:
        for img_info in page.get_images(full=True):
            info = doc.extract_image(img_info[0])
            iw, ih = info['width'], info['height']
            area = iw * ih
            if area > best_area and iw > 400 and ih > 200 and iw > ih * 0.5:
                best_area  = area
                best_bytes = info['image']

    if best_bytes is not None:
        try:
            arr = np.array(Image.open(io.BytesIO(best_bytes)).convert('RGB'), dtype=np.uint8)
            h, w = arr.shape[:2]
            print(f"[PDF Image] Embedded PNG: {w}×{h}")
            result = _waveform_from_array(arr)
            if result is not None:
                return result
            print("[PDF Image] Extraction failed, trying page render")
        except Exception as e:
            print(f"[PDF Image] Error: {e}")

    # Fallback: render halaman
    return _waveform_from_rendered_page(doc)


def _waveform_from_array(arr):
    """Core pipeline: array RGB → waveform [280]."""
    h, w = arr.shape[:2]
    y_top, y_zero, x_left, x_right = _detect_plot_bounds(arr)
    plot_h = y_zero - y_top
    print(f"[PDF Image] Plot: y=[{y_top},{y_zero}] ({plot_h}px) x=[{x_left},{x_right}]")

    if plot_h < 30 or (x_right - x_left) < 50:
        return None

    flow_raw = _extract_flow_signal(arr, y_top, y_zero, x_left, x_right)
    n_nz = (flow_raw > 0.02).sum()
    print(f"[PDF Image] Non-zero: {n_nz}/{len(flow_raw)}, max={flow_raw.max():.3f}")

    if n_nz < 15:
        return None

    return _process_flow_signal(flow_raw)


def _waveform_from_rendered_page(doc):
    """Fallback: render PDF halaman 1 pada resolusi tinggi."""
    import fitz
    from PIL import Image
    try:
        page = doc[0]
        mat  = fitz.Matrix(4.0, 4.0)
        pix  = page.get_pixmap(matrix=mat)
        arr  = np.array(Image.open(io.BytesIO(pix.tobytes("png"))).convert('RGB'), dtype=np.uint8)
        h, w = arr.shape[:2]
        print(f"[PDF Render] Rendered: {w}×{h}")
        return _waveform_from_array(arr)
    except Exception as e:
        print(f"[PDF Render] Error: {e}"); return None