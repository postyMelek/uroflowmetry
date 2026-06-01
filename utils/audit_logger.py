"""
utils/audit_logger.py — Append-only prediction audit log untuk NIVA-BOO CDSS
Setiap prediksi dicatat dengan UUID, tidak bisa dimodifikasi.
"""

import uuid, hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_FILE = LOG_DIR / "prediction_audit.jsonl"


def log_prediction(boo_probability, model_version, input_data, user_id="anonymous"):
    LOG_DIR.mkdir(exist_ok=True)
    prediction_id = str(uuid.uuid4())
    input_hash = hashlib.sha256(
        json.dumps(input_data, sort_keys=True, default=str).encode()
    ).hexdigest()
    prob = float(boo_probability)

    entry = {
        "prediction_id": prediction_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "input_hash": input_hash,
        "boo_probability": round(prob, 4),
        "boo_label": "BOO" if prob >= 0.5 else "Non-BOO",
        "confidence_zone": "uncertain" if 0.40 <= prob <= 0.60 else "confident",
        "user_id": user_id,
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return prediction_id


def load_audit_log():
    if not LOG_FILE.exists():
        return []
    entries = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries