#!/usr/bin/env python3
# video_processor.py — Full untrimmed updated version
# Implements multi-pipeline OCR readers with robust formatting & JSON-safe report generation.

import os
import uuid
import threading
import ffmpeg
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Tuple
from ultralytics import YOLO
import yaml
from collections import deque, Counter, defaultdict
from database import save_test_result
import time
import math
import logging

# -------------------- LOGGING / DEBUG -------------------- #
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
DEBUG = True

def _dprint(*args, **kwargs):
    if DEBUG:
        logging.debug(" ".join(str(a) for a in args), **kwargs)

# -------------------- PATHS / CONFIG -------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(BASE_DIR, "lcd_ocr_model.pt")
YAML_PATH = os.path.join(BASE_DIR, "data.yaml")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Load model and class names
try:
    MODEL = YOLO(MODEL_PATH)
except Exception as e:
    logging.warning("YOLO model load failed: %s", e)
    MODEL = None

if os.path.exists(YAML_PATH):
    with open(YAML_PATH, "r") as f:
        DATA_CFG = yaml.safe_load(f)
    CLASS_NAMES = DATA_CFG.get("names", [])
else:
    CLASS_NAMES = []

# -------------------- GLOBALS -------------------- #
TASK_PROGRESS: Dict[str, Dict] = {}
SESSIONS: Dict[str, Dict] = {}  # session_id -> {meta, series, report}

# -------------------- HELPERS -------------------- #
def reset_progress(task_id: str) -> None:
    TASK_PROGRESS[task_id] = {"status": "pending", "progress": 0, "phase": "queued", "message": ""}

def get_progress(task_id: str) -> Dict:
    return TASK_PROGRESS.get(task_id, {"status": "unknown", "progress": 0, "phase": "unknown"})

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def clamp(val, minv=None, maxv=None):
    if val is None:
        return None
    try:
        if minv is not None and val < minv:
            return minv
        if maxv is not None and val > maxv:
            return maxv
    except Exception:
        return val
    return val

# -------------------- SESSION -------------------- #
def _init_session(session_id: str, meta: Dict) -> None:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "meta": {
                "prop": meta.get("prop", ""),
                "motor": meta.get("motor", ""),
                "esc": meta.get("esc", ""),
                "voltage": float(meta.get("voltage")) if meta.get("voltage") not in [None, ""] else None,
            },
            # series holds last processed dataframe per type
            "series": {"current": None, "thrust": None, "rpm": None},
            "report": None,
        }
    else:
        for k in ["prop", "motor", "esc", "voltage"]:
            v = meta.get(k)
            if v not in [None, ""]:
                if k == "voltage":
                    try:
                        v = float(v)
                    except (ValueError, TypeError):
                        v = None
                SESSIONS[session_id]["meta"][k] = v

# -------------------- FRAME EXTRACTION -------------------- #
def _ffmpeg_extract(video_path: str, out_dir: str, fps: int) -> None:
    # Use fps_mode=vfr and qscale to ensure frames saved
    try:
        (
            ffmpeg
            .input(video_path)
            .filter("fps", fps=fps)
            .output(os.path.join(out_dir, "frame_%06d.jpg"), start_number=0, qscale=2, fps_mode="vfr")
            .overwrite_output()
            .run(quiet=not DEBUG)
        )
    except Exception as e:
        _dprint("ffmpeg extract failed:", e)
        raise

def _opencv_fallback_extract(video_path: str, out_dir: str, fps: int) -> None:
    cap = cv2.VideoCapture(video_path)
    real_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 1
    ratio = max(1.0, real_fps / float(fps))
    frame_idx, saved = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if int(round(frame_idx % ratio)) == 0:
            cv2.imwrite(os.path.join(out_dir, f"frame_{saved:06d}.jpg"), frame)
            saved += 1
        frame_idx += 1
    cap.release()

def extract_frames_custom(task_id: str, video_path: str, fps: int) -> List[str]:
    frames_dir = os.path.join(RESULT_DIR, f"{task_id}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    TASK_PROGRESS[task_id].update({"phase": "extracting", "message": f"Extracting frames ({fps} FPS)..."})
    try:
        _ffmpeg_extract(video_path, frames_dir, fps=fps)
    except Exception as e:
        TASK_PROGRESS[task_id].update({"message": f"FFmpeg failed, fallback OpenCV. ({e})"})
        _opencv_fallback_extract(video_path, frames_dir, fps=fps)

    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")])
    if len(frame_files) < 5:
        # retry with OpenCV if ffmpeg created too few
        _opencv_fallback_extract(video_path, frames_dir, fps=fps)
        frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")])

    # if extremely large number of frames, downsample to reasonable size to avoid UI lockups
    MAX_FRAMES = 3000  # keep processing tractable; adjust if you want
    if len(frame_files) > MAX_FRAMES:
        step = math.ceil(len(frame_files) / MAX_FRAMES)
        frame_files = frame_files[::step]
        TASK_PROGRESS[task_id].update({"message": f"Downsampled frames to {len(frame_files)} to keep processing responsive."})

    _dprint(f"[DEBUG] Extracted {len(frame_files)} frames at {fps} FPS")
    return frame_files

# -------------------- PREPROCESS -------------------- #
def preprocess_frame(img: Optional[np.ndarray], invert: bool = False) -> Optional[np.ndarray]:
    if img is None:
        return None
    h, w = img.shape[:2]
    # rotate tall frames to landscape
    if h > w * 1.15:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if invert:
        gray = cv2.bitwise_not(gray)
    # noise reduction and contrast
    den = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enh = clahe.apply(den)
    thr = cv2.adaptiveThreshold(enh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    k = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, k)
    blur = cv2.GaussianBlur(morph, (0, 0), 1.5)
    sharp = cv2.addWeighted(morph, 1.4, blur, -0.4, 0)
    # return BGR for model expecting 3 channels
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

# -------------------- OCR CORE -------------------- #
def _predict_reading(img: np.ndarray, conf: float) -> Tuple[str, float]:
    """
    Predicts character sequence using the YOLO model.
    Returns (reading_string, avg_confidence).
    This function assumes MODEL is loaded and CLASS_NAMES is present.
    """
    if MODEL is None:
        return "", 0.0

    # MODEL.predict can accept numpy images directly
    results = MODEL.predict(img, conf=conf, verbose=False)
    if not results:
        return "", 0.0
    res0 = results[0]
    boxes = []
    try:
        for b in res0.boxes:
            cls_idx = int(b.cls[0].item())
            ch = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else ""
            # Keep only characters that matter (digits, '.', '-', maybe 'g' or 'A' stripped later)
            if ch not in "0123456789.-gG aA":
                continue
            # xywh center x
            try:
                xc = float(b.xywh[0][0].item())
            except Exception:
                xc = 0.0
            conf_s = float(b.conf[0].item()) if hasattr(b, "conf") else 0.0
            boxes.append((xc, ch, conf_s))
    except Exception as e:
        _dprint("Predict reading parse error:", e)
        return "", 0.0

    if not boxes:
        return "", 0.0

    # Sort by x-center to assemble reading left-to-right
    boxes.sort(key=lambda x: x[0])
    reading = "".join(ch for _, ch, _ in boxes)
    # Cleanup repeating decimal points
    if reading.count(".") > 1:
        first = reading.find(".")
        reading = reading[:first + 1] + reading[first + 1:].replace(".", "")
    avg_conf = float(np.mean([s for _, _, s in boxes])) if boxes else 0.0
    return reading.strip(), avg_conf

# -------------------- MINI OCR PIPELINES -------------------- #
def read_lcd_current(frame: np.ndarray) -> Tuple[Optional[float], str, float]:
    """
    OCR reader for current meter display.
      - tries to ensure minus sign presence (via OCR + contour heuristics)
      - returns value as float, and display string formatted as '-xx.xx' or 'xx.xx'
    """
    h, w = frame.shape[:2]
    # Preprocess targeted region heuristics: current usually in a specific area; but fall back to full frame
    img = preprocess_frame(frame, invert=False)
    # Also build an inverted / thresholded variant for minus-line detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    inv_bin = 255 - binary

    reading, conf = _predict_reading(img, 0.25)
    reading = (reading or "").strip()

    # Normalize common artifacts
    reading = reading.replace("..", ".").replace("—", "-").replace("_", "-").replace(" ", "")
    if reading.endswith("."):
        reading = reading[:-1]
    if reading.lower().endswith("a"):
        reading = reading[:-1]

    # If OCR missed minus sign, try to detect a minus in left area using contours
    sign_found = "-" in reading
    if not sign_found:
        # examine left band where sign often appears
        left_w = max(10, int(w * 0.18))
        band = inv_bin[:, :left_w]
        try:
            contours, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except Exception:
            contours = []
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / (ch + 1e-6)
            # minus sign tends to be a thin wide rect - heuristics
            if aspect > 3.0 and ch < h * 0.2 and cw > 4:
                reading = "-" + reading
                sign_found = True
                break

    # Clean double signs
    reading = reading.replace("--", "-")

    # Convert reading to float carefully: handle leading zeros like 00.70 and strings without decimal
    numeric = None
    try:
        # Replace stray characters
        cleaned = "".join(ch for ch in reading if ch in "0123456789.-")
        if cleaned in ["", "-", ".", "-."]:
            numeric = None
        else:
            numeric = float(cleaned)
    except Exception:
        numeric = None

    # Format display string: ensure two decimals and sign
    display = ""
    if numeric is not None:
        # Round to 2 decimals for display, but keep numeric float as-is
        # Always show sign if negative
        if numeric < 0:
            display = f"-{abs(numeric):05.2f}"  # ensures -xx.xx or -x.xx becomes -xx.xx with leading zero
        else:
            # positive: ensure at least two integer places (leading zeros if <10)
            display = f"{numeric:06.2f}" if numeric >= 0 else f"{abs(numeric):06.2f}"
            # Strip leading extra zero to get xx.xx if it creates leading zero? user insisted -xx.xx format; positive they want xx.xx
            # For positive, make it xx.xx (no leading plus or extra leading zero)
            # e.g., 0.7 -> 00.70 -> drop leading zero to "00.70"? They asked xx.xx format — preserve 2 integer positions if needed
            # We'll keep consistent two decimal places and show leading zeros as "00.70" to match device reading style
            # But user earlier asked for "-xx.xx", so negative will have sign and padded digits.
            display = display  # keep as is like "00.70" or "12.34"
    else:
        display = reading or ""

    return numeric, display, float(conf or 0.0)


def read_lcd_thrust(frame: np.ndarray) -> Tuple[Optional[float], str, float]:
    """
    OCR reader for thrust display.
      - remove decimal artifacts by mapping "13.6" -> 136, "15.68" -> 1568 etc when appropriate
      - reject enormous impossible readings and filter flicker noise
      - returns integer grams (float numeric, and string display)
    """
    inverted = preprocess_frame(frame, invert=True)
    reading, conf = _predict_reading(inverted, 0.25)
    reading = (reading or "").strip()

    # Common cleanup
    reading = reading.replace("..", ".").replace("—", "-").replace("_", "-").replace(",", ".").replace(" ", "")
    if reading.lower().endswith("g"):
        reading = reading[:-1]
    if reading.endswith("."):
        reading = reading[:-1]
    if reading.startswith("--"):
        reading = "-" + reading.lstrip("-")

    if reading == "":
        return None, "", float(conf or 0.0)

    cleaned = reading
    # If contains decimals, try to decide intended meaning
    if "." in cleaned:
        parts = cleaned.split(".")
        left = parts[0] if parts else ""
        right = "".join(parts[1:]) if len(parts) > 1 else ""
        # heuristics:
        # if right has len 1 -> maybe decimal like 13.6 meaning 136 -> join
        if len(right) == 1 and len(left) >= 1:
            cleaned = left + right
        # if right length == 2 -> likely 2 decimal digits to be removed (15.68 -> 1568)
        elif len(right) == 2:
            cleaned = left + right
        # if more decimals - join first two segments
        elif len(right) > 2:
            cleaned = left + right
        else:
            cleaned = left + right

    # Keep digits and optional leading '-'
    cleaned = "".join(ch for ch in cleaned if ch.isdigit() or ch == "-")
    if cleaned in ["", "-", "-0"]:
        cleaned = "0"

    numeric = None
    try:
        numeric = float(cleaned)
    except Exception:
        numeric = None

    # Filter impossible values and smooth spikes with function attribute
    MAX_ALLOWED = 10000  # grams; tune if necessary
    MIN_NOISE = 2

    if numeric is None:
        return None, "", float(conf or 0.0)

    if abs(numeric) > MAX_ALLOWED:
        # impossible - treat as 0 or ignore
        numeric = 0.0
    if abs(numeric) < MIN_NOISE:
        numeric = 0.0
    else:
        numeric = round(float(numeric))

    # smoothing memory
    if not hasattr(read_lcd_thrust, "last_val"):
        read_lcd_thrust.last_val = 0.0

    last = getattr(read_lcd_thrust, "last_val", 0.0)
    if last and numeric and abs(numeric - last) > max(1500, last * 2):
        # huge jump (e.g., spike) -> likely OCR glitch, keep last
        numeric = last
    else:
        read_lcd_thrust.last_val = numeric

    display = str(int(numeric)) if numeric is not None else ""
    return numeric, display, float(conf or 0.0)


def read_lcd_rpm(frame: np.ndarray) -> Tuple[Optional[float], str, float]:
    """
    RPM reader: integer RPMs no decimals.
    """
    processed = preprocess_frame(frame, invert=False)
    reading, conf = _predict_reading(processed, 0.30)
    reading = (reading or "").strip()
    reading = reading.replace(" ", "").replace(",", "")
    # Keep only digits and optional '-'
    cleaned = "".join(ch for ch in reading if ch.isdigit() or ch == "-")
    try:
        val = float(cleaned) if cleaned not in ["", "-", "."] else None
    except Exception:
        val = None
    display = str(int(val)) if val is not None else ""
    return val, display, float(conf or 0.0)

# -------------------- MERGE / REPORT -------------------- #
def _safe_merge_series(cur: Optional[pd.DataFrame], thr: Optional[pd.DataFrame], rpm: Optional[pd.DataFrame]) -> pd.DataFrame:
    times = set()
    for df in (cur, thr, rpm):
        if df is not None and not df.empty and "time_s" in df.columns:
            times.update(df["time_s"].tolist())
    if not times:
        return pd.DataFrame(columns=["time_s"])
    merged = pd.DataFrame({"time_s": sorted(times)})
    if cur is not None and not cur.empty:
        merged = merged.merge(cur[["time_s", "current_a", "current_display"]], on="time_s", how="left")
    else:
        merged["current_a"] = np.nan
        merged["current_display"] = np.nan
    if thr is not None and not thr.empty:
        merged = merged.merge(thr[["time_s", "thrust_g", "thrust_display"]], on="time_s", how="left")
    else:
        merged["thrust_g"] = np.nan
        merged["thrust_display"] = np.nan
    if rpm is not None and not rpm.empty:
        merged = merged.merge(rpm[["time_s", "rpm", "rpm_display"]], on="time_s", how="left")
    else:
        merged["rpm"] = np.nan
        merged["rpm_display"] = np.nan
    # Fill forward small gaps to create contiguous table for plotting
    merged = merged.sort_values("time_s").reset_index(drop=True)
    return merged

def build_session_report(session_id: str) -> Dict:
    sess = SESSIONS.get(session_id)
    if not sess:
        return {}
    meta = sess["meta"]
    merged = _safe_merge_series(sess["series"].get("current"), sess["series"].get("thrust"), sess["series"].get("rpm"))

    if merged.empty:
        sess["report"] = {"table_csv": None, "graphs": [], "table_records": []}
        return sess["report"]

    # convert numeric columns safely
    for col in ["current_a", "thrust_g", "rpm"]:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # compute power & efficiency only if voltage is provided
    voltage = meta.get("voltage")
    if voltage is not None and "current_a" in merged.columns:
        merged["power_w"] = merged["current_a"] * float(voltage)
    else:
        merged["power_w"] = np.nan
    merged["efficiency_gw"] = merged["thrust_g"] / merged["power_w"].replace({0: np.nan}) if "thrust_g" in merged.columns else np.nan

    # Replace inf/nan with zeros for JSON safety, but keep copy for CSV
    csv_df = merged.copy()
    csv_path = os.path.join(RESULT_DIR, f"{session_id}_report.csv")
    csv_df.to_csv(csv_path, index=False)

    # For passing to frontend and plotting, replace inf/nan with 0
    merged = merged.replace([np.inf, -np.inf], np.nan).fillna(0)
    # convert any numpy types to python
    merged = merged.applymap(lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)

    # -------- Plot Graphs -------- #
    graphs = []
    def plot(col, label, name):
        if col in merged.columns:
            numeric = pd.to_numeric(merged[col], errors="coerce")
            if numeric.notna().any():
                plt.figure(figsize=(12, 4))
                plt.plot(merged["time_s"].astype(float), numeric.astype(float))
                plt.xlabel("Time (s)")
                plt.ylabel(label)
                plt.title(f"{label} vs Time")
                plt.grid(True)
                out = os.path.join(RESULT_DIR, f"{session_id}_{name}.png")
                plt.savefig(out, bbox_inches="tight")
                plt.close()
                graphs.append(out)

    plot("current_a", "Current (A)", "current")
    plot("thrust_g", "Thrust (G)", "thrust")
    plot("rpm", "RPM", "rpm")

    # -------- Build table records for frontend -------- #
    records = []
    for rec in merged.to_dict(orient="records"):
        safe = {}
        for k, v in rec.items():
            # sanitize NaN / inf
            try:
                if isinstance(v, (float, np.floating)) and (math.isnan(v) or math.isinf(v)):
                    v = 0.0
            except Exception:
                pass
            # ensure native python types
            if isinstance(v, np.generic):
                v = np.asscalar(v) if hasattr(np, "asscalar") else v.item()
            safe[k] = v

        frontend_row = {
            "Time (s)": round(float(safe.get("time_s", 0)), 3),
            "Voltage (V)": meta.get("voltage", 0),
            "Prop": meta.get("prop", ""),
            "Motor": meta.get("motor", ""),
            "ESC": meta.get("esc", ""),
            "Throttle": "",  # placeholder
            "Current (A)": round(float(safe.get("current_a", 0)), 2) if safe.get("current_a", 0) is not None else 0.0,
            "Power (W)": round(float(safe.get("power_w", 0)), 2) if safe.get("power_w", 0) is not None else 0.0,
            "Thrust (G)": int(round(float(safe.get("thrust_g", 0)))) if safe.get("thrust_g", 0) is not None else 0,
            "RPM": int(round(float(safe.get("rpm", 0)))) if safe.get("rpm", 0) is not None else 0,
            "Efficiency (G/W)": round(float(safe.get("efficiency_gw", 0)), 3) if safe.get("efficiency_gw", 0) is not None else 0.0,
            "Operating Temperature (°C)": "",  # placeholder
        }
        records.append(frontend_row)

    sess["report"] = {"table_csv": csv_path, "graphs": graphs, "table_records": records}
    return sess["report"]

# -------------------- MAIN PROCESS -------------------- #
def process_video_task(task_id: str, session_id: str, video_type: str, video_path: str, meta: Dict) -> None:
    """
    Processes a single uploaded video (one per call).
    video_type in {"current", "thrust", "rpm"}
    """
    try:
        reset_progress(task_id)
        TASK_PROGRESS[task_id].update({"status": "running", "phase": "init", "progress": 0})
        _init_session(session_id, meta)

        fps = int(meta.get("fps", 5))
        TASK_PROGRESS[task_id].update({"phase": "extracting", "message": "Extracting frames..."})
        frames = extract_frames_custom(task_id, video_path, fps)
        total = max(1, len(frames))
        TASK_PROGRESS[task_id].update({"message": f"Processing {total} frames...", "progress": 1})

        # rows will be time-series data for this video_type
        rows = []
        # recent buffer of raw display strings for consensus
        recent_raw = deque(maxlen=9)  # increased length for stability
        recent_conf = deque(maxlen=9)
        # track last valid numeric for smoothing (store per video_type)
        last_val_attr = f"last_val_{video_type}"
        if not hasattr(process_video_task, last_val_attr):
            setattr(process_video_task, last_val_attr, 0.0)

        # Warm up the model if needed
        if MODEL is not None:
            try:
                # small warmup with blank image to avoid first-inference spike
                blank = np.zeros((64, 64, 3), dtype=np.uint8)
                MODEL.predict(blank, verbose=False)
            except Exception:
                pass

        for i, fp in enumerate(frames):
            img = cv2.imread(fp)
            if img is None:
                # update progress and continue
                TASK_PROGRESS[task_id]["progress"] = int((i / total) * 85)
                continue

            # select reader
            if video_type == "current":
                val, raw, conf = read_lcd_current(img)
                display_key = "current_display"
                numeric_key = "current_a"
            elif video_type == "thrust":
                val, raw, conf = read_lcd_thrust(img)
                display_key = "thrust_display"
                numeric_key = "thrust_g"
            else:
                val, raw, conf = read_lcd_rpm(img)
                display_key = "rpm_display"
                numeric_key = "rpm"

            raw = raw or ""
            conf = float(conf or 0.0)
            recent_raw.append(raw)
            recent_conf.append(conf)

            # Consensus logic:
            # pick most common non-empty reading in recent buffer weighted by average confidence
            non_empty = [r for r in recent_raw if r not in ["", None]]
            chosen = raw
            if non_empty:
                counts = Counter(non_empty)
                most_common, cnt = counts.most_common(1)[0]
                # compute avg conf of frames that reported most_common
                confs_for_mc = [recent_conf[idx] for idx, r in enumerate(recent_raw) if r == most_common]
                avg_conf_mc = float(np.mean(confs_for_mc)) if confs_for_mc else 0.0
                # compute avg conf for current raw
                avg_conf_raw = conf
                # prefer most_common if it appears >= 2 times and average confidence comparable
                if cnt >= 2 and avg_conf_mc >= (avg_conf_raw - 0.15):
                    chosen = most_common
                else:
                    chosen = raw

            # If the previous saved raw had a negative sign, keep sign if digits match
            prev_row = rows[-1] if rows else None
            if prev_row is not None:
                prev_disp = prev_row.get(display_key, "")
                if prev_disp and prev_disp.startswith("-") and not chosen.startswith("-"):
                    # compare digits-only
                    prev_digits = "".join(ch for ch in prev_disp if ch.isdigit() or ch == ".")
                    now_digits = "".join(ch for ch in chosen if ch.isdigit() or ch == ".")
                    if prev_digits == now_digits and now_digits != "":
                        chosen = "-" + chosen

            # try to parse chosen into numeric if not already
            numeric_val = None
            try:
                # for current, chosen may be like "00.70" or "-00.70" -> parse
                cleaned = "".join(ch for ch in chosen if ch in "0123456789.-")
                if cleaned not in ["", "-", ".", "-."]:
                    numeric_val = float(cleaned)
                else:
                    numeric_val = None
            except Exception:
                numeric_val = None

            # smoothing and spike filter per type
            last_val = getattr(process_video_task, last_val_attr, 0.0)
            if numeric_val is not None:
                if video_type == "thrust":
                    # clamp unrealistic values and smooth spikes
                    if abs(numeric_val) > 20000:  # safety clamp
                        numeric_val = last_val
                    # remove small noise
                    if abs(numeric_val) < 2:
                        numeric_val = 0.0
                    else:
                        numeric_val = round(float(numeric_val))
                    # spike rejection: > 3x last and absolute diff big => keep last
                    if last_val and abs(numeric_val - last_val) > max(2000, last_val * 2):
                        numeric_val = last_val
                    else:
                        setattr(process_video_task, last_val_attr, numeric_val)
                elif video_type == "current":
                    # keep 2 decimals
                    # numeric_val remains float with decimals; format later for display
                    # minor smoothing
                    if last_val and abs(numeric_val - last_val) > 5:
                        # improbable change in current per frame: use last val if large jump
                        numeric_val = last_val
                    else:
                        setattr(process_video_task, last_val_attr, numeric_val)
                else:  # rpm
                    # integer rpm
                    if numeric_val < 0:
                        # RPM negative? unlikely: keep absolute
                        numeric_val = abs(numeric_val)
                    numeric_val = int(round(float(numeric_val)))
                    if last_val and abs(numeric_val - last_val) > max(500, last_val * 2):
                        numeric_val = last_val
                    else:
                        setattr(process_video_task, last_val_attr, numeric_val)

            # timestamp
            t = round(i * (1.0 / float(max(1, fps))), 4)
            row = {"time_s": t, display_key: chosen, numeric_key: numeric_val}
            rows.append(row)

            # Update progress (cap at 85 while processing)
            if total:
                TASK_PROGRESS[task_id]["progress"] = min(85, int(((i + 1) / total) * 85))
            _dprint(f"[DEBUG] Frame {i+1}/{total} | Type:{video_type} | Raw:'{raw}' | Chosen:'{chosen}' | Conf:{conf:.2f} | Val:{numeric_val}")

        # Create DataFrame and store to session
        df = pd.DataFrame(rows)
        # Ensure columns exist consistently
        if video_type == "current":
            if "current_a" not in df.columns:
                df["current_a"] = np.nan
            if "current_display" not in df.columns:
                df["current_display"] = ""
        elif video_type == "thrust":
            if "thrust_g" not in df.columns:
                df["thrust_g"] = np.nan
            if "thrust_display" not in df.columns:
                df["thrust_display"] = ""
        else:
            if "rpm" not in df.columns:
                df["rpm"] = np.nan
            if "rpm_display" not in df.columns:
                df["rpm_display"] = ""

        # coerce types: numeric columns to floats
        if video_type == "current":
            df["current_a"] = pd.to_numeric(df["current_a"], errors="coerce").fillna(0.0)
        elif video_type == "thrust":
            df["thrust_g"] = pd.to_numeric(df["thrust_g"], errors="coerce").fillna(0.0)
        else:
            df["rpm"] = pd.to_numeric(df["rpm"], errors="coerce").fillna(0.0)

        # store updated series into session (overwrites previous for this type)
        if session_id not in SESSIONS:
            _init_session(session_id, meta)
        SESSIONS[session_id]["series"][video_type] = df

        TASK_PROGRESS[task_id].update({"phase": "report", "progress": 90, "message": "Building report..."})
        report = build_session_report(session_id)

        # Persist results
        save_test_result(
            session_id=session_id,
            prop_name=meta.get("prop", ""),
            motor_name=meta.get("motor", ""),
            esc_name=meta.get("esc", ""),
            voltage=meta.get("voltage"),
            csv_path=report.get("table_csv"),
            graph_paths=report.get("graphs", []),
            table_data=report.get("table_records", [])
        )

        TASK_PROGRESS[task_id].update({"status": "done", "progress": 100, "message": "Completed"})
        _dprint(f"[INFO] Completed {video_type} ({len(frames)} frames) in session {session_id}")

    except Exception as e:
        logging.exception("Error processing video task")
        TASK_PROGRESS[task_id].update({"status": "error", "message": str(e)})
    finally:
        # attempt to remove uploaded file to keep disk tidy
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception:
            pass

# -------------------- BACKGROUND STARTER / SESSION API HELPERS -------------------- #
def start_background_task(session_id: str, video_type: str, video_path: str, meta: Dict) -> str:
    task_id = str(uuid.uuid4())
    reset_progress(task_id)
    threading.Thread(target=process_video_task, args=(task_id, session_id, video_type, video_path, meta), daemon=True).start()
    return task_id

def get_session_report(session_id: str) -> Optional[Dict]:
    sess = SESSIONS.get(session_id)
    if not sess or not sess.get("report"):
        return None
    rep = sess["report"]
    meta = sess["meta"]
    return {
        "meta": meta,
        "table": rep.get("table_records", []),
        "csv_url": f"/session/{session_id}/csv" if rep.get("table_csv") else None,
        "graphs": [f"/session/{session_id}/graph/{i}" for i in range(len(rep.get("graphs", [])))]
    }

def get_session_graph_path(session_id: str, idx: int) -> Optional[str]:
    sess = SESSIONS.get(session_id)
    if not sess or not sess.get("report"):
        return None
    graphs = sess["report"].get("graphs", [])
    if 0 <= idx < len(graphs):
        return graphs[idx]
    return None

def get_session_csv_path(session_id: str) -> Optional[str]:
    sess = SESSIONS.get(session_id)
    if not sess or not sess.get("report"):
        return None
    return sess["report"].get("table_csv")

# -------------------- END -------------------- #
