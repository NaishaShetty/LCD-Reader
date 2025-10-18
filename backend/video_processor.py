#!/usr/bin/env python3
# video_processor.py — Full untrimmed, fixed version
# - preserves existing pipelines, adds robustness and fixes column/merge bugs
# - copy-paste to replace your backend file

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
from collections import deque, Counter
from database import save_test_result
import time
import math

# -------------------- CONFIG -------------------- #
DEBUG = True
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(BASE_DIR, "lcd_ocr_model.pt")
YAML_PATH = os.path.join(BASE_DIR, "data.yaml")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Load model and class names
MODEL = YOLO(MODEL_PATH)
with open(YAML_PATH, "r") as f:
    DATA_CFG = yaml.safe_load(f)
CLASS_NAMES = DATA_CFG.get("names", [])

TASK_PROGRESS: Dict[str, Dict] = {}
SESSIONS: Dict[str, Dict] = {}

def _dprint(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

# -------------------- SESSION -------------------- #
def _init_session(session_id: str, meta: Dict) -> None:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "meta": {
                "prop": meta.get("prop", ""),
                "motor": meta.get("motor", ""),
                "esc": meta.get("esc", ""),
                "voltage": float(meta.get("voltage")) if meta.get("voltage") not in [None, ""] else None,
                "fps": int(meta.get("fps", 5)) if meta.get("fps") not in [None, ""] else 5,
            },
            "series": {"current": None, "thrust": None, "rpm": None},
            "report": None,
        }
    else:
        for k in ["prop", "motor", "esc", "voltage", "fps"]:
            v = meta.get(k)
            if v not in [None, ""]:
                if k == "voltage":
                    try:
                        v = float(v)
                    except (ValueError, TypeError):
                        v = None
                if k == "fps":
                    try:
                        v = int(v)
                    except (ValueError, TypeError):
                        v = 5
                SESSIONS[session_id]["meta"][k] = v

def reset_progress(task_id: str) -> None:
    TASK_PROGRESS[task_id] = {"status": "pending", "progress": 0, "phase": "queued", "message": ""}

def get_progress(task_id: str) -> Dict:
    return TASK_PROGRESS.get(task_id, {"status": "unknown", "progress": 0, "phase": "unknown"})

# -------------------- FRAME EXTRACTION -------------------- #
def _ffmpeg_extract(video_path: str, out_dir: str, fps: int) -> None:
    """
    Use ffmpeg to extract frames at requested fps. Use fps_mode='vfr' equivalent.
    """
    try:
        (
            ffmpeg
            .input(video_path)
            .filter("fps", fps=fps)
            .output(os.path.join(out_dir, "frame_%06d.jpg"), start_number=0, qscale=2, fps_mode="vfr")
            .overwrite_output()
            .run(quiet=not DEBUG)
        )
    except TypeError:
        # older ffmpeg-python versions may not support fps_mode argument
        (
            ffmpeg
            .input(video_path)
            .filter("fps", fps=fps)
            .output(os.path.join(out_dir, "frame_%06d.jpg"), start_number=0, qscale=2)
            .overwrite_output()
            .run(quiet=not DEBUG)
        )

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
    TASK_PROGRESS[task_id].update({"phase": "extracting", "message": f"Extracting frames ({fps} FPS)...", "progress": 1})
    try:
        _ffmpeg_extract(video_path, frames_dir, fps=fps)
    except Exception as e:
        TASK_PROGRESS[task_id].update({"message": f"FFmpeg failed, fallback OpenCV. ({e})"})
        _opencv_fallback_extract(video_path, frames_dir, fps=fps)
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")])
    if len(frame_files) < 5:
        _opencv_fallback_extract(video_path, frames_dir, fps=fps)
        frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")])
    _dprint(f"[DEBUG] Extracted {len(frame_files)} frames from {os.path.basename(video_path)} at {fps} FPS")
    return frame_files

# -------------------- PREPROCESSING -------------------- #
def preprocess_frame(img: Optional[np.ndarray], invert: bool = False) -> Optional[np.ndarray]:
    if img is None:
        return None
    h, w = img.shape[:2]
    # rotate if portrait-ish
    if h > w * 1.15:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if invert:
        gray = cv2.bitwise_not(gray)
    den = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enh = clahe.apply(den)
    thr = cv2.adaptiveThreshold(enh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    k = np.ones((2, 2), np.uint8)
    morph = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, k)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, k)
    blur = cv2.GaussianBlur(morph, (0, 0), 1.5)
    sharp = cv2.addWeighted(morph, 1.4, blur, -0.4, 0)
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

# -------------------- OCR CORE -------------------- #
def _predict_reading(img: np.ndarray, conf: float) -> Tuple[str, float]:
    """
    Run YOLO and build a reading string from predicted characters.
    Skip classes that are units or letters likely to be 'g' etc.
    Only keep characters in digits, '.', '-' and treat 'g','G' as units to strip.
    """
    results = MODEL.predict(img, conf=conf, verbose=False)[0]
    boxes: List[Tuple[float, str, float]] = []
    for b in results.boxes:
        try:
            cls = int(b.cls[0].item())
            xc = float(b.xywh[0][0].item())
        except Exception:
            continue
        ch = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else ""
        # Normalize char
        ch = str(ch).strip()
        # Skip common letters that are units (g, G)
        if ch.lower() in ["g", "v", "a", "r", "p", "t", "c", "o", "s"]:
            continue
        # Accept only digit, dot, minus
        if ch not in "0123456789.-":
            continue
        score = float(b.conf[0].item()) if hasattr(b, "conf") else 0.0
        boxes.append((xc, ch, score))
    if not boxes:
        return "", 0.0
    boxes.sort(key=lambda x: x[0])
    reading = "".join([ch for _, ch, _ in boxes])
    # Fix multiple decimals by keeping first decimal point
    if reading.count(".") > 1:
        first = reading.find(".")
        reading = reading[:first + 1] + reading[first + 1:].replace(".", "")
    avg_conf = float(np.mean([s for _, _, s in boxes])) if boxes else 0.0
    return reading.strip(), avg_conf

# -------------------- MINI OCR PIPELINES -------------------- #
def read_lcd_current(frame: np.ndarray) -> Tuple[Optional[float], str, float]:
    """
    Read current meter display. Aim: always capture leading '-' if present.
    Output display string formatted as xx.xx (two decimals).
    """
    # Preprocess both polarities; minus sign often appears as a thin horizontal glyph
    processed = preprocess_frame(frame, invert=False)
    inv = preprocess_frame(frame, invert=True)

    # Heuristics: run predict on both images & choose highest confidence
    r1, c1 = _predict_reading(processed, 0.22) if processed is not None else ("", 0.0)
    r2, c2 = _predict_reading(inv, 0.22) if inv is not None else ("", 0.0)

    # Prefer the reading with higher avg confidence
    reading, conf = (r2, c2) if c2 > c1 else (r1, c1)
    reading = reading.replace("..", ".").replace("—", "-").replace("_", "-").strip()

    # Remove trailing units/letters
    if reading.lower().endswith("a"):
        reading = reading[:-1]

    # If reading contains letters/garbage, keep only digits, '.', '-'
    filtered = ''.join(ch for ch in reading if ch in "0123456789.-")
    reading = filtered

    # Try to detect missing minus sign by analyzing left-side thin contours (common minus)
    if "-" not in reading:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            # Left area (minus often sits to the left of digits)
            roi = gray[int(h * 0.35):int(h * 0.65), 0:int(w * 0.22)]
            _, bin_roi = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(bin_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, cw, ch = cv2.boundingRect(cnt)
                aspect = cw / float(max(ch, 1))
                # a long thin contour is a good minus sign candidate
                if aspect > 3.5 and ch < h * 0.25:
                    reading = "-" + reading
                    break
        except Exception:
            pass

    # Clean multiple '-' and leading formatting
    reading = reading.replace("--", "-")
    reading = reading.strip()

    # Convert to float and format as xx.xx (two decimals). Keep sign.
    val = None
    try:
        if reading not in ["", "-", "."]:
            val = float(reading)
        else:
            val = None
    except Exception:
        val = None

    # Format readout for display: ensure two decimal places with consistent width
    display = "-"
    if val is None:
        display = "-"
    else:
        # Use formatting preserving sign and two decimals, but do not add extra padding beyond sign
        # Example: -0.71 -> "-0.71" ; 0.71 -> "0.71"
        display = f"{val:.2f}"

    return val, display, float(conf)

def read_lcd_thrust(frame: np.ndarray) -> Tuple[Optional[float], str, float]:
    """
    Thrust OCR: aim to produce integer grams. Remove stray decimal artifacts (like '13.6' being read vs '136').
    Reject physically impossible spikes. Keep display as integer string.
    """
    proc = preprocess_frame(frame, invert=True)
    reading, conf = _predict_reading(proc, 0.25) if proc is not None else ("", 0.0)
    reading = reading.replace("..", ".").replace("—", "-").replace("_", "-").replace(",", ".").strip()

    # Remove trailing unit 'g' if OCR left it
    if reading.lower().endswith("g"):
        reading = reading[:-1]

    # Basic normalization: keep only digits, '.', '-'
    cleaned = ''.join(ch for ch in reading if ch in "0123456789.-")

    # If multiple decimals, keep the first decimal and remove others
    if cleaned.count(".") > 1:
        parts = cleaned.split(".")
        cleaned = parts[0] + "." + "".join(parts[1:])

    # Heuristic: if there is a single decimal digit (like 13.6), map to integer by left+right -> '136'
    numeric_for_int = cleaned
    if "." in numeric_for_int:
        left, right = numeric_for_int.split(".", 1)
        # If right is length 1, interpret as tenths displayed and convert: "13.6" -> 136
        if len(right) == 1:
            numeric_for_int = left + right
        else:
            # if right length is more (e.g., "15.68"), join left+right to remove decimal -> "1568"
            numeric_for_int = left + right

    # Remove any remaining non-digit/minus
    numeric_for_int = ''.join(ch for ch in numeric_for_int if ch.isdigit() or ch == '-')

    # Normalize empty or lone '-'
    if numeric_for_int in ["", "-", "."]:
        numeric_for_int = "0"

    val = None
    try:
        val = float(numeric_for_int)
    except Exception:
        val = None

    # Sanity checks & rounding for thrust:
    if val is not None:
        # Reject impossible huge readings
        if abs(val) > 10000:
            val = None
        elif abs(val) < 2:
            # treat as zero/noise
            val = 0.0
        else:
            # round to nearest integer (thrust reported as grams)
            val = float(round(val))

    display = "-" if val is None else str(int(val))

    # Keep a small smoothing memory to avoid single-frame spikes
    if not hasattr(read_lcd_thrust, "last"):
        read_lcd_thrust.last = None
    if val is not None and read_lcd_thrust.last is not None:
        if abs(val - read_lcd_thrust.last) > max(200, read_lcd_thrust.last * 2):
            # spike — ignore and retain last
            val = read_lcd_thrust.last
    if val is not None:
        read_lcd_thrust.last = val

    return val, display, float(conf)

def read_lcd_rpm(frame: np.ndarray) -> Tuple[Optional[float], str, float]:
    """
    RPM OCR: expect integer-like numbers (no decimals). Remove stray '.' and non-digit chars.
    """
    proc = preprocess_frame(frame, invert=False)
    reading, conf = _predict_reading(proc, 0.30) if proc is not None else ("", 0.0)
    reading = reading.replace(".", "").strip()  # drop decimal artifacts for rpm (rare)
    cleaned = ''.join(ch for ch in reading if ch.isdigit() or ch == '-')
    if cleaned in ["", "-", "."]:
        val = None
        display = "-"
    else:
        try:
            val = float(cleaned)
            display = str(int(round(val)))
        except Exception:
            val = None
            display = "-"
    return val, display, float(conf)

# -------------------- MERGING & REPORTS -------------------- #
def _safe_merge_series(cur: Optional[pd.DataFrame], thr: Optional[pd.DataFrame], rpm: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge three series into a single DataFrame with aligned time_s.
    Ensure column names: current_a, current_display, thrust_g, thrust_display, rpm, rpm_display
    """
    times = set()
    for df in (cur, thr, rpm):
        if df is not None and not df.empty and "time_s" in df.columns:
            times.update(df["time_s"].tolist())
    if not times:
        return pd.DataFrame(columns=["time_s"])
    merged = pd.DataFrame({"time_s": sorted(times)})

    if cur is not None and not cur.empty:
        # Accept columns current_a and/or current_display
        cols = [c for c in ["time_s", "current_a", "current_display"] if c in cur.columns]
        merged = merged.merge(cur[cols], on="time_s", how="left")
    else:
        merged["current_a"] = np.nan
        merged["current_display"] = np.nan

    if thr is not None and not thr.empty:
        cols = [c for c in ["time_s", "thrust_g", "thrust_display"] if c in thr.columns]
        merged = merged.merge(thr[cols], on="time_s", how="left")
    else:
        merged["thrust_g"] = np.nan
        merged["thrust_display"] = np.nan

    if rpm is not None and not rpm.empty:
        cols = [c for c in ["time_s", "rpm", "rpm_display"] if c in rpm.columns]
        merged = merged.merge(rpm[cols], on="time_s", how="left")
    else:
        merged["rpm"] = np.nan
        merged["rpm_display"] = np.nan

    # Ensure display columns exist
    for col in ["current_display", "thrust_display", "rpm_display"]:
        if col not in merged.columns:
            merged[col] = np.nan

    # Ensure numeric columns exist
    for col in ["current_a", "thrust_g", "rpm"]:
        if col not in merged.columns:
            merged[col] = np.nan

    return merged

def build_session_report(session_id: str) -> Dict:
    """
    Build report, CSV and graphs. Return dict with table_records, graphs list and csv path.
    """
    sess = SESSIONS[session_id]
    meta = sess["meta"]

    cur_df = sess["series"].get("current")
    thr_df = sess["series"].get("thrust")
    rpm_df = sess["series"].get("rpm")

    merged = _safe_merge_series(cur_df, thr_df, rpm_df)

    if merged.empty:
        sess["report"] = {"table_csv": None, "graphs": [], "table_records": []}
        return sess["report"]

    # convert numeric columns
    for c in ["current_a", "thrust_g", "rpm"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    # compute power if voltage present
    voltage = meta.get("voltage")
    if voltage is not None:
        merged["power_w"] = merged["current_a"] * float(voltage)
    else:
        merged["power_w"] = np.nan

    # compute efficiency (protect against div-by-zero)
    merged["efficiency_gw"] = merged["thrust_g"] / merged["power_w"]
    # sanitize infinite/nan
    merged = merged.replace([np.inf, -np.inf], np.nan)

    # Save raw merged CSV (machine format)
    csv_path = os.path.join(RESULT_DIR, f"{session_id}_merged.csv")
    merged.to_csv(csv_path, index=False)

    # Plot graphs and save
    graphs: List[str] = []
    def _plot(col: str, ylabel: str, fname: str):
        if col not in merged.columns:
            return
        ser = pd.to_numeric(merged[col], errors="coerce")
        if ser.dropna().empty:
            return
        plt.figure(figsize=(12, 4))
        plt.plot(merged["time_s"], ser, marker="o", markersize=2)
        plt.xlabel("Time (s)")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs Time")
        plt.grid(True, alpha=0.3)
        out = os.path.join(RESULT_DIR, f"{session_id}_{fname}.png")
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        graphs.append(out)

    _plot("current_a", "Current (A)", "current")
    _plot("thrust_g", "Thrust (G)", "thrust")
    _plot("rpm", "RPM", "rpm")
    _plot("power_w", "Power (W)", "power")
    _plot("efficiency_gw", "Efficiency (G/W)", "efficiency")

    # Build frontend-friendly pretty table rows preserving display strings when available
    records: List[Dict] = []
    # prefer display columns if available, fallback to numeric formatting
    for _, row in merged.iterrows():
        # get display columns or format numeric
        def safe_display(numcol: str, dispcol: str, int_mode=False, decimals=2, default="-"):
            if dispcol in merged.columns and pd.notna(row.get(dispcol)):
                return str(row.get(dispcol))
            if numcol in merged.columns and not pd.isna(row.get(numcol)):
                v = row.get(numcol)
                if int_mode:
                    return str(int(round(float(v))))
                else:
                    return f"{float(v):.{decimals}f}"
            return default

        rec = {
            "Time (s)": round(float(row.get("time_s", 0.0)), 2),
            "Voltage (V)": meta.get("voltage") if meta.get("voltage") is not None else "-",
            "Prop": meta.get("prop") or "-",
            "Motor": meta.get("motor") or "-",
            "ESC": meta.get("esc") or "-",
            "Throttle": "-",
            "Current (A)": safe_display("current_a", "current_display", int_mode=False, decimals=2, default="-"),
            "Power (W)": safe_display("power_w", "power_w", int_mode=False, decimals=2, default="-"),
            "Thrust (G)": safe_display("thrust_g", "thrust_display", int_mode=True, decimals=0, default="-"),
            "RPM": safe_display("rpm", "rpm_display", int_mode=True, decimals=0, default="-"),
            "Efficiency (G/W)": safe_display("efficiency_gw", "efficiency_gw", int_mode=False, decimals=3, default="-"),
            "Operating Temperature (°C)": "-",
        }
        records.append(rec)

    # Save pretty CSV for user download
    pretty_df = pd.DataFrame(records)
    pretty_csv = os.path.join(RESULT_DIR, f"{session_id}_report.csv")
    pretty_df.to_csv(pretty_csv, index=False)

    sess["report"] = {"table_csv": pretty_csv, "graphs": graphs, "table_records": records}
    return sess["report"]

# -------------------- MAIN PROCESS -------------------- #
def process_video_task(task_id: str, session_id: str, video_type: str, video_path: str, meta: Dict) -> None:
    """
    Process a single video file in background.
    video_type must be one of: 'current', 'thrust', 'rpm'
    """
    try:
        reset_progress(task_id)
        TASK_PROGRESS[task_id].update({"status": "running", "phase": "init", "progress": 0})
        _init_session(session_id, meta)
        fps = int(meta.get("fps", 5)) if meta.get("fps") else 5
        if fps <= 0:
            fps = 5

        # Extract frames
        frames = extract_frames_custom(task_id, video_path, fps)
        if not frames:
            TASK_PROGRESS[task_id].update({"status": "error", "message": "No frames extracted", "progress": 0})
            return

        total = len(frames)
        TASK_PROGRESS[task_id].update({"phase": "processing", "progress": 1})
        _dprint(f"[DEBUG] Starting processing {total} frames for {video_type}")

        # Warm-up model once (YOLO can be slow first call)
        try:
            if not hasattr(MODEL, "_warmed_up"):
                _dprint("[INFO] Warming up YOLO model...")
                MODEL.predict(np.zeros((64, 64, 3), np.uint8))
                MODEL._warmed_up = True
        except Exception as e:
            _dprint(f"[WARN] YOLO warm-up failed: {e}")

        rows = []
        recent = deque(maxlen=7)
        prev_raw_global = None

        for i, fp in enumerate(frames):
            img = cv2.imread(fp)
            if img is None:
                # continue but still update progress
                TASK_PROGRESS[task_id]["progress"] = int(((i+1)/total) * 90)
                continue

            # choose pipeline
            if video_type == "current":
                val, raw, conf = read_lcd_current(img)
                col_val_name = "current_a"
                col_disp_name = "current_display"
            elif video_type == "thrust":
                val, raw, conf = read_lcd_thrust(img)
                col_val_name = "thrust_g"
                col_disp_name = "thrust_display"
            else:
                val, raw, conf = read_lcd_rpm(img)
                col_val_name = "rpm"
                col_disp_name = "rpm_display"

            # Stabilize by recent mode
            recent.append(str(raw))
            non_empty = [r for r in recent if r not in ["", None, "-"]]
            chosen_raw = None
            if non_empty:
                counter = Counter(non_empty)
                most_common, count = counter.most_common(1)[0]
                if count >= 2:
                    chosen_raw = most_common
            if chosen_raw is None:
                chosen_raw = str(raw)

            # Preserve minus sign heuristics
            if non_empty:
                neg_count = sum(1 for r in non_empty if isinstance(r, str) and r.startswith("-"))
                if neg_count >= max(1, (len(non_empty) // 2)) and not (isinstance(chosen_raw, str) and chosen_raw.startswith("-")):
                    s = "" if chosen_raw in [None] else str(chosen_raw)
                    chosen_raw = "-" + s.lstrip("-")

            if prev_raw_global and isinstance(prev_raw_global, str) and prev_raw_global.startswith("-") and isinstance(chosen_raw, str) and not chosen_raw.startswith("-"):
                digits_prev = "".join([c for c in prev_raw_global if c.isdigit() or c == "."])
                digits_now = "".join([c for c in chosen_raw if c.isdigit() or c == "."])
                if digits_prev == digits_now and digits_now != "":
                    chosen_raw = "-" + chosen_raw.lstrip("-")

            prev_raw_global = chosen_raw if isinstance(chosen_raw, str) and chosen_raw != "" else prev_raw_global

            # Try parse numeric value from chosen_raw (be lenient)
            chosen_val = None
            try:
                tmp = ''.join(ch for ch in chosen_raw if ch in "0123456789.-")
                if tmp not in ["", "-", "."]:
                    chosen_val = float(tmp)
            except Exception:
                chosen_val = None

            # normalize storage: use canonical column names expected later
            entry = {"time_s": round(i * (1.0 / max(1, fps)), 4)}
            entry[col_disp_name] = chosen_raw
            entry[col_val_name] = chosen_val

            rows.append(entry)

            # update progress intermittently
            if i % max(1, total // 100) == 0 or i == total - 1:
                TASK_PROGRESS[task_id]["progress"] = int(((i+1)/total) * 90)

            _dprint(f"[DEBUG] Frame {i+1}/{total} | Type:{video_type} | Raw:'{raw}' | Chosen:'{chosen_raw}' | Conf:{conf:.2f} | Val:{chosen_val}")

        # Build dataframe and attach to session
        if rows:
            df = pd.DataFrame(rows)
        else:
            df = pd.DataFrame(columns=["time_s", col_val_name, col_disp_name])

        # Ensure columns exist with proper names
        # For current: current_a, current_display
        # For thrust: thrust_g, thrust_display
        # For rpm: rpm, rpm_display
        expected_val_col = col_val_name
        expected_disp_col = col_disp_name
        if expected_val_col not in df.columns:
            df[expected_val_col] = np.nan
        if expected_disp_col not in df.columns:
            df[expected_disp_col] = np.nan
        if "time_s" not in df.columns:
            df["time_s"] = [round(i * (1.0 / max(1, fps)), 4) for i in range(len(df))]

        # coerce numeric column to numeric type
        try:
            df[expected_val_col] = pd.to_numeric(df[expected_val_col], errors="coerce")
        except Exception:
            pass

        # store the series in the session in canonical name
        SESSIONS[session_id]["series"][video_type] = df

        TASK_PROGRESS[task_id].update({"phase": "report", "progress": 95})
        report = build_session_report(session_id)

        # persist result in DB as before
        save_test_result(
            session_id=session_id,
            prop_name=SESSIONS[session_id]["meta"].get("prop", ""),
            motor_name=SESSIONS[session_id]["meta"].get("motor", ""),
            esc_name=SESSIONS[session_id]["meta"].get("esc", ""),
            voltage=SESSIONS[session_id]["meta"].get("voltage"),
            csv_path=report.get("table_csv"),
            graph_paths=report.get("graphs", []),
            table_data=report.get("table_records", []),
        )

        TASK_PROGRESS[task_id].update({"status": "done", "progress": 100, "message": "Completed"})
        _dprint(f"[INFO] Completed {video_type} ({len(frames)} frames) in session {session_id}")

    except Exception as e:
        _dprint(f"[ERROR] process_video_task exception: {type(e).__name__}: {e}")
        TASK_PROGRESS[task_id].update({"status": "error", "message": f"{type(e).__name__}: {e}"})
    finally:
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception:
            pass

# -------------------- SESSION HELPERS -------------------- #
def start_background_task(session_id: str, video_type: str, video_path: str, meta: Dict) -> str:
    task_id = str(uuid.uuid4())
    threading.Thread(
        target=process_video_task,
        args=(task_id, session_id, video_type, video_path, meta),
        daemon=True
    ).start()
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
        "graphs": [f"/session/{session_id}/graph/{i}" for i in range(len(rep.get("graphs", [])))],
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
