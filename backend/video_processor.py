#!/usr/bin/env python3
# ============================================================
# video_processor.py — Full untrimmed version (with JSON-safe output)
# - Keeps current display format as -xx.xx (two digits before decimal)
# - Keeps Plotly interactive graph (hover shows values)
# - Adds JSON-safe sanitization of table_records to avoid
#   "Out of range float values are not JSON compliant" errors.
# ============================================================

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
import plotly.graph_objects as go
from typing import Dict, Optional, List, Tuple
from ultralytics import YOLO
import yaml
from collections import deque, Counter
from database import save_test_result
import time
import math
import logging

# -------------------- LOGGING & DEBUG --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
DEBUG = True

def _dprint(*args, **kwargs):
    if DEBUG:
        logging.debug(" ".join(str(a) for a in args))

# -------------------- PATHS / CONFIG --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
# model filename you stated you have
MODEL_PATH = os.path.join(BASE_DIR, "lcd_ocr_model_fixed.pt")
YAML_PATH = os.path.join(BASE_DIR, "data.yaml")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# -------------------- LOAD YOLO MODEL --------------------
MODEL = None
CLASS_NAMES = []
try:
    MODEL = YOLO(MODEL_PATH)
    if os.path.exists(YAML_PATH):
        with open(YAML_PATH, "r") as f:
            data_cfg = yaml.safe_load(f)
            CLASS_NAMES = data_cfg.get("names", [])
except Exception as e:
    logging.warning("YOLO model failed to load at startup: %s", e)
    MODEL = None
    if os.path.exists(YAML_PATH):
        with open(YAML_PATH, "r") as f:
            data_cfg = yaml.safe_load(f)
            CLASS_NAMES = data_cfg.get("names", [])

# -------------------- GLOBALS --------------------
TASK_PROGRESS: Dict[str, Dict] = {}
SESSIONS: Dict[str, Dict] = {}

# -------------------- UTIL --------------------
def reset_progress(task_id: str) -> None:
    TASK_PROGRESS[task_id] = {"status": "pending", "progress": 0, "phase": "queued", "message": ""}

def get_progress(task_id: str) -> Dict:
    return TASK_PROGRESS.get(task_id, {"status": "unknown", "progress": 0, "phase": "unknown"})

# -------------------- FRAME EXTRACTION --------------------
def _ffmpeg_extract(video_path: str, out_dir: str, fps: int) -> None:
    (
        ffmpeg
        .input(video_path)
        .filter("fps", fps=fps)
        .output(os.path.join(out_dir, "frame_%06d.jpg"), start_number=0, qscale=2, fps_mode="vfr")
        .overwrite_output()
        .run(quiet=not DEBUG)
    )

def _opencv_fallback_extract(video_path: str, out_dir: str, fps: int) -> None:
    cap = cv2.VideoCapture(video_path)
    real_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 0:
        fps = 1
    ratio = real_fps / float(fps)
    ratio = max(1.0, ratio)
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
    TASK_PROGRESS[task_id].update({"phase": "extracting", "message": f"Extracting frames ({fps} FPS)..."} )
    try:
        _ffmpeg_extract(video_path, frames_dir, fps=fps)
    except Exception as e:
        TASK_PROGRESS[task_id].update({"message": f"FFmpeg failed, fallback OpenCV. ({e})"})
        _opencv_fallback_extract(video_path, frames_dir, fps=fps)
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")])
    if len(frame_files) < 5:
        # fallback
        _opencv_fallback_extract(video_path, frames_dir, fps=fps)
        frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")])

    # downsample if absurdly many frames
    MAX_FRAMES = 6000
    if len(frame_files) > MAX_FRAMES:
        step = math.ceil(len(frame_files) / MAX_FRAMES)
        frame_files = frame_files[::step]
        TASK_PROGRESS[task_id].update({"message": f"Downsampled frames to {len(frame_files)}."})

    _dprint(f"[DEBUG] Extracted {len(frame_files)} frames at {fps} FPS")
    return frame_files

# -------------------- PREPROCESS --------------------
def preprocess_frame(img: Optional[np.ndarray], invert: bool = False) -> Optional[np.ndarray]:
    if img is None:
        return None
    h, w = img.shape[:2]
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

# -------------------- OCR CORE --------------------
def _predict_reading(img: np.ndarray, conf: float) -> Tuple[str, float]:
    if MODEL is None:
        return "", 0.0
    results = MODEL.predict(img, conf=conf, verbose=False)
    if not results:
        return "", 0.0
    res0 = results[0]
    boxes = []
    for b in getattr(res0, "boxes", []):
        try:
            cls_idx = int(b.cls[0].item())
            ch = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else ""
        except Exception:
            ch = ""
        # allow digits, ., -, and common labels
        if ch not in "0123456789.-gG aA":
            continue
        try:
            xc = float(b.xywh[0][0].item())
        except Exception:
            xc = 0.0
        conf_s = float(b.conf[0].item()) if hasattr(b, "conf") else 0.0
        boxes.append((xc, ch, conf_s))
    if not boxes:
        return "", 0.0
    boxes.sort(key=lambda x: x[0])
    reading = "".join(ch for _, ch, _ in boxes)
    if reading.count(".") > 1:
        first = reading.find(".")
        reading = reading[:first + 1] + reading[first + 1:].replace(".", "")
    avg_conf = float(np.mean([s for _, _, s in boxes])) if boxes else 0.0
    return reading.strip(), avg_conf

# -------------------- READERS --------------------
def read_lcd_current(frame: np.ndarray) -> Tuple[Optional[float], str, float]:
    """Read current and format display as -xx.xx (two digits before decimal)"""
    h, w = frame.shape[:2]
    proc = preprocess_frame(frame, invert=False)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inv_binary = 255 - binary

    reading, conf = _predict_reading(proc, 0.25)
    reading = (reading or "").strip()
    reading = reading.replace("..", ".").replace("—", "-").replace("_", "-").replace(" ", "")
    if reading.endswith("."):
        reading = reading[:-1]
    if reading.lower().endswith("a"):
        reading = reading[:-1]

    sign_found = "-" in reading
    if not sign_found:
        left_w = max(12, int(w * 0.18))
        band = inv_binary[:, :left_w]
        contours, _ = cv2.findContours(band, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = cw / (ch + 1e-6)
            if aspect > 3.0 and ch < h * 0.2 and cw > 3:
                reading = "-" + reading
                break

    cleaned = "".join(ch for ch in reading if ch in "0123456789.-")
    numeric = None
    try:
        if cleaned not in ["", "-", ".", "-."]:
            numeric = float(cleaned)
    except Exception:
        numeric = None

    if numeric is not None:
        # ensure two digits before decimal and two after
        if numeric < 0:
            # -00.69 format
            display = f"-{abs(numeric):06.2f}"
        else:
            display = f"{numeric:06.2f}"
    else:
        display = reading or ""

    return numeric, display, float(conf or 0.0)

def read_lcd_thrust(frame: np.ndarray) -> Tuple[Optional[float], str, float]:
    inverted = preprocess_frame(frame, invert=True)
    reading, conf = _predict_reading(inverted, 0.25)
    reading = (reading or "").strip()
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
    if "." in cleaned:
        parts = cleaned.split(".")
        left = parts[0] if parts else ""
        right = "".join(parts[1:]) if len(parts) > 1 else ""
        if len(right) == 1 and len(left) >= 1:
            cleaned = left + right
        elif len(right) >= 2:
            cleaned = left + right

    cleaned = "".join(ch for ch in cleaned if ch.isdigit() or ch == "-")
    if cleaned in ["", "-", "-0"]:
        cleaned = "0"

    numeric = None
    try:
        numeric = float(cleaned)
    except Exception:
        numeric = None

    MAX_ALLOWED = 10000
    MIN_NOISE = 2

    if numeric is None:
        return None, "", float(conf or 0.0)

    if abs(numeric) > MAX_ALLOWED:
        numeric = 0.0
    if abs(numeric) < MIN_NOISE:
        numeric = 0.0
    else:
        numeric = round(float(numeric))

    if not hasattr(read_lcd_thrust, "stable_vals"):
        read_lcd_thrust.stable_vals = deque(maxlen=6)
        read_lcd_thrust.last_val = numeric

    last = getattr(read_lcd_thrust, "last_val", 0.0)
    read_lcd_thrust.stable_vals.append(numeric)
    avg_val = float(np.mean(read_lcd_thrust.stable_vals)) if read_lcd_thrust.stable_vals else numeric

    if (abs(numeric - avg_val) > max(300, 0.5 * abs(avg_val))) and (conf < 0.7):
        numeric = last
    if (last and abs(numeric - last) > max(2000, last * 2)):
        numeric = last

    read_lcd_thrust.last_val = numeric

    display = str(int(numeric)) if numeric is not None else ""
    return numeric, display, float(conf or 0.0)

def read_lcd_rpm(frame: np.ndarray) -> Tuple[Optional[float], str, float]:
    processed = preprocess_frame(frame, invert=False)
    reading, conf = _predict_reading(processed, 0.30)
    reading = (reading or "").strip()
    reading = reading.replace(" ", "").replace(",", "")
    cleaned = "".join(ch for ch in reading if ch.isdigit() or ch == "-")
    try:
        val = float(cleaned) if cleaned not in ["", "-", "."] else None
    except Exception:
        val = None
    display = str(int(val)) if val is not None else ""
    return val, display, float(conf or 0.0)

# -------------------- MERGING / REPORT --------------------
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
    return merged.sort_values("time_s").reset_index(drop=True)

def _sanitize_value_for_json(v):
    # Convert numpy scalar to python float/int, and handle NaN/Inf
    try:
        if isinstance(v, (np.floating, float)):
            if not math.isfinite(float(v)):
                return None
            return float(v)
        if isinstance(v, (np.integer, int)):
            return int(v)
        # numpy types
        if isinstance(v, np.ndarray):
            # shouldn't happen for scalar fields but guard
            return v.tolist()
        # native
        if v is None:
            return None
        if isinstance(v, str):
            return v
        # other numeric-like (pandas NA etc)
        try:
            f = float(v)
            if math.isfinite(f):
                # prefer int if it is integer-like
                if abs(f - int(f)) < 1e-9:
                    return int(f)
                return f
            else:
                return None
        except Exception:
            return str(v)
    except Exception:
        return None

def build_session_report(session_id: str) -> Dict:
    sess = SESSIONS.get(session_id)
    if not sess:
        return {"table_csv": None, "graphs": [], "table_records": []}
    meta = sess["meta"]
    merged = _safe_merge_series(sess["series"].get("current"), sess["series"].get("thrust"), sess["series"].get("rpm"))

    if merged.empty:
        sess["report"] = {"table_csv": None, "graphs": [], "table_records": []}
        return sess["report"]

    # numeric conversions
    for c in ["current_a", "thrust_g", "rpm"]:
        if c in merged.columns:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")

    voltage = meta.get("voltage")
    try:
        if voltage not in [None, ""]:
            merged["power_w"] = merged["current_a"] * float(voltage)
        else:
            merged["power_w"] = np.nan
    except Exception:
        merged["power_w"] = np.nan

    # Avoid dividing by zero; replace 0 -> NaN temporarily so efficiency becomes NaN
    merged["efficiency_gw"] = merged["thrust_g"] / merged["power_w"].replace({0: np.nan})

    # Save CSV (raw numeric values)
    csv_path = os.path.join(RESULT_DIR, f"{session_id}_report.csv")
    merged.to_csv(csv_path, index=False)

    # replace inf and nan with 0 for plotting convenience
    merged_plot = merged.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Plotly interactive graph (hover shows values)
    graph_path = os.path.join(RESULT_DIR, f"{session_id}_interactive.html")
    fig = go.Figure()
    if "current_a" in merged_plot.columns and merged_plot["current_a"].notna().any():
        fig.add_trace(go.Scatter(x=merged_plot["time_s"], y=merged_plot["current_a"], mode="lines+markers", name="Current (A)", hovertemplate="Time %{x}s<br>Current: %{y} A"))
    if "thrust_g" in merged_plot.columns and merged_plot["thrust_g"].notna().any():
        fig.add_trace(go.Scatter(x=merged_plot["time_s"], y=merged_plot["thrust_g"], mode="lines+markers", name="Thrust (G)", hovertemplate="Time %{x}s<br>Thrust: %{y} g"))
    if "rpm" in merged_plot.columns and merged_plot["rpm"].notna().any():
        fig.add_trace(go.Scatter(x=merged_plot["time_s"], y=merged_plot["rpm"], mode="lines+markers", name="RPM", hovertemplate="Time %{x}s<br>RPM: %{y}"))

    fig.update_layout(title=f"Session {session_id} — Live Graph", xaxis_title="Time (s)", yaxis_title="Values", hovermode="x unified", template="plotly_white")
    try:
        fig.write_html(graph_path, include_plotlyjs="cdn")
    except Exception as e:
        logging.warning("Failed to write interactive graph: %s", e)

    # Build JSON-safe table records for frontend
    records = []
    for rec in merged.to_dict(orient="records"):
        safe = {}
        for k, v in rec.items():
            safe_val = _sanitize_value_for_json(v)
            safe[k] = safe_val
        # convert to frontend layout
        frontend_row = {
            "Time (s)": _sanitize_value_for_json(safe.get("time_s", 0)),
            "Voltage (V)": meta.get("voltage", None),
            "Prop": meta.get("prop", ""),
            "Motor": meta.get("motor", ""),
            "ESC": meta.get("esc", ""),
            "Throttle": "",
            # Current (A) — keep two decimals for display in table if present
            "Current (A)": (round(float(safe["current_a"]), 2) if safe.get("current_a") is not None else None),
            "Power (W)": (round(float(safe["power_w"]), 2) if safe.get("power_w") is not None else None),
            "Thrust (G)": (int(round(float(safe["thrust_g"]))) if safe.get("thrust_g") is not None else None),
            "RPM": (int(round(float(safe["rpm"]))) if safe.get("rpm") is not None else None),
            "Efficiency (G/W)": (round(float(safe["efficiency_gw"]), 3) if safe.get("efficiency_gw") is not None else None),
            "Operating Temperature (°C)": "",
        }
        records.append(frontend_row)

    sess["report"] = {"table_csv": csv_path, "graphs": [graph_path], "table_records": records}
    return sess["report"]

# -------------------- MAIN PROCESS --------------------
def _init_session(session_id: str, meta: Dict) -> None:
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "meta": {
                "prop": meta.get("prop", ""),
                "motor": meta.get("motor", ""),
                "esc": meta.get("esc", ""),
                "voltage": float(meta.get("voltage")) if meta.get("voltage") not in [None, ""] else None,
            },
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

def process_video_task(task_id: str, session_id: str, video_type: str, video_path: str, meta: Dict) -> None:
    try:
        reset_progress(task_id)
        TASK_PROGRESS[task_id].update({"status": "running", "phase": "init", "progress": 0})
        _init_session(session_id, meta)
        fps = int(meta.get("fps", 5))
        frames = extract_frames_custom(task_id, video_path, fps)
        rows = []
        recent = deque(maxlen=7)
        prev_raw = None

        # warm model
        if MODEL:
            try:
                MODEL.predict(np.zeros((64, 64, 3), np.uint8), verbose=False)
            except Exception:
                pass

        total = max(1, len(frames))
        for i, fp in enumerate(frames):
            img = cv2.imread(fp)
            if img is None:
                continue

            if video_type == "current":
                val, raw, conf = read_lcd_current(img)
                disp_key = "current_display"
                numeric_key = "current_a"
            elif video_type == "thrust":
                val, raw, conf = read_lcd_thrust(img)
                disp_key = "thrust_display"
                numeric_key = "thrust_g"
            else:
                val, raw, conf = read_lcd_rpm(img)
                disp_key = "rpm_display"
                numeric_key = "rpm"

            raw = raw or ""
            conf = float(conf or 0.0)
            recent.append(raw)
            non_empty = [r for r in recent if r]
            chosen = non_empty[0] if non_empty else raw
            if non_empty:
                counts = Counter(non_empty)
                chosen = counts.most_common(1)[0][0]

            if non_empty:
                negs = sum(1 for r in non_empty if str(r).startswith("-"))
                if negs >= max(1, len(non_empty)//2) and not str(chosen).startswith("-"):
                    chosen = "-" + chosen

            if prev_raw and str(prev_raw).startswith("-") and not str(chosen).startswith("-"):
                digits_prev = "".join(c for c in str(prev_raw) if c.isdigit() or c == ".")
                digits_now = "".join(c for c in str(chosen) if c.isdigit() or c == ".")
                if digits_prev == digits_now:
                    chosen = "-" + chosen

            prev_raw = chosen if chosen else prev_raw

            numeric_val = None
            try:
                cleaned = "".join(ch for ch in str(chosen) if ch in "0123456789.-")
                if cleaned not in ["", "-", ".", "-."]:
                    numeric_val = float(cleaned)
            except Exception:
                numeric_val = None

            if numeric_val is not None:
                if video_type == "thrust":
                    if abs(numeric_val) > 20000:
                        numeric_val = getattr(process_video_task, f"last_val_{video_type}", 0)
                    if abs(numeric_val) < 2:
                        numeric_val = 0.0
                    else:
                        numeric_val = round(float(numeric_val))
                elif video_type == "current":
                    last = getattr(process_video_task, f"last_val_{video_type}", None)
                    if last is not None and abs(numeric_val - last) > 5:
                        numeric_val = last
                    else:
                        setattr(process_video_task, f"last_val_{video_type}", numeric_val)
                else:  # rpm
                    numeric_val = abs(int(round(float(numeric_val))))
                    last = getattr(process_video_task, f"last_val_{video_type}", None)
                    if last is not None and abs(numeric_val - last) > max(500, last * 2):
                        numeric_val = last
                    else:
                        setattr(process_video_task, f"last_val_{video_type}", numeric_val)

            t = round(i * (1.0 / float(max(1, fps))), 4)
            rows.append({
                "time_s": t,
                disp_key: chosen,
                numeric_key: numeric_val
            })

            TASK_PROGRESS[task_id]["progress"] = min(85, int(((i+1)/total) * 85))
            _dprint(f"[DEBUG] Frame {i+1}/{total} | Type:{video_type} | Raw:'{raw}' | Chosen:'{chosen}' | Conf:{conf:.2f} | Val:{numeric_val}")

        df = pd.DataFrame(rows)
        if video_type == "current":
            df["current_a"] = pd.to_numeric(df.get("current_a", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        elif video_type == "thrust":
            df["thrust_g"] = pd.to_numeric(df.get("thrust_g", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
        else:
            df["rpm"] = pd.to_numeric(df.get("rpm", pd.Series(dtype=float)), errors="coerce").fillna(0.0)

        existing = SESSIONS[session_id]["series"].get(video_type)
        if existing is not None and not existing.empty:
            combined = pd.concat([existing, df], ignore_index=True).drop_duplicates(subset=["time_s"], keep="last").sort_values("time_s").reset_index(drop=True)
            SESSIONS[session_id]["series"][video_type] = combined
        else:
            SESSIONS[session_id]["series"][video_type] = df

        TASK_PROGRESS[task_id].update({"phase": "report", "progress": 90})
        report = build_session_report(session_id)

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

    except Exception as e:
        logging.exception("Error while processing video")
        TASK_PROGRESS[task_id].update({"status": "error", "message": str(e)})
    finally:
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception:
            pass

# -------------------- BACKGROUND --------------------
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
