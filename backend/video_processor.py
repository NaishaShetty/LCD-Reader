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

# -------------------- PATHS & MODEL -------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(BASE_DIR, "lcd_ocr_model.pt")
YAML_PATH = os.path.join(BASE_DIR, "data.yaml")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

MODEL = YOLO(MODEL_PATH)
with open(YAML_PATH, "r") as f:
    DATA_CFG = yaml.safe_load(f)
CLASS_NAMES = DATA_CFG.get("names", [])

TASK_PROGRESS: Dict[str, Dict] = {}
SESSIONS: Dict[str, Dict] = {}

# -------------------- SESSION MANAGEMENT -------------------- #
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

def reset_progress(task_id: str) -> None:
    TASK_PROGRESS[task_id] = {"status": "pending", "progress": 0, "phase": "queued", "message": ""}

def get_progress(task_id: str) -> Dict:
    return TASK_PROGRESS.get(task_id, {"status": "unknown", "progress": 0, "phase": "unknown"})

# -------------------- FRAME EXTRACTION -------------------- #
def _ffmpeg_extract(video_path: str, out_dir: str, fps: int) -> None:
    """
    Use vsync='vfr' to avoid dropping frames on VFR recordings.
    """
    (
        ffmpeg
        .input(video_path)
        .filter("fps", fps=fps)
        .output(
            os.path.join(out_dir, "frame_%06d.jpg"),
            start_number=0,
            qscale=2,
            vsync="vfr"
        )
        .overwrite_output()
        .run(quiet=True)
    )

def _opencv_fallback_extract(video_path: str, out_dir: str, fps: int) -> None:
    cap = cv2.VideoCapture(video_path)
    real_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1.0, real_fps / float(max(1, fps)))
    frame_idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if int(round(frame_idx % frame_interval)) == 0:
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

    # If too few frames, run fallback to ensure coverage
    if len(frame_files) < 5:
        _opencv_fallback_extract(video_path, frames_dir, fps=fps)
        frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.lower().endswith(".jpg")])

    print(f"[DEBUG] extract_frames_custom: extracted {len(frame_files)} frames for task {task_id} at {fps} fps")
    return frame_files

# -------------------- PREPROCESSING -------------------- #
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

# -------------------- OCR -------------------- #
def _predict_reading(img: np.ndarray, conf: float) -> Tuple[str, float]:
    results = MODEL.predict(img, conf=conf, verbose=False)[0]
    boxes: List[Tuple[float, str, float]] = []
    for b in results.boxes:
        cls = int(b.cls[0].item())
        xc = float(b.xywh[0][0].item())
        ch = CLASS_NAMES[cls]
        score = float(b.conf[0].item()) if hasattr(b, "conf") else 0.0
        boxes.append((xc, ch, score))
    if not boxes:
        return "", 0.0
    boxes.sort(key=lambda x: x[0])
    reading = "".join([ch for _, ch, _ in boxes if ch in "0123456789.-"])
    confs = [s for _, _, s in boxes if s is not None]
    avg_conf = float(np.mean(confs)) if confs else 0.0
    if reading.count(".") > 1:
        i = reading.find(".")
        reading = reading[:i + 1] + reading[i + 1:].replace(".", "")
    reading = reading.strip()
    return reading, avg_conf

def read_lcd_from_frame(frame: np.ndarray, conf: float = 0.25) -> Tuple[Optional[float], str, float]:
    processed_normal = preprocess_frame(frame, invert=False)
    processed_inverted = preprocess_frame(frame, invert=True)

    reading_n, conf_n = _predict_reading(processed_normal, conf)
    reading_i, conf_i = _predict_reading(processed_inverted, conf)

    if conf_i > conf_n:
        reading = reading_i
        confidence = conf_i
    else:
        reading = reading_n
        confidence = conf_n

    if confidence < 0.2 and any(ch.isdigit() for ch in reading):
        confidence = 0.25

    if reading.startswith("--"):
        reading = "-" + reading.lstrip("-")
    if reading.endswith("-"):
        reading = reading[:-1]

    reading = reading.strip()
    try:
        val = float(reading) if reading not in ["", "-", "."] else None
    except Exception:
        val = None

    return val, reading, confidence

# -------------------- MERGING -------------------- #
def _safe_merge_series(cur: Optional[pd.DataFrame], thr: Optional[pd.DataFrame], rpm: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge three series into a single DataFrame with aligned time_s.
    Ensures each column (current_a, thrust_g, rpm) sits in its own column correctly.
    """
    # collect all times
    times = set()
    for df in (cur, thr, rpm):
        if df is not None and not df.empty and "time_s" in df.columns:
            times.update(df["time_s"].tolist())
    if not times:
        return pd.DataFrame(columns=["time_s"])

    merged = pd.DataFrame({"time_s": sorted(times)})

    if cur is not None and not cur.empty:
        merged = merged.merge(cur[["time_s", "current_a", "current_display"]] if "current_display" in cur.columns else cur[["time_s", "current_a"]],
                              on="time_s", how="left")
    else:
        merged["current_a"] = np.nan
        merged["current_display"] = np.nan

    if thr is not None and not thr.empty:
        merged = merged.merge(thr[["time_s", "thrust_g", "thrust_display"]] if "thrust_display" in thr.columns else thr[["time_s", "thrust_g"]],
                              on="time_s", how="left")
    else:
        merged["thrust_g"] = np.nan
        merged["thrust_display"] = np.nan

    if rpm is not None and not rpm.empty:
        merged = merged.merge(rpm[["time_s", "rpm", "rpm_display"]] if "rpm_display" in rpm.columns else rpm[["time_s", "rpm"]],
                              on="time_s", how="left")
    else:
        merged["rpm"] = np.nan
        merged["rpm_display"] = np.nan

    # Ensure display columns exist
    for col in ["current_display", "thrust_display", "rpm_display"]:
        if col not in merged.columns:
            merged[col] = np.nan

    return merged

def _align_time(df: Optional[pd.DataFrame], fps: int = 5) -> Optional[pd.DataFrame]:
    if df is None or df.empty or "time_s" not in df.columns:
        return df
    df["time_s"] = df["time_s"].round(2)
    return df

# -------------------- REPORT BUILDING -------------------- #
def build_session_report(session_id: str) -> Dict:
    sess = SESSIONS[session_id]
    meta = sess["meta"]

    cur_df = _align_time(sess["series"]["current"])
    thr_df = _align_time(sess["series"]["thrust"])
    rpm_df = _align_time(sess["series"]["rpm"])

    merged = _safe_merge_series(cur_df, thr_df, rpm_df)

    if merged is None or merged.empty:
        sess["report"] = {"table_csv": None, "graphs": [], "table_records": []}
        return sess["report"]

    merged = merged.sort_values("time_s")

    voltage = meta.get("voltage")
    if voltage is not None and "current_a" in merged.columns:
        merged["power_w"] = merged["current_a"] * float(voltage)
    else:
        merged["power_w"] = np.nan
    if "thrust_g" in merged.columns and "power_w" in merged.columns:
        merged["efficiency_gw"] = merged["thrust_g"] / merged["power_w"]
    else:
        merged["efficiency_gw"] = np.nan

    # For display columns: prefer the display string if available, else format numeric with zero padding if needed
    def display_series(merged_df, numeric_col, display_col, default="-"):
        if display_col in merged_df.columns and merged_df[display_col].notna().any():
            return merged_df[display_col].fillna(default).tolist()
        else:
            # format numeric values to preserve two decimals if they have decimals, preserve leading zeros by formatting with 2 decimals
            if numeric_col in merged_df.columns:
                out = []
                for v in merged_df[numeric_col].tolist():
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        out.append(default)
                    else:
                        # if number is integer-like, show as integer string without decimals when original had none
                        # but since we don't know original format, keep two decimals for decimals
                        if isinstance(v, float) and (abs(v - round(v)) > 1e-9):
                            out.append(f"{v:.2f}")
                        else:
                            # integer-like
                            out.append(str(int(round(v))))
                return out
            else:
                return [default for _ in range(len(merged_df))]

    display_current = display_series(merged, "current_a", "current_display")
    display_thrust = display_series(merged, "thrust_g", "thrust_display")
    display_rpm = display_series(merged, "rpm", "rpm_display")

    pretty = pd.DataFrame({
        "Time (s)": merged.get("time_s", pd.Series(dtype=float)),
        "Voltage (V)": [meta["voltage"] if meta["voltage"] is not None else "-" for _ in range(len(merged))],
        "Prop": [meta["prop"] or "-" for _ in range(len(merged))],
        "Motor": [meta["motor"] or "-" for _ in range(len(merged))],
        "ESC": [meta["esc"] or "-" for _ in range(len(merged))],
        "Throttle": ["-" for _ in range(len(merged))],
        "Current (A)": display_current,
        "Power (W)": merged.get("power_w", pd.Series(dtype=float)),
        "Thrust (G)": display_thrust,
        "RPM": display_rpm,
        "Efficiency (G/W)": merged.get("efficiency_gw", pd.Series(dtype=float)),
        "Operating Temperature (°C)": ["-" for _ in range(len(merged))],
    })

    pretty = pretty.replace([np.inf, -np.inf], np.nan)
    pretty = pretty.where(pd.notnull(pretty), "-")

    csv_path = os.path.join(RESULT_DIR, f"{session_id}_report.csv")
    pretty.to_csv(csv_path, index=False)

    graphs: List[str] = []
    def _plot(series_key: str, ylabel: str, filename: str) -> None:
        if series_key not in merged.columns:
            return
        ser = merged[["time_s", series_key]].dropna()
        if ser.empty:
            return
        plt.figure(figsize=(12, 4))
        plt.plot(ser["time_s"], ser[series_key], marker="o", markersize=3)
        plt.xlabel("Time (s)")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs Time")
        plt.grid(True, alpha=0.3)
        out = os.path.join(RESULT_DIR, f"{session_id}_{filename}.png")
        plt.savefig(out, bbox_inches="tight")
        plt.close()
        graphs.append(out)

    _plot("current_a", "Current (A)", "current")
    _plot("thrust_g", "Thrust (G)", "thrust")
    _plot("rpm", "RPM", "rpm")
    _plot("power_w", "Power (W)", "power")
    _plot("efficiency_gw", "Efficiency (G/W)", "efficiency")

    sess["report"] = {"table_csv": csv_path, "graphs": graphs, "table_records": pretty.to_dict(orient="records")}
    return sess["report"]

# -------------------- MAIN PROCESS -------------------- #
def process_video_task(task_id: str, session_id: str, video_type: str, video_path: str, meta: Dict) -> None:
    """
    Process video:
    - run dual-pass OCR per frame
    - keep stabilized display string for each frame (display column)
    - keep numeric value for plotting
    - append rows with time_s, numeric and display columns
    """
    try:
        reset_progress(task_id)
        TASK_PROGRESS[task_id].update({"status": "running", "phase": "init", "progress": 1})
        _init_session(session_id, meta)
        fps = int(meta.get("fps", 5))
        frames = extract_frames_custom(task_id, video_path, fps=fps)
        n = len(frames)
        if n == 0:
            TASK_PROGRESS[task_id].update({"status": "error", "message": "No frames extracted"})
            return

        TASK_PROGRESS[task_id].update({"phase": "reading", "message": "Reading frames..."})
        rows: List[Dict] = []
        recent_readings = deque(maxlen=7)

        for i, fp in enumerate(frames):
            img = cv2.imread(fp)
            if img is None:
                TASK_PROGRESS[task_id]["progress"] = int(((i + 1) / n) * 90)
                continue

            val, raw, confidence = read_lcd_from_frame(img, conf=0.22)
            recent_readings.append(raw)

            # Stabilize by mode of recent non-empty strings (require count>=2 to accept)
            non_empty = [r for r in recent_readings if r not in ["", None]]
            chosen_raw = None
            if non_empty:
                counter = Counter(non_empty)
                most_common, count = counter.most_common(1)[0]
                if count >= 2:
                    chosen_raw = most_common
            if chosen_raw is None:
                chosen_raw = raw

            try:
                chosen_val = float(chosen_raw) if chosen_raw not in ["", "-", "."] else None
            except Exception:
                chosen_val = None

            t = i * (1.0 / fps)

            # store both numeric and raw display columns
            if video_type == "current":
                rows.append({"time_s": round(t, 4), "current_a": chosen_val, "current_display": chosen_raw})
            elif video_type == "thrust":
                rows.append({"time_s": round(t, 4), "thrust_g": chosen_val, "thrust_display": chosen_raw})
            elif video_type == "rpm":
                rows.append({"time_s": round(t, 4), "rpm": chosen_val, "rpm_display": chosen_raw})

            TASK_PROGRESS[task_id]["progress"] = int(((i + 1) / n) * 90)

        df = pd.DataFrame(rows)

        # Ensure consistent columns and smoothing for numeric columns
        if not df.empty:
            if "current_a" in df.columns:
                df["current_a"] = df["current_a"].astype(float, errors="ignore")
                df["current_a"] = df["current_a"].rolling(window=3, min_periods=1, center=True).median()
            if "thrust_g" in df.columns:
                df["thrust_g"] = df["thrust_g"].astype(float, errors="ignore")
                df["thrust_g"] = df["thrust_g"].rolling(window=3, min_periods=1, center=True).median()
            if "rpm" in df.columns:
                df["rpm"] = df["rpm"].astype(float, errors="ignore")
                df["rpm"] = df["rpm"].rolling(window=3, min_periods=1, center=True).median()

        series_csv = os.path.join(RESULT_DIR, f"{task_id}_{video_type}.csv")
        df.to_csv(series_csv, index=False)

        # Attach series to session (keep as-is: may have time_s and other columns)
        SESSIONS[session_id]["series"][video_type] = df

        TASK_PROGRESS[task_id].update({"phase": "merging", "message": "Merging & plotting...", "progress": 95})
        report = build_session_report(session_id)
        sess = SESSIONS[session_id]

        save_test_result(
            session_id=session_id,
            prop_name=sess["meta"].get("prop", ""),
            motor_name=sess["meta"].get("motor", ""),
            esc_name=sess["meta"].get("esc", ""),
            voltage=sess["meta"].get("voltage"),
            csv_path=report.get("table_csv"),
            graph_paths=report.get("graphs", []),
            table_data=report.get("table_records", []),
        )

        TASK_PROGRESS[task_id].update({"status": "done", "phase": "done", "progress": 100, "message": "Completed"})

    except Exception as e:
        print(f"[ERROR] process_video_task: {e}")
        TASK_PROGRESS[task_id].update({"status": "error", "message": f"{type(e).__name__}: {str(e)}"})
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
