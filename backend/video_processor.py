import os
import uuid
import threading
import shutil
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
from database import save_test_result

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

def _ffmpeg_extract(video_path: str, out_dir: str, fps: int) -> None:
    (
        ffmpeg
        .input(video_path)
        .filter("fps", fps=fps)
        .output(os.path.join(out_dir, "frame_%06d.jpg"), start_number=0, qscale=2)
        .overwrite_output()
        .run(quiet=True)
    )

def _opencv_fallback_extract(video_path: str, out_dir: str, fps: int) -> None:
    cap = cv2.VideoCapture(video_path)
    real_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(real_fps / fps)))
    count, saved = 0, 0
    ok, frame = cap.read()
    while ok:
        if count % step == 0:
            cv2.imwrite(os.path.join(out_dir, f"frame_{saved:06d}.jpg"), frame)
            saved += 1
        count += 1
        ok, frame = cap.read()
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
    return frame_files

def preprocess_frame(img: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if img is None:
        return None
    h, w = img.shape[:2]
    if h > w * 1.15:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

def read_lcd_from_frame(frame: np.ndarray, conf: float = 0.25) -> Tuple[Optional[float], str]:
    processed = preprocess_frame(frame)
    if processed is None:
        return None, ""
    results = MODEL.predict(processed, conf=conf, verbose=False)[0]
    boxes: List[Tuple[float, str]] = []
    for b in results.boxes:
        cls = int(b.cls[0].item())
        xc = float(b.xywh[0][0].item())
        ch = CLASS_NAMES[cls]
        boxes.append((xc, ch))
    boxes.sort(key=lambda x: x[0])
    reading = "".join([ch for _, ch in boxes if ch in "0123456789.-"])
    if reading.count(".") > 1:
        i = reading.find(".")
        reading = reading[:i + 1] + reading[i + 1:].replace(".", "")
    try:
        val = float(reading) if reading else None
    except (ValueError, AttributeError):
        val = None
    return val, reading

def _safe_merge(left: Optional[pd.DataFrame], right: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if left is None and right is None:
        return None
    if left is None:
        return right.copy()
    if right is None:
        return left.copy()
    return pd.merge(left, right, on="time_s", how="outer")

def build_session_report(session_id: str) -> Dict:
    sess = SESSIONS[session_id]
    meta = sess["meta"]
    cur_df = sess["series"]["current"]
    thr_df = sess["series"]["thrust"]
    rpm_df = sess["series"]["rpm"]
    merged = _safe_merge(cur_df, thr_df)
    merged = _safe_merge(merged, rpm_df)
    if merged is None or merged.empty:
        sess["report"] = {"table": [], "graphs": [], "csv_url": None, "table_records": []}
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
    pretty = pd.DataFrame({
        "Time (s)": merged.get("time_s", pd.Series(dtype=float)),
        "Voltage (V)": [meta["voltage"] if meta["voltage"] is not None else "-" for _ in range(len(merged))],
        "Prop": [meta["prop"] or "-" for _ in range(len(merged))],
        "Motor": [meta["motor"] or "-" for _ in range(len(merged))],
        "ESC": [meta["esc"] or "-" for _ in range(len(merged))],
        "Throttle": ["-" for _ in range(len(merged))],
        "Current (A)": merged.get("current_a", pd.Series(dtype=float)),
        "Power (W)": merged.get("power_w", pd.Series(dtype=float)),
        "Thrust (G)": merged.get("thrust_g", pd.Series(dtype=float)),
        "RPM": merged.get("rpm", pd.Series(dtype=float)),
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

def process_video_task(task_id: str, session_id: str, video_type: str, video_path: str, meta: Dict) -> None:
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
        for i, fp in enumerate(frames):
            img = cv2.imread(fp)
            val, raw = read_lcd_from_frame(img, conf=0.22)
            t = i * (1.0 / fps)
            if video_type == "current":
                rows.append({"time_s": t, "current_a": val})
            elif video_type == "thrust":
                rows.append({"time_s": t, "thrust_g": val})
            elif video_type == "rpm":
                rows.append({"time_s": t, "rpm": val})
            TASK_PROGRESS[task_id]["progress"] = int(((i + 1) / n) * 90)
        df = pd.DataFrame(rows)
        if not df.empty and video_type in ["current", "thrust", "rpm"]:
            col = "current_a" if video_type == "current" else "thrust_g" if video_type == "thrust" else "rpm"
            df[col] = df[col].where(df[col].between(df[col].quantile(0.05), df[col].quantile(0.95)))
            df[col] = df[col].rolling(window=3, min_periods=1, center=True).median()
        series_csv = os.path.join(RESULT_DIR, f"{task_id}_{video_type}.csv")
        df.to_csv(series_csv, index=False)
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
            table_data=report.get("table_records", [])
        )
        TASK_PROGRESS[task_id].update({"status": "done", "phase": "done", "progress": 100, "message": "Completed"})
    except Exception as e:
        TASK_PROGRESS[task_id].update({"status": "error", "message": f"{type(e).__name__}: {str(e)}"})
    finally:
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except Exception:
            pass

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
