#!/usr/bin/env python3
# ============================================================
# video_processor.py — Final fixed version
# - Fixes KeyError when not all videos are uploaded
# - Keeps -xx.xx format for current
# - Generates interactive graphs (current, thrust, rpm, power, efficiency)
# - Generates PDF report
# - JSON-safe, 100% progress completion
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
import plotly.graph_objects as go
from typing import Dict, Optional, List, Tuple
from ultralytics import YOLO
import yaml
from collections import deque, Counter
from database import save_test_result
import math
import logging
from fpdf import FPDF

# -------------------- LOGGING --------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
DEBUG = True

def _dprint(*args):
    if DEBUG:
        logging.debug(" ".join(str(a) for a in args))

# -------------------- PATHS --------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
RESULT_DIR = os.path.join(BASE_DIR, "results")
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
    logging.warning("YOLO model failed to load: %s", e)
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
        ffmpeg.input(video_path)
        .filter("fps", fps=fps)
        .output(os.path.join(out_dir, "frame_%06d.jpg"), start_number=0, qscale=2)
        .overwrite_output()
        .run(quiet=not DEBUG)
    )

def _opencv_fallback_extract(video_path: str, out_dir: str, fps: int) -> None:
    cap = cv2.VideoCapture(video_path)
    real_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ratio = max(1.0, real_fps / float(max(fps, 1)))
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
        _ffmpeg_extract(video_path, frames_dir, fps)
    except Exception as e:
        TASK_PROGRESS[task_id].update({"message": f"FFmpeg failed, using OpenCV ({e})"})
        _opencv_fallback_extract(video_path, frames_dir, fps)
    files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    if not files:
        _opencv_fallback_extract(video_path, frames_dir, fps)
        files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    return files

# -------------------- PREPROCESS --------------------
def preprocess_frame(img: np.ndarray, invert: bool=False) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if invert:
        gray = cv2.bitwise_not(gray)
    den = cv2.bilateralFilter(gray, 9, 75, 75)
    thr = cv2.adaptiveThreshold(den,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    return cv2.cvtColor(thr, cv2.COLOR_GRAY2BGR)

# -------------------- YOLO OCR --------------------
def _predict_reading(img: np.ndarray, conf: float) -> Tuple[str, float]:
    if MODEL is None: return "", 0.0
    res = MODEL.predict(img, conf=conf, verbose=False)[0]
    boxes = []
    for b in getattr(res,"boxes",[]):
        try:
            cls = int(b.cls[0])
            ch = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else ""
        except: ch = ""
        if ch not in "0123456789.-": continue
        x = float(b.xywh[0][0])
        c = float(b.conf[0]) if hasattr(b,"conf") else 0.0
        boxes.append((x,ch,c))
    boxes.sort(key=lambda x:x[0])
    return "".join([c for _,c,_ in boxes]), np.mean([s for _,_,s in boxes]) if boxes else 0.0

# -------------------- READERS --------------------
import os
import cv2
import numpy as np
import logging
from collections import deque
from typing import Optional, Tuple

# Assumes you already have a working _predict_reading(img, conf) function
# that returns (text, confidence). It must accept a 3-channel BGR (H,W,3) numpy array.

def read_lcd_current(frame: np.ndarray):
    """
    Simplified, robust current LCD reader for cropped video.
    """
    # Preprocess with controlled contrast and inversion
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray, alpha=1.6, beta=-20)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Try both normal and inverted
    variants = [gray, 255 - gray]

    best_txt, best_conf = "", 0.0
    for g in variants:
        txt, conf = _predict_reading(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), 0.25)
        if conf > best_conf:
            best_txt, best_conf = txt.strip(), conf

    # Normalize and interpret as before
    cleaned = "".join(c for c in best_txt.replace("..", ".") if c in "0123456789.-")
    try:
        val = float(cleaned) if cleaned not in ["", ".", "-"] else None
    except:
        val = None

    if val is None:
        return None, "-00.00", best_conf

    display = "00.00" if abs(val) < 0.005 else f"-{abs(val):05.2f}"
    return val, display, best_conf


def read_lcd_thrust(frame: np.ndarray):
    img = preprocess_frame(frame, invert=True)
    txt, conf = _predict_reading(img, 0.25)
    cleaned = "".join(c for c in (txt or "").replace("..",".") if c.isdigit() or c=="-")
    try:
        v=float(cleaned) if cleaned not in ["","-","."] else None
    except: v=None
    return v,str(int(v)) if v else "",conf

def read_lcd_rpm(frame: np.ndarray):
    img = preprocess_frame(frame)
    txt, conf = _predict_reading(img, 0.3)
    cleaned = "".join(c for c in (txt or "").replace(",","") if c.isdigit() or c=="-")
    try:
        v=float(cleaned) if cleaned not in ["","-","."] else None
    except: v=None
    return v,str(int(v)) if v else "",conf

# -------------------- SAFE VALUE --------------------
def _safe(v):
    try:
        if isinstance(v,(float,np.floating)): return v if math.isfinite(v) else 0
        if isinstance(v,(int,np.integer)): return v
        return float(v)
    except: return 0

# -------------------- PDF --------------------
def _save_pdf(session, meta, records, images):
    pdf_path=os.path.join(RESULT_DIR,f"{session}_report.pdf")
    pdf=FPDF()
    pdf.add_page()
    pdf.set_font("Arial","B",16)
    pdf.cell(0,10,"Propeller Test Report",ln=True,align="C")
    pdf.ln(10)
    pdf.set_font("Arial","B",12)
    pdf.cell(0,10,"Metadata",ln=True)
    pdf.set_font("Arial","",11)
    for k,v in meta.items():
        pdf.cell(0,8,f"{k.capitalize()}: {v}",ln=True)
    pdf.ln(5)
    pdf.set_font("Arial","B",12)
    pdf.cell(0,10,"Graphs",ln=True)
    for img in images:
        if os.path.exists(img): pdf.image(img,w=180); pdf.ln(10)
    pdf.output(pdf_path)
    return pdf_path

# -------------------- BUILD REPORT --------------------
def build_session_report(session_id: str) -> Dict:
    sess = SESSIONS.get(session_id)
    if not sess:
        return {"table_csv": None, "graphs": [], "pdf_path": None, "table_records": []}

    meta = sess["meta"]

    # Merge safely
    merged = pd.DataFrame()
    for name in ["current", "thrust", "rpm"]:
        df = sess["series"].get(name)
        if df is not None and not df.empty:
            merged = pd.merge(merged, df, on="time_s", how="outer") if not merged.empty else df.copy()
    merged = merged.sort_values("time_s").reset_index(drop=True)

    for col in ["current_a", "thrust_g", "rpm"]:
        if col not in merged.columns:
            merged[col] = 0.0

    # Optional smoothing for OCR noise
    merged["current_a"] = merged["current_a"].rolling(3, min_periods=1).median()

    voltage = meta.get("voltage", 0) or 0
    merged["power_w"] = merged["current_a"].astype(float) * float(voltage)
    merged["efficiency_gw"] = merged["thrust_g"].replace(0, np.nan) / merged["power_w"].replace(0, np.nan)
    merged = merged.replace([np.inf, -np.inf], np.nan).fillna(0)

    csv_path = os.path.join(RESULT_DIR, f"{session_id}_report.csv")
    merged.to_csv(csv_path, index=False)

    graphs, img_paths = [], []
    params = [
        ("current_a", "Current (A)"),
        ("thrust_g", "Thrust (G)"),
        ("rpm", "RPM"),
        ("power_w", "Power (W)"),
        ("efficiency_gw", "Efficiency (G/W)")
    ]

    for col, title in params:
        try:
            fig = go.Figure()

            # ✅ Custom hover for current using display values
            if col == "current_a" and "current_display" in merged.columns:
                hover_texts = [
                    f"Time: {t:.1f}s<br>Current: {disp}"
                    for t, disp in zip(merged["time_s"], merged["current_display"])
                ]
                fig.add_trace(go.Scatter(
                    x=merged["time_s"],
                    y=merged[col],
                    mode="lines+markers",
                    name=title,
                    text=hover_texts,
                    hoverinfo="text",
                    line=dict(width=2),
                    marker=dict(size=4)
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=merged["time_s"],
                    y=merged[col],
                    mode="lines+markers",
                    name=title,
                    line=dict(width=2),
                    marker=dict(size=4)
                ))

            fig.update_layout(
                title=f"{title} vs Time",
                xaxis_title="Time (s)",
                yaxis_title=title,
                template="plotly_white"
            )

            html = os.path.join(RESULT_DIR, f"{session_id}_{col}.html")
            png = os.path.join(RESULT_DIR, f"{session_id}_{col}.png")

            # Overwrite safely to prevent cross-session graph glitches
            if os.path.exists(html):
                os.remove(html)
            if os.path.exists(png):
                os.remove(png)

            fig.write_html(html, include_plotlyjs="cdn")
            try:
                fig.write_image(png, format="png", scale=2)
            except Exception as e:
                logging.warning(f"Failed PNG save {col}: {e}")

            graphs.append(html)
            img_paths.append(png)
            del fig  # clear memory
        except Exception as e:
            logging.error(f"Graph generation failed for {col}: {e}")

    # Prepare table data for CSV and PDF
    records = [{
        "Time (s)": _safe(r.get("time_s")),
        "Voltage (V)": meta.get("voltage"),
        "Prop": meta.get("prop", ""),
        "Motor": meta.get("motor", ""),
        "ESC": meta.get("esc", ""),
        "Current (A)": f"-{abs(_safe(r.get('current_a'))):05.2f}" if abs(_safe(r.get('current_a'))) >= 0.005 else "00.00",
        "Power (W)": round(_safe(r.get("power_w")), 2),
        "Thrust (G)": int(round(_safe(r.get("thrust_g")))),
        "RPM": int(round(_safe(r.get("rpm")))),
        "Efficiency (G/W)": round(_safe(r.get("efficiency_gw")), 3),
    } for r in merged.to_dict(orient="records")]

    pdf_path = _save_pdf(session_id, meta, records, img_paths)
    sess["report"] = {
        "table_csv": csv_path,
        "graphs": graphs,
        "pdf_path": pdf_path,
        "table_records": records
    }

    return sess["report"]


# -------------------- PROCESS TASK --------------------
def _init_session(sid:str,meta:Dict):
    if sid not in SESSIONS:
        SESSIONS[sid]={"meta":meta,"series":{"current":None,"thrust":None,"rpm":None},"report":None}

def process_video_task(tid,sid,vtype,vpath,meta):
    try:
        reset_progress(tid)
        _init_session(sid,meta)
        fps=int(meta.get("fps",5))
        frames=extract_frames_custom(tid,vpath,fps)
        rows=[]
        for i,f in enumerate(frames):
            img=cv2.imread(f)
            if img is None: continue
            if vtype=="current":v,_,_=read_lcd_current(img);key="current_a"
            elif vtype=="thrust":v,_,_=read_lcd_thrust(img);key="thrust_g"
            else:v,_,_=read_lcd_rpm(img);key="rpm"
            rows.append({"time_s":i/fps,key:v})
            TASK_PROGRESS[tid]["progress"]=min(85,int(((i+1)/len(frames))*85))
        df=pd.DataFrame(rows)
        SESSIONS[sid]["series"][vtype]=df
        TASK_PROGRESS[tid]["progress"]=95
        logging.info(f"Building session report for {sid}")
        report=build_session_report(sid)
        logging.info("Report built, saving DB...")
        try:
            save_test_result(sid,meta.get("prop",""),meta.get("motor",""),meta.get("esc",""),meta.get("voltage"),
                             report.get("table_csv"),report.get("graphs",[]),report.get("table_records",[]))
        except Exception as e:
            logging.error(f"DB save failed: {e}")
        TASK_PROGRESS[tid].update({"status":"done","progress":100})
        logging.info(f"Task {tid} completed successfully.")
    except Exception as e:
        logging.exception("Video processing error")
        TASK_PROGRESS[tid].update({"status":"error","message":str(e)})

def start_background_task(sid,vtype,vpath,meta)->str:
    tid=str(uuid.uuid4())
    reset_progress(tid)
    threading.Thread(target=process_video_task,args=(tid,sid,vtype,vpath,meta),daemon=True).start()
    return tid

# -------------------- GETTERS --------------------
def get_session_report(sid:str)->Optional[Dict]:
    s=SESSIONS.get(sid)
    if not s or not s.get("report"):return None
    r=s["report"]
    return {"meta":s["meta"],"table":r.get("table_records",[]),"csv_url":f"/session/{sid}/csv",
            "graphs":[f"/session/{sid}/graph/{i}" for i in range(len(r.get("graphs",[])))]}

def get_session_graph_path(sid:str,idx:int)->Optional[str]:
    s=SESSIONS.get(sid)
    if not s:return None
    g=s["report"].get("graphs",[])
    return g[idx] if 0<=idx<len(g) else None

def get_session_csv_path(sid:str)->Optional[str]:
    s=SESSIONS.get(sid)
    return s["report"].get("table_csv") if s and s.get("report") else None

def get_session_pdf_path(sid:str)->Optional[str]:
    s=SESSIONS.get(sid)
    return s["report"].get("pdf_path") if s and s.get("report") else None
