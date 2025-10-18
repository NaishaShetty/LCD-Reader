#!/usr/bin/env python3
# ============================================================
# main.py — Full working backend with multi-video support,
#           frontend-compatible progress IDs, and graph updates.
# ============================================================

import os
import uuid
import shutil
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from video_processor import (
    UPLOAD_DIR,
    start_background_task,
    get_progress,
    get_session_report,
    get_session_graph_path,
    get_session_csv_path,
)
from database import (
    init_db,
    save_test_result,
    search_test_results,
    get_all_test_results,
    get_test_result_by_session,
)
import plotly.graph_objects as go

# ============================================================
#  FASTAPI APP INIT
# ============================================================
app = FastAPI(title="Propellor Test OCR API", version="1.3.0")

@app.on_event("startup")
async def startup_event():
    """Initialize the database on startup"""
    init_db()

# ============================================================
#  CORS CONFIG
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # You can restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
#  START ENDPOINT (Supports single + multiple uploads)
# ============================================================
@app.post("/start")
async def start(
    file: Optional[UploadFile] = File(None),          # Accept single upload
    files: Optional[List[UploadFile]] = File(None),   # Accept multiple uploads
    video_type: str = Form(...),                      # "current" | "thrust" | "rpm"
    session_id: Optional[str] = Form(None),
    prop: Optional[str] = Form(""),
    motor: Optional[str] = Form(""),
    esc: Optional[str] = Form(""),
    voltage: Optional[str] = Form(""),
    fps: Optional[int] = Form(5),
):
    """
    Upload one or more videos for OCR processing.
    Works with both single ('file') and multiple ('files') uploads.
    """

    # Validate video type
    if video_type not in {"current", "thrust", "rpm"}:
        return JSONResponse({"error": "Invalid video_type"}, status_code=400)

    # Auto-generate session if not provided
    if not session_id:
        session_id = str(uuid.uuid4())

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    # Normalize upload list
    upload_list = []
    if files:
        upload_list.extend(files)
    elif file:
        upload_list.append(file)

    if not upload_list:
        return JSONResponse({"error": "No video files received"}, status_code=400)

    task_ids = []
    for uf in upload_list:
        save_path = os.path.join(UPLOAD_DIR, f"{session_id}_{video_type}_{uf.filename}")
        with open(save_path, "wb") as f:
            shutil.copyfileobj(uf.file, f)
        meta = {"prop": prop, "motor": motor, "esc": esc, "voltage": voltage, "fps": fps}
        task_id = start_background_task(session_id, video_type, save_path, meta)
        task_ids.append(task_id)

    # ✅ Return both singular and plural for frontend compatibility
    return {
        "task_id": task_ids[0] if task_ids else None,
        "task_ids": task_ids,
        "session_id": session_id
    }

# ============================================================
#  PROGRESS ENDPOINT
# ============================================================
@app.get("/progress/{task_id}")
async def progress(task_id: str):
    """Return task progress by task_id"""
    return get_progress(task_id)

# ============================================================
#  SESSION RESULT ENDPOINT
# ============================================================
@app.get("/session/{session_id}/result")
async def session_result(session_id: str):
    """Fetch computed report for a session"""
    rep = get_session_report(session_id)
    if not rep:
        return JSONResponse({"error": "Report not ready"}, status_code=404)
    return rep

# ============================================================
#  INTERACTIVE GRAPH ENDPOINT (Plotly)
# ============================================================
@app.get("/session/{session_id}/interactive")
async def session_interactive(session_id: str):
    """Generate interactive Plotly graph for a session."""
    rep = get_session_report(session_id)
    if not rep or "table" not in rep:
        return JSONResponse({"error": "Report not ready"}, status_code=404)

    table = rep["table"]
    times = [row.get("Time (s)", 0) for row in table]
    thrust = [row.get("Thrust (G)", 0) for row in table]
    current = [row.get("Current (A)", 0) for row in table]
    rpm = [row.get("RPM", 0) for row in table]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=thrust, mode="lines+markers", name="Thrust (G)",
                             hovertemplate="Time %{x}s<br>Thrust: %{y} g"))
    fig.add_trace(go.Scatter(x=times, y=current, mode="lines+markers", name="Current (A)",
                             hovertemplate="Time %{x}s<br>Current: %{y} A"))
    fig.add_trace(go.Scatter(x=times, y=rpm, mode="lines+markers", name="RPM",
                             hovertemplate="Time %{x}s<br>RPM: %{y}"))

    fig.update_layout(
        title=f"Session {session_id} - Interactive Graph",
        xaxis_title="Time (s)",
        yaxis_title="Value",
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    html_path = os.path.join(UPLOAD_DIR, f"{session_id}_interactive.html")
    fig.write_html(html_path, include_plotlyjs="cdn")
    return FileResponse(html_path, media_type="text/html")

# ============================================================
#  STATIC GRAPH ENDPOINT
# ============================================================
@app.get("/session/{session_id}/graph/{index}")
async def session_graph(session_id: str, index: int):
    """Serve saved static graph images"""
    p = get_session_graph_path(session_id, index)
    if not p or not os.path.exists(p):
        return JSONResponse({"error": "Graph not found"}, status_code=404)
    return FileResponse(p)

# ============================================================
#  CSV DOWNLOAD ENDPOINT
# ============================================================
@app.get("/session/{session_id}/csv")
async def session_csv(session_id: str):
    """Serve the processed CSV for a given session"""
    p = get_session_csv_path(session_id)
    if not p or not os.path.exists(p):
        return JSONResponse({"error": "CSV not found"}, status_code=404)
    return FileResponse(p, filename=os.path.basename(p), media_type="text/csv")

# ============================================================
#  SAVE SESSION RESULT
# ============================================================
@app.post("/session/{session_id}/save")
async def save_session(session_id: str):
    """Save a completed session to the database."""
    rep = get_session_report(session_id)
    if not rep:
        return JSONResponse({"error": "Report not ready"}, status_code=404)

    meta = rep.get("meta", {})
    csv_path = rep.get("csv_url")
    graph_paths = rep.get("graphs", [])
    table_data = rep.get("table", [])

    success = save_test_result(
        session_id=session_id,
        prop_name=meta.get("prop", ""),
        motor_name=meta.get("motor", ""),
        esc_name=meta.get("esc", ""),
        voltage=meta.get("voltage"),
        csv_path=csv_path,
        graph_paths=graph_paths,
        table_data=table_data
    )

    if success:
        return {"message": "Test result saved successfully"}
    else:
        return JSONResponse({"error": "Failed to save test result"}, status_code=500)

# ============================================================
#  HISTORY: GET ALL RESULTS
# ============================================================
@app.get("/history")
async def get_history():
    """Fetch all test results"""
    results = get_all_test_results()
    return {"results": results}

# ============================================================
#  HISTORY: SEARCH RESULTS
# ============================================================
@app.get("/history/search")
async def search_history(
    prop: Optional[str] = None,
    motor: Optional[str] = None,
    esc: Optional[str] = None
):
    """Search test history by prop, motor, or ESC"""
    results = search_test_results(prop_name=prop, motor_name=motor, esc_name=esc)
    return {"results": results}

# ============================================================
#  HISTORY: GET SPECIFIC SESSION
# ============================================================
@app.get("/history/{session_id}")
async def get_history_result(session_id: str):
    """Get a specific saved test result"""
    result = get_test_result_by_session(session_id)
    if not result:
        return JSONResponse({"error": "Test result not found"}, status_code=404)
    return result

# ============================================================
#  ROOT ENDPOINT
# ============================================================
@app.get("/")
async def root():
    """Health check"""
    return {"message": "Propellor Test OCR API running ✅ (multi-upload + progress fixed)"}
