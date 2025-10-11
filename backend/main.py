import os
import uuid
import shutil
from typing import Optional
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
from database import init_db, save_test_result, search_test_results, get_all_test_results, get_test_result_by_session

app = FastAPI(title="Propellor Test OCR API", version="1.0.0")

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    init_db()

# CORS (allow your React app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
#  START ENDPOINT (ADDED FPS SUPPORT)
# ============================================================
@app.post("/start")
async def start(
    file: UploadFile = File(...),
    video_type: str = Form(...),          # "current" | "thrust" | "rpm"
    session_id: Optional[str] = Form(None),
    prop: Optional[str] = Form(""),
    motor: Optional[str] = Form(""),
    esc: Optional[str] = Form(""),
    voltage: Optional[str] = Form(""),
    fps: Optional[int] = Form(5),         # ✅ NEW PARAMETER (default 5)
):
    """
    Upload a video for OCR processing.
    Now accepts FPS parameter from the frontend (1–10).
    """
    if video_type not in {"current", "thrust", "rpm"}:
        return JSONResponse({"error": "Invalid video_type"}, status_code=400)

    if not session_id:
        session_id = str(uuid.uuid4())

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    save_path = os.path.join(UPLOAD_DIR, f"{session_id}_{video_type}_{file.filename}")
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    meta = {"prop": prop, "motor": motor, "esc": esc, "voltage": voltage, "fps": fps}

    task_id = start_background_task(session_id, video_type, save_path, meta)
    return {"task_id": task_id, "session_id": session_id}

# ============================================================
#  PROGRESS ENDPOINT
# ============================================================
@app.get("/progress/{task_id}")
async def progress(task_id: str):
    return get_progress(task_id)

# ============================================================
#  SESSION RESULT ENDPOINT
# ============================================================
@app.get("/session/{session_id}/result")
async def session_result(session_id: str):
    rep = get_session_report(session_id)
    if not rep:
        return JSONResponse({"error": "Report not ready"}, status_code=404)
    return rep

# ============================================================
#  SESSION GRAPH ENDPOINT
# ============================================================
@app.get("/session/{session_id}/graph/{index}")
async def session_graph(session_id: str, index: int):
    p = get_session_graph_path(session_id, index)
    if not p or not os.path.exists(p):
        return JSONResponse({"error": "Graph not found"}, status_code=404)
    return FileResponse(p)

# ============================================================
#  SESSION CSV ENDPOINT
# ============================================================
@app.get("/session/{session_id}/csv")
async def session_csv(session_id: str):
    p = get_session_csv_path(session_id)
    if not p or not os.path.exists(p):
        return JSONResponse({"error": "CSV not found"}, status_code=404)
    return FileResponse(p, filename=os.path.basename(p), media_type="text/csv")

# ============================================================
#  SAVE SESSION RESULT
# ============================================================
@app.post("/session/{session_id}/save")
async def save_session(session_id: str):
    """Save the current session's results to the database."""
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
    """Get all test results from history."""
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
    """Search test history by prop, motor, or ESC name."""
    results = search_test_results(prop_name=prop, motor_name=motor, esc_name=esc)
    return {"results": results}

# ============================================================
#  HISTORY: GET SPECIFIC SESSION
# ============================================================
@app.get("/history/{session_id}")
async def get_history_result(session_id: str):
    """Get a specific test result from history."""
    result = get_test_result_by_session(session_id)
    if not result:
        return JSONResponse({"error": "Test result not found"}, status_code=404)
    return result
