## ⚙️ Propellor Testing System — Backend (FastAPI + YOLO + OCR)

This backend powers the Propellor Testing System, a video-based OCR pipeline built using FastAPI, YOLOv8, and OpenCV.
It extracts LCD readings (Current, Thrust, RPM) from uploaded videos, generates analytical reports (CSV + Graphs), and saves test data in a local SQLite database.

## Core Backend Technologies

- FastAPI
- Uvicorn
- OpenCV
- NumPy
- Pandas
- Matplotlib
- FFmpeg-Python
- Ultralytics (YOLOv8)
- PyYAML
- Python-Multipart


## Backend Setup

- mkdir backend
- cd backend
- python -m venv venv
- venv\Scripts\activate
- pip install -r requirements.txt
- pip install fastapi uvicorn opencv-python numpy pandas matplotlib ffmpeg-python ultralytics PyYAML python-multipart
- pip install sqlite3
- uvicorn main:app --reload

