## ⚙️ Backend – Propellor Testing System 

A high-performance FastAPI backend for the Propellor Testing System, designed to process video inputs, extract LCD readings using YOLOv8 OCR, and generate analytical performance reports.
It powers the React frontend by handling video uploads, background frame processing, and database management for test sessions.

## 🧠 Overview

The backend provides APIs and background processing logic to:

- Accept video uploads (Current, Thrust, and RPM tests).
- Extract frames at user-selected FPS (1–10 FPS).
- Use YOLOv8 to recognize LCD digits in frames.
- Generate time-series CSVs and analytical graphs.
- Calculate performance metrics such as Power and Efficiency.
- Store and retrieve test sessions from a local SQLite database.
- Expose all results and history via REST API endpoints.

  ## 🧩 Tech Stack

  | Technology                     | Purpose                                    |
| ------------------------------ | ------------------------------------------ |
| **FastAPI**                    | High-speed backend framework for REST APIs |
| **Uvicorn**                    | ASGI server to host the FastAPI app        |
| **OpenCV**                     | Frame extraction and image preprocessing   |
| **FFmpeg-Python**              | Frame extraction at custom FPS rates       |
| **YOLOv8 (Ultralytics)**       | OCR model for LCD digit recognition        |
| **NumPy & Pandas**             | Data manipulation and filtering            |
| **Matplotlib**                 | Plot generation for test graphs            |
| **SQLite (via `database.py`)** | Local database to store test results       |
| **PyYAML**                     | Model configuration parsing                |
| **Python-Multipart**           | File upload handling in FastAPI            |



## Core Framework

- FastAPI
- Uvicorn

## Computer Vision & Processing

- opencv-python
- ffmpeg-python
- ultralytics
- numpy
- pandas
- matplotlib

## Configuration & Upload Handling

- PyYAML
- python-multipart


## Backend Setup

- mkdir backend
- cd backend
- python -m venv venv
- venv\Scripts\activate
- pip install -r requirements.txt
- pip install fastapi uvicorn opencv-python numpy pandas matplotlib ffmpeg-python ultralytics PyYAML python-multipart
- pip install sqlite3
- uvicorn main:app --reload

