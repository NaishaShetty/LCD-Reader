## Propellor Testing System

A modern, intelligent system built to analyze drone motor and propeller performance using video-based OCR.
Just upload your test videos, sit back, and let the system do the rest — extracting LCD readings, plotting live performance graphs, and creating detailed reports automatically.

## 🌐 Overview

The Propellor Testing System (LCD Reader) was designed to simplify the testing process for drone enthusiasts, engineers, and researchers.
By combining AI-powered OCR and a clean, modern interface, it automates what used to be a manual, time-consuming process.

With this system, you can:

- Upload videos from your test bench (Current, Thrust, RPM).
- Set your preferred frame extraction rate (1–10 FPS).
- Watch progress updates in real time.
- Automatically view calculated power, efficiency, and RPM graphs.
- Store and compare results from previous tests — all in one place.

## 💡 Features

- ✨ AI-Powered OCR — Reads LCD digits from test videos with YOLOv8.
- 📊 Automatic Reports — Generates graphs and CSVs instantly.
- ⚙️ Adjustable FPS — Fine-tune accuracy and processing speed.
- 🧠 Smart Analytics — Calculates thrust, current, RPM, power, and efficiency.
- 💾 Test History — Saves every test for easy access and comparison.
- 🎨 Modern UI — Built with React and Material-UI for a sleek experience.

## Purpose

The goal of this project is to make propeller performance testing simple, visual, and data-driven.
Instead of manually recording readings, engineers can use this tool to:

- Upload test videos
- Automatically extract data
- Instantly analyze and visualize results.

## Components

| Component    | Description                                                      |
| ------------ | ---------------------------------------------------------------- |
| **Frontend** | React + Material UI app for uploading videos and viewing reports |
| **Backend**  | FastAPI server that handles OCR, analysis, and data storage      |
| **Database** | SQLite file that saves test histories and results                |
