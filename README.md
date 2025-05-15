# 🔥 Forest Fire Detection using Deep Learning

A machine learning-based forest fire detection system using **TensorFlow**, **FastAPI**, and image classification to detect the presence of fire in forest images.

---

## 📌 Overview

This project helps detect forest fires from images using a pre-trained deep learning model served via a FastAPI backend.

**Includes:**
- `Forest_Fire_Detection.ipynb` – Model training and exploration
- `main.py` – API endpoint for prediction
- `util.py` – Image preprocessing utilities
- `render.yaml` – For Render deployment
- `model/fire_model.keras` – Auto-downloaded at runtime

---

## ⚙️ Installation & Setup
  # 1. Clone the repo
  - git clone https://github.com/Adhiksha007/forest-fire-api.git
  - cd forest-fire-api
  
  # 2. Create a virtual environment
  python -m venv venv
  # macOS/Linux
  source venv/bin/activate
  # Windows
  venv\Scripts\activate
  
  # 3. Install dependencies
  pip install -r requirements.txt
  
  # 4. Run the FastAPI server
  uvicorn main:app --reload

---


## 👤 Author
- Name: Adhiksha Reddy Uppalapati
- Email: 📧 uppalapatiadhikshareddy@gmail.com

---

Copyright (c) 2025 Adhiksha Reddy

---
