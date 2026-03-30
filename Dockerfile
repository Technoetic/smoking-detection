FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성 (OpenCV용)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py index.html smoking_detector.py ./

# YOLOv8 모델 자동 다운로드 (첫 실행 시)
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"

EXPOSE 8000

CMD ["python", "app.py"]
