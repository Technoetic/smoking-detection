"""
흡연 감지 SaaS - 로컬 서버
FastAPI 백엔드 + 정적 HTML 프론트엔드
"""

from __future__ import annotations
import uuid, shutil, base64
from pathlib import Path
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import sys
sys.path.insert(0, str(Path(__file__).parent))
from smoking_detector import SmokingDetector, Config

app = FastAPI(title="흡연 감지 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
RESULT_DIR = Path("api_results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# 모델 로드 (서버 시작 시 1회)
print("모델 로드 중...")
cfg = Config()
cfg.OUTPUT_DIR     = str(RESULT_DIR) + "/"
cfg.SAVE_ANNOTATED = False
detector = SmokingDetector(cfg)
print("모델 로드 완료")


# COCO skeleton 연결쌍
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),
    (5,6),(5,7),(7,9),(6,8),(8,10),
    (5,11),(6,12),(11,12),
    (11,13),(13,15),(12,14),(14,16),
]
WRIST_IDX = {9, 10}
POSE_CONF_MIN = 0.25


def _put_text_bg(img, text, pos, scale=0.55, color=(255,255,255),
                 bg=(30,30,30), thickness=1, alpha=0.65):
    x, y = pos
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    ov = img.copy()
    cv2.rectangle(ov, (x-4, y-th-5), (x+tw+4, y+bl+2), bg, -1)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def _draw_skeleton(frame, kpts, confs, suspect: bool):
    kp_col = (0, 80, 255) if suspect else (0, 200, 80)
    sk_col = (0, 60, 200) if suspect else (0, 160, 60)
    wr_col = (0, 0, 255)  if suspect else (0, 255, 160)
    for a, b in SKELETON:
        if a >= len(kpts) or b >= len(kpts): continue
        ca = float(confs[a]) if confs is not None else 1.0
        cb = float(confs[b]) if confs is not None else 1.0
        if ca < POSE_CONF_MIN or cb < POSE_CONF_MIN: continue
        ax, ay = int(kpts[a][0]), int(kpts[a][1])
        bx, by = int(kpts[b][0]), int(kpts[b][1])
        if ax == 0 and ay == 0: continue
        if bx == 0 and by == 0: continue
        cv2.line(frame, (ax, ay), (bx, by), sk_col, 2, cv2.LINE_AA)
    for i, (kx, ky) in enumerate(kpts):
        conf = float(confs[i]) if confs is not None else 1.0
        if conf < POSE_CONF_MIN or (kx == 0 and ky == 0): continue
        ix, iy = int(kx), int(ky)
        if i in WRIST_IDX:
            cv2.circle(frame, (ix, iy), 8 if suspect else 6, wr_col, -1, cv2.LINE_AA)
            cv2.circle(frame, (ix, iy), 10 if suspect else 8, (255,255,255), 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (ix, iy), 4, kp_col, -1, cv2.LINE_AA)


def _draw_bbox(frame, x1, y1, x2, y2, suspect: bool, wn: float):
    color = (0, 0, 220) if suspect else (0, 200, 60)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if suspect else 2)
    label = f"SUSPECT wrist={wn:.2f}" if suspect else f"person wrist={wn:.2f}"
    bg    = (0, 0, 150) if suspect else (0, 100, 30)
    _put_text_bg(frame, label, (x1, max(y1-8, 12)),
                 color=(255,255,255), bg=bg)


def extract_snapshots(video_path: str, frame_results, max_shots: int = 4) -> list[str]:
    """의심 프레임 중 대표 스냅샷을 bbox+skeleton 오버레이 후 base64 JPEG으로 반환"""
    suspect_frames = sorted(
        [fr for fr in frame_results if fr.is_suspect],
        key=lambda fr: fr.wrist_norm
    )

    selected = []
    if suspect_frames:
        step = max(1, len(suspect_frames) // max_shots)
        selected = suspect_frames[::step][:max_shots]

    if not selected:
        return []

    cap = cv2.VideoCapture(video_path)
    snapshots = []

    for fr in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr.frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # YOLOv8-pose 추론으로 bbox + skeleton 그리기
        det = detector.pose_model(frame, verbose=False, conf=0.30, iou=0.5)[0]
        if det.keypoints is not None and len(det.keypoints.xy) > 0:
            kpts_all  = det.keypoints.xy.cpu().numpy()
            confs_all = (det.keypoints.conf.cpu().numpy()
                         if det.keypoints.conf is not None else None)
            boxes = det.boxes.xyxy.cpu().numpy()

            for pi in range(len(kpts_all)):
                x1, y1, x2, y2 = map(int, boxes[pi][:4])
                info = detector.pose_analyzer.score(
                    kpts_all[pi],
                    confs_all[pi] if confs_all is not None else None,
                )
                wn      = info["wrist_norm"]
                suspect = wn < 1.5
                _draw_bbox(frame, x1, y1, x2, y2, suspect, wn)
                _draw_skeleton(frame, kpts_all[pi], confs_all[pi] if confs_all is not None else None, suspect)

        # SUSPECT 박스 크롭 + 여백 추가해서 확대
        fh, fw = frame.shape[:2]
        crop = frame  # 기본: 전체 프레임
        if det.keypoints is not None and len(det.keypoints.xy) > 0:
            # SUSPECT 인물 중 wrist_norm 가장 낮은 박스 찾기
            best_box = None
            best_wn_val = 999
            for pi in range(len(kpts_all)):
                info = detector.pose_analyzer.score(
                    kpts_all[pi],
                    confs_all[pi] if confs_all is not None else None,
                )
                if info["wrist_norm"] < best_wn_val:
                    best_wn_val = info["wrist_norm"]
                    best_box = boxes[pi][:4]
            if best_box is not None and best_wn_val < 1.5:
                x1c, y1c, x2c, y2c = map(int, best_box)
                pad = int(max(x2c - x1c, y2c - y1c) * 0.5)
                x1c = max(0, x1c - pad)
                y1c = max(0, y1c - pad)
                x2c = min(fw, x2c + pad)
                y2c = min(fh, y2c + pad)
                crop = frame[y1c:y2c, x1c:x2c]

        # 640px 너비로 리사이즈
        th, tw = crop.shape[:2]
        target_w = 640
        scale = target_w / max(tw, 1)
        crop = cv2.resize(crop, (target_w, int(th * scale)), interpolation=cv2.INTER_LINEAR)

        # 정보 바
        ov = crop.copy()
        cv2.rectangle(ov, (0, 0), (crop.shape[1], 44), (20, 8, 8), -1)
        cv2.addWeighted(ov, 0.7, crop, 0.3, 0, crop)
        cv2.putText(crop,
                    f"t={fr.timestamp:.1f}s  wrist_norm={fr.wrist_norm:.3f}  SUSPECT",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 100, 255), 2, cv2.LINE_AA)

        _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        snapshots.append("data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8"))

    cap.release()
    return snapshots


@app.get("/")
def index():
    return FileResponse("index.html")


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="mp4 파일만 지원합니다.")

    job_id   = uuid.uuid4().hex[:8]
    tmp_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        vr = detector.process_video(str(tmp_path))

        # 의심 스냅샷 (흡연 감지 시에만)
        snapshots = []
        if vr.smoking_detected:
            snapshots = extract_snapshots(str(tmp_path), vr.frame_results)

        result = {
            "job_id":           job_id,
            "filename":         file.filename,
            "smoking_detected": bool(vr.smoking_detected),
            "frac":             vr.frac_under_thresh,
            "max_run":          vr.max_consecutive_run,
            "duration":         round(vr.duration, 1),
            "fps":              round(vr.fps, 1),
            "first_detection":  vr.first_detection_time,
            "peak_motion":      vr.peak_motion,
            "frame_count":      len(vr.frame_results),
            "snapshots":        snapshots,
            "timeline": [
                {
                    "t":          round(fr.timestamp, 2),
                    "wrist_norm": fr.wrist_norm,
                    "suspect":    fr.is_suspect,
                }
                for fr in vr.frame_results[::3]
            ],
        }
        return JSONResponse(result)

    finally:
        tmp_path.unlink(missing_ok=True)


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
