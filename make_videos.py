"""
결과 동영상 생성기 v2
======================
1. overlay  : 원본 + 사람 바운딩박스 + skeleton + 흡연 판정 텍스트
2. analysis : 좌(overlay) + 우(wrist_norm 실시간 그래프)
"""

import cv2
import numpy as np
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import torch
from ultralytics import YOLO

# ── 설정 ──────────────────────────────────────────────────────
VIDEO_DIR   = "C:/Users/Admin/Desktop/0327/흡연/"
RESULT_JSON = "C:/Users/Admin/Desktop/0327/흡연/results/results_detail.json"
OUT_DIR     = "C:/Users/Admin/Desktop/0327/흡연/results/videos/"
POSE_MODEL  = "yolov8n-pose.pt"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
WRIST_THRESH = 1.5   # 이 값 이하이면 흡연 의심

# COCO skeleton 연결쌍
SKELETON = [
    (0,1),(0,2),(1,3),(2,4),           # 얼굴
    (5,6),(5,7),(7,9),(6,8),(8,10),    # 팔
    (5,11),(6,12),(11,12),             # 몸통
    (11,13),(13,15),(12,14),(14,16),   # 다리
]
KP_NAMES = [
    "nose","l_eye","r_eye","l_ear","r_ear",
    "l_shldr","r_shldr","l_elbow","r_elbow","l_wrist","r_wrist",
    "l_hip","r_hip","l_knee","r_knee","l_ankle","r_ankle"
]
WRIST_IDX = {9, 10}   # l_wrist, r_wrist


# ── 유틸 ──────────────────────────────────────────────────────
def put_text_bg(img, text, pos, scale=0.7, color=(255,255,255),
                bg=(30,30,30), thickness=2, alpha=0.65):
    x, y = pos
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    ov = img.copy()
    cv2.rectangle(ov, (x-4, y-th-5), (x+tw+4, y+bl+2), bg, -1)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_skeleton(frame, kpts, confs, suspect: bool):
    """키포인트 + skeleton + 손목 마커 그리기"""
    h, w = frame.shape[:2]
    kp_color  = (0, 80, 255) if suspect else (0, 200, 80)
    sk_color  = (0, 60, 200) if suspect else (0, 160, 60)
    wrist_col = (0, 0, 255)  if suspect else (0, 255, 160)

    # skeleton 선
    for a, b in SKELETON:
        if a >= len(kpts) or b >= len(kpts):
            continue
        ca = float(confs[a]) if confs is not None else 1.0
        cb = float(confs[b]) if confs is not None else 1.0
        if ca < 0.25 or cb < 0.25:
            continue
        ax, ay = int(kpts[a][0]), int(kpts[a][1])
        bx, by = int(kpts[b][0]), int(kpts[b][1])
        if ax == 0 and ay == 0: continue
        if bx == 0 and by == 0: continue
        cv2.line(frame, (ax, ay), (bx, by), sk_color, 2, cv2.LINE_AA)

    # 키포인트 원
    for i, (kx, ky) in enumerate(kpts):
        conf = float(confs[i]) if confs is not None else 1.0
        if conf < 0.25 or (kx == 0 and ky == 0):
            continue
        ix, iy = int(kx), int(ky)
        if i in WRIST_IDX:
            # 손목: 크고 강조
            radius = 8 if suspect else 6
            cv2.circle(frame, (ix, iy), radius, wrist_col, -1, cv2.LINE_AA)
            cv2.circle(frame, (ix, iy), radius+2, (255,255,255), 1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (ix, iy), 4, kp_color, -1, cv2.LINE_AA)


def draw_bbox(frame, x1, y1, x2, y2, suspect: bool, wn: float):
    """사람 바운딩 박스"""
    color     = (0, 0, 220) if suspect else (0, 200, 60)
    thickness = 3 if suspect else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    label = f"SUSPECT wrist={wn:.2f}" if suspect else f"person wrist={wn:.2f}"
    bg    = (0, 0, 150) if suspect else (0, 100, 30)
    put_text_bg(frame, label, (x1, max(y1-8, 10)),
                scale=0.5, color=(255,255,255), bg=bg, thickness=1)


def draw_wrist_graph(frame_summary, current_idx, w, h, thresh_t=1.5):
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 30)
    n = len(frame_summary)
    if n == 0:
        return canvas

    pad_l, pad_r = 55, 15
    pad_t, pad_b = 30, 35
    gw = w - pad_l - pad_r
    gh = h - pad_t - pad_b
    wn_min, wn_max = -0.5, 4.0

    def to_px(idx, wn):
        x = pad_l + int(idx / max(n-1, 1) * gw)
        wc = max(wn_min, min(wn_max, wn))
        y  = pad_t + int((1 - (wc-wn_min)/(wn_max-wn_min)) * gh)
        return x, y

    # 그리드
    for tick in [0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        _, py = to_px(0, tick)
        cv2.line(canvas, (pad_l, py), (pad_l+gw, py), (50,50,65), 1)
        cv2.putText(canvas, f"{tick:.1f}", (4, py+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (110,110,140), 1)

    # 임계선
    _, ty = to_px(0, thresh_t)
    cv2.line(canvas, (pad_l, ty), (pad_l+gw, ty), (80,80,220), 1)
    cv2.putText(canvas, f"thresh {thresh_t}", (pad_l+4, ty-4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, (100,100,255), 1)

    # 의심 구간 채우기
    for i, fr in enumerate(frame_summary):
        wn = fr["wrist_norm"]
        if wn < thresh_t and wn < 900:
            x1 = pad_l + int(i/max(n-1,1)*gw)
            x2 = pad_l + int((i+1)/max(n-1,1)*gw)
            _, y0 = to_px(i, thresh_t)
            _, y1 = to_px(i, wn)
            cv2.rectangle(canvas,(x1,min(y0,y1)),(x2,max(y0,y1)),(60,40,120),-1)

    # 선
    pts = []
    for i, fr in enumerate(frame_summary):
        wn = fr["wrist_norm"] if fr["wrist_norm"] < 900 else wn_max
        pts.append(to_px(i, wn))
    for i in range(len(pts)-1):
        wn = frame_summary[i]["wrist_norm"]
        col = (80,80,220) if wn < thresh_t else (100,200,100)
        cv2.line(canvas, pts[i], pts[i+1], col, 2)

    # 현재 커서
    if 0 <= current_idx < len(pts):
        cx, cy = pts[current_idx]
        cv2.line(canvas, (cx, pad_t), (cx, pad_t+gh), (255,220,0), 2)
        cv2.circle(canvas, (cx, cy), 5, (255,220,0), -1)

    # 레이블
    cv2.putText(canvas, "wrist_norm (코~어깨 정규화)",
                (pad_l, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (170,170,200), 1)
    cv2.rectangle(canvas,(pad_l,pad_t),(pad_l+gw,pad_t+gh),(70,70,90),1)
    return canvas


# ── 동영상 생성 (공통 pose 추론) ──────────────────────────────
def process_video_pair(model, vdata: dict, src_path: str,
                       out_overlay: str, out_analysis: str):
    cap     = cv2.VideoCapture(src_path)
    fps_src = cap.get(cv2.CAP_PROP_FPS) or 3.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_src   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_src   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 출력 크기
    OV_W, OV_H = w_src, h_src          # overlay: 원본 크기
    VW, VH     = 800, 450              # analysis 좌 패널
    GW, GH     = 480, 450              # analysis 우 그래프
    AN_W, AN_H = VW+GW, VH

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    wr_ov  = cv2.VideoWriter(out_overlay,  fourcc, fps_src, (OV_W, OV_H))
    wr_an  = cv2.VideoWriter(out_analysis, fourcc, fps_src, (AN_W, AN_H))

    frame_summary = vdata["frame_summary"]
    smoking = vdata["smoking_detected"]
    gt      = vdata["gt_label"]
    frac    = vdata["frac_under_thresh"]
    run_max = vdata["max_consecutive_run"]

    sorted_ts = sorted(fr["t"] for fr in frame_summary)
    ts_to_idx = {fr["t"]: i for i, fr in enumerate(frame_summary)}

    fidx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts = fidx / fps_src
        closest_ts  = min(sorted_ts, key=lambda t: abs(t-ts)) if sorted_ts else None
        cur_idx     = ts_to_idx.get(closest_ts, 0) if closest_ts is not None else 0
        fr_data     = frame_summary[cur_idx] if cur_idx < len(frame_summary) else {}
        saved_wn    = fr_data.get("wrist_norm", 999)

        # ── YOLOv8-pose 추론 (매 프레임) ──────────────
        det = model(frame, verbose=False, conf=0.30)[0]

        has_kpts   = det.keypoints is not None and len(det.keypoints.xy) > 0
        kpts_all   = det.keypoints.xy.cpu().numpy()   if has_kpts else []
        confs_all  = (det.keypoints.conf.cpu().numpy()
                      if (has_kpts and det.keypoints.conf is not None) else None)
        boxes_all  = det.boxes.xyxy.cpu().numpy()     if has_kpts else []

        # ── overlay 프레임 그리기 ──────────────────────
        ov = frame.copy()

        # 상단 상태 바
        bar_h  = 85
        bar_bg = (30,8,8) if smoking else (8,25,8)
        ovl_bg = ov.copy()
        cv2.rectangle(ovl_bg, (0,0), (OV_W, bar_h), bar_bg, -1)
        cv2.addWeighted(ovl_bg, 0.72, ov, 0.28, 0, ov)

        status_txt = "SMOKING DETECTED" if smoking else "NORMAL"
        status_col = (60,60,255) if smoking else (60,220,60)
        cv2.putText(ov, status_txt, (18, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, status_col, 3, cv2.LINE_AA)

        info_lines = [
            f"gt={gt}",
            f"frac={frac:.2f} run={run_max}",
            f"wrist={saved_wn:.2f}" if saved_wn<900 else "wrist=N/A",
            f"t={ts:.1f}s",
        ]
        for li, line in enumerate(info_lines):
            put_text_bg(ov, line, (OV_W-255, 16+li*18),
                        scale=0.48, color=(200,200,200),
                        bg=(15,15,15), thickness=1)

        # 타임바 (하단)
        prog = fidx / max(total-1, 1)
        by   = OV_H - 10
        cv2.rectangle(ov, (0,by-5),(OV_W,by+5),(40,40,40),-1)
        bar_col = (80,80,200) if smoking else (60,160,60)
        cv2.rectangle(ov, (0,by-5),(int(OV_W*prog),by+5), bar_col, -1)

        # 사람별 bbox + skeleton
        for pi in range(len(kpts_all)):
            kpts  = kpts_all[pi]
            confs = confs_all[pi] if confs_all is not None else None
            x1,y1,x2,y2 = map(int, boxes_all[pi])

            # 이 사람의 wrist_norm 계산
            nose_y = float(kpts[0][1])
            ls_y   = float(kpts[5][1]); rs_y = float(kpts[6][1])
            lw_y   = float(kpts[9][1]); rw_y = float(kpts[10][1])
            shldr_y = (ls_y+rs_y)/2 if ls_y>0 and rs_y>0 else nose_y+10
            sd = shldr_y - nose_y
            ws = [w for w in [lw_y,rw_y] if w>0]
            if ws and sd > 2:
                wn_live = (min(ws)-nose_y)/sd
            else:
                wn_live = saved_wn

            suspect = wn_live < WRIST_THRESH

            draw_bbox(ov, x1, y1, x2, y2, suspect, wn_live)
            draw_skeleton(ov, kpts, confs, suspect)

        # 프레임 테두리 (의심 시)
        any_suspect = any(True for pi in range(len(kpts_all))
                          for _ws in [
                              [float(kpts_all[pi][9][1]),
                               float(kpts_all[pi][10][1])]]
                          if _ws)
        # 영상 판정 기준 테두리
        if smoking:
            cv2.rectangle(ov, (0,0),(OV_W-1,OV_H-1),(0,0,180),3)

        wr_ov.write(ov)

        # ── analysis 프레임 (좌=overlay 리사이즈, 우=그래프) ──
        vid_panel   = cv2.resize(ov, (VW, VH))
        graph_panel = draw_wrist_graph(frame_summary, cur_idx, GW, GH)

        put_text_bg(graph_panel,
                    f"frac={frac:.3f}  run={run_max}"
                    f"  wrist={saved_wn:.2f}" if saved_wn<900
                    else f"frac={frac:.3f}  run={run_max}",
                    (8, GH-10), scale=0.42,
                    color=(180,180,200), bg=(15,15,25), thickness=1)

        canvas = np.zeros((AN_H, AN_W, 3), dtype=np.uint8)
        canvas[:, :VW] = vid_panel
        canvas[:, VW:] = graph_panel
        cv2.line(canvas, (VW,0),(VW,AN_H),(90,90,110),2)
        wr_an.write(canvas)

        fidx += 1

    cap.release()
    wr_ov.release()
    wr_an.release()


# ── 메인 ──────────────────────────────────────────────────────
def main():
    os.makedirs(OUT_DIR+"overlay/",  exist_ok=True)
    os.makedirs(OUT_DIR+"analysis/", exist_ok=True)

    with open(RESULT_JSON, encoding="utf-8") as f:
        results = json.load(f)

    print(f"YOLOv8-pose 로드 ({DEVICE})...")
    model = YOLO(POSE_MODEL)
    model.to(DEVICE)

    total = len(results)
    print(f"총 {total}개 영상 처리 시작\n")

    for i, vdata in enumerate(results):
        fname    = vdata["filename"]
        src_path = VIDEO_DIR + fname
        stem     = Path(fname).stem

        if not os.path.exists(src_path):
            print(f"[{i+1:2d}/{total}] 원본 없음: {fname}")
            continue

        out_ov = OUT_DIR + "overlay/"  + stem + "_overlay.mp4"
        out_an = OUT_DIR + "analysis/" + stem + "_analysis.mp4"

        t0 = time.time()
        print(f"[{i+1:2d}/{total}] {fname}")

        process_video_pair(model, vdata, src_path, out_ov, out_an)

        elapsed = time.time() - t0
        pred = "흡연" if vdata["smoking_detected"] else "정상"
        print(f"         gt={vdata['gt_label']} pred={pred}  ({elapsed:.1f}s)")

    print(f"\n완료!")
    print(f"  overlay/  → {len(os.listdir(OUT_DIR+'overlay/'))}개")
    print(f"  analysis/ → {len(os.listdir(OUT_DIR+'analysis/'))}개")
    print(f"  {OUT_DIR}")


if __name__ == "__main__":
    main()
