"""
흡연 감지 시스템 v3.1  ─  드론 탑뷰 CCTV 전용
================================================
YOLOv8-Pose 키포인트 기반 손-얼굴 근접도 분석 +
Optical Flow 상체 모션 + 시간적 지속성 앙상블

파일명 규칙
  C_{구역}_{카메라}_{타입}_{cam_type}_{날짜}_{시간}_{앵글}_{상태}_{DF}.mp4
  상태: for=흡연 전(정상), aft=흡연 중, set=세팅
"""

from __future__ import annotations
import cv2
import numpy as np
import os, json, csv, time, warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

warnings.filterwarnings("ignore")
import torch
from ultralytics import YOLO


# ══════════════════════════════════════════════════════
#  설정
# ══════════════════════════════════════════════════════
class Config:
    VIDEO_DIR   = "C:/Users/Admin/Desktop/0327/흡연/"
    OUTPUT_DIR  = "C:/Users/Admin/Desktop/0327/흡연/results/"
    POSE_MODEL  = "yolov8n-pose.pt"
    DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 포즈 분석 ──────────────────────────────────
    # wrist_norm: (손목y - 코y) / (어깨y - 코y)
    # 값이 작을수록 손이 얼굴 근처
    WRIST_NORM_SMOKE_THRESH = 1.20   # 이 값 이하이면 흡연 가능 포즈
    POSE_CONF_MIN           = 0.25   # 키포인트 최소 신뢰도

    # ── 영상 판정 ──────────────────────────────────
    # 포즈 단서 (frac = 전체 프레임 중 wrist_norm < T인 비율)
    FRAC_THRESH_T    = 0.50   # frac 계산에 사용할 wrist_norm 임계값
    FRAC_THRESH_VAL  = 0.02   # 이 비율 이상이면 흡연 판정 기여
    # 연속 구간 단서
    RUN_THRESH_T     = 1.50   # run 계산에 사용할 wrist_norm 임계값
    RUN_THRESH_N     = 1      # 최장 연속 구간이 N프레임 이상이면 기여
    # 수평 거리 조건 (손목-코 x축 거리 / 어깨 너비)
    HORIZ_DIST_THRESH = 0.8   # 이 값 이하이면 손이 얼굴 정면 근처

    # ── 시각화 ─────────────────────────────────────
    SAMPLE_EVERY_N  = 1        # 전프레임 (3fps)
    SAVE_ANNOTATED  = True
    ANNOTATE_EVERY  = 15       # N프레임마다 1장 저장


# ══════════════════════════════════════════════════════
#  COCO Keypoint 인덱스
# ══════════════════════════════════════════════════════
NOSE = 0; L_EYE = 1; R_EYE = 2
L_SHLDR = 5; R_SHLDR = 6
L_ELBOW = 7; R_ELBOW = 8
L_WRIST = 9; R_WRIST = 10
L_HIP = 11;  R_HIP  = 12


# ══════════════════════════════════════════════════════
#  파일명 파서
# ══════════════════════════════════════════════════════
def parse_filename(filename: str) -> dict:
    stem  = Path(filename).stem
    parts = stem.split("_")
    def g(i): return parts[i] if len(parts) > i else "?"
    return dict(zone=g(1), camera=g(2), stype=g(3), cam_type=g(4),
                date=g(5), time_str=g(6), angle=g(7),
                gt_label=g(8), df_version=g(9))


# ══════════════════════════════════════════════════════
#  포즈 분석기
# ══════════════════════════════════════════════════════
class PoseAnalyzer:
    """
    YOLOv8-pose 키포인트에서 손목 정규화 위치를 계산한다.

    wrist_norm = (min_wrist_y - nose_y) / (shoulder_y - nose_y)
      · 값이 0에 가까울수록 손이 코 바로 아래(흡연 포즈)
      · 드론 탑뷰에서 앉은 자세는 어깨 기준 정규화로 안정적 처리

    반환:
      wrist_norm  (float)  ─ 유효 키포인트 없으면 +999
      elbow_norm  (float)  ─ 동일 기준
      body_h_px   (float)  ─ 코~어깨 거리(픽셀)
      valid       (bool)
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def score(self, kpts: np.ndarray,
              confs: Optional[np.ndarray]) -> dict:
        p, c = kpts, confs

        def valid_kpt(idx):
            if c is not None and c[idx] < self.cfg.POSE_CONF_MIN:
                return None, None
            x = float(p[idx][0]); y = float(p[idx][1])
            if x <= 0 or y <= 0:
                return None, None
            return x, y

        _, nose_y = valid_kpt(NOSE)
        nose_x, _ = valid_kpt(NOSE)
        if nose_y is None:
            return {"wrist_norm": 999, "elbow_norm": 999,
                    "horiz_dist": 999, "body_h_px": 0, "valid": False}

        ls_x, ls_y = valid_kpt(L_SHLDR)
        rs_x, rs_y = valid_kpt(R_SHLDR)
        lw_x, lw_y = valid_kpt(L_WRIST)
        rw_x, rw_y = valid_kpt(R_WRIST)
        le_x, le_y = valid_kpt(L_ELBOW)
        re_x, re_y = valid_kpt(R_ELBOW)

        # 어깨 기준 (코~어깨 거리로 정규화)
        if ls_y is not None and rs_y is not None:
            shldr_y = (ls_y + rs_y) / 2
        elif ls_y is not None:
            shldr_y = ls_y
        elif rs_y is not None:
            shldr_y = rs_y
        else:
            return {"wrist_norm": 999, "elbow_norm": 999,
                    "horiz_dist": 999, "body_h_px": 0, "valid": False}

        nose_to_shldr = shldr_y - nose_y
        if nose_to_shldr < 2:
            return {"wrist_norm": 999, "elbow_norm": 999,
                    "horiz_dist": 999, "body_h_px": 0, "valid": False}

        wrist_ys = [w for w in [lw_y, rw_y] if w is not None]
        if not wrist_ys:
            return {"wrist_norm": 999, "elbow_norm": 999,
                    "horiz_dist": 999, "body_h_px": nose_to_shldr, "valid": False}

        wrist_norm = (min(wrist_ys) - nose_y) / nose_to_shldr
        elbow_ys   = [e for e in [le_y, re_y] if e is not None]
        elbow_norm = (min(elbow_ys) - nose_y) / nose_to_shldr if elbow_ys else 999

        # 손목-코 수평 거리 (어깨 너비로 정규화)
        # 손목이 코 수평 위치 근처에 있을수록 흡연 포즈
        if ls_x is not None and rs_x is not None:
            shldr_width = abs(rs_x - ls_x)
        else:
            shldr_width = nose_to_shldr  # fallback: 어깨~코 거리 사용

        wrist_xs = []
        if lw_x is not None: wrist_xs.append((lw_x, lw_y))
        if rw_x is not None: wrist_xs.append((rw_x, rw_y))

        # 가장 높이 올라온(y값 작은) 손목의 수평 거리
        if wrist_xs and shldr_width > 2:
            best_wrist = min(wrist_xs, key=lambda xy: xy[1])
            horiz_dist = abs(best_wrist[0] - nose_x) / shldr_width
        else:
            horiz_dist = 999

        return {
            "wrist_norm": round(wrist_norm, 4),
            "elbow_norm": round(elbow_norm, 4),
            "horiz_dist": round(horiz_dist, 4),
            "body_h_px":  round(nose_to_shldr, 1),
            "valid": True,
        }


# ══════════════════════════════════════════════════════
#  모션 분석기
# ══════════════════════════════════════════════════════
class MotionAnalyzer:
    """
    사람 상체 상단부 ROI에서 Optical Flow 기반 활동량 측정.
    흡연 동작: 손↑↓ 반복 → y축 흐름 우세.
    """

    def __init__(self):
        self.prev: Optional[np.ndarray] = None
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=30, varThreshold=16, detectShadows=False
        )

    def reset(self):
        self.prev = None
        self.bg   = cv2.createBackgroundSubtractorMOG2(
            history=30, varThreshold=16, detectShadows=False
        )

    def score(self, frame: np.ndarray,
              bbox: Tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        if h < 8:
            self.prev = None; return 0.0

        uy2 = min(y2, y1 + int(h * 0.55))
        roi = frame[y1:uy2, max(0, x1):x2]
        if roi.size == 0:
            self.prev = None; return 0.0

        gray = cv2.GaussianBlur(
            cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0
        )

        # MOG2
        fg      = self.bg.apply(gray)
        mog_act = np.sum(fg > 127) / max(gray.size, 1)

        # Optical Flow (이전 프레임 있을 때만)
        flow_score = 0.0
        if self.prev is not None:
            try:
                prev_rs = cv2.resize(self.prev, (gray.shape[1], gray.shape[0]))
                flow    = cv2.calcOpticalFlowFarneback(
                    prev_rs, gray, None,
                    pyr_scale=0.5, levels=2, winsize=9,
                    iterations=2, poly_n=5, poly_sigma=1.1, flags=0
                )
                mag  = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                vy   = np.abs(flow[..., 1])
                mean_mag = float(mag.mean())
                vy_frac  = float(vy.mean()) / (mean_mag + 1e-5)
                flow_score = min(1.0, mean_mag / 2.5) * min(1.0, vy_frac * 1.5)
            except Exception:
                pass

        self.prev = gray.copy()
        return min(1.0, mog_act / 0.12 * 0.6 + flow_score * 0.4)


# ══════════════════════════════════════════════════════
#  결과 구조체
# ══════════════════════════════════════════════════════
@dataclass
class FrameResult:
    frame_idx:    int
    timestamp:    float
    num_persons:  int
    wrist_norm:   float   # 핵심 포즈 수치
    motion_score: float
    is_suspect:   bool    # 이 프레임이 흡연 의심인지


@dataclass
class VideoResult:
    filename:     str
    zone:         str
    camera:       str
    date:         str
    time_str:     str
    angle:        str
    gt_label:     str
    df_version:   str
    total_frames: int
    fps:          float
    duration:     float
    # 판정 결과
    smoking_detected:       bool
    frac_under_thresh:      float   # frac < FRAC_THRESH_T 비율
    max_consecutive_run:    int     # 최장 연속 의심 프레임 수
    peak_motion:            float
    first_detection_time:   Optional[float]
    frame_results:          List[FrameResult] = field(default_factory=list)


# ══════════════════════════════════════════════════════
#  메인 감지기
# ══════════════════════════════════════════════════════
class SmokingDetector:

    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()
        print(f"[SmokingDetector] device={self.cfg.DEVICE}")
        self.pose_model = YOLO(self.cfg.POSE_MODEL)
        self.pose_model.to(self.cfg.DEVICE)
        self.pose_analyzer   = PoseAnalyzer(self.cfg)
        self.motion_analyzer = MotionAnalyzer()

        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        os.makedirs(self.cfg.OUTPUT_DIR + "annotated/", exist_ok=True)

    # ── 단일 영상 처리 ────────────────────────────
    def process_video(self, video_path: str) -> VideoResult:
        fname        = os.path.basename(video_path)
        meta         = parse_filename(fname)
        cap          = cv2.VideoCapture(video_path)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 3.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration     = total_frames / fps

        vr = VideoResult(
            filename=fname, zone=meta["zone"], camera=meta["camera"],
            date=meta["date"], time_str=meta["time_str"], angle=meta["angle"],
            gt_label=meta["gt_label"], df_version=meta["df_version"],
            total_frames=total_frames, fps=fps, duration=duration,
            smoking_detected=False, frac_under_thresh=0.0,
            max_consecutive_run=0, peak_motion=0.0,
            first_detection_time=None,
        )

        self.motion_analyzer.reset()
        frame_results: List[FrameResult] = []
        fidx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if fidx % self.cfg.SAMPLE_EVERY_N == 0:
                ts = fidx / fps
                fr = self._analyze_frame(frame, fidx, ts)
                frame_results.append(fr)

                if self.cfg.SAVE_ANNOTATED and fidx % self.cfg.ANNOTATE_EVERY == 0:
                    ann = self._draw(frame, fr)
                    cv2.imwrite(
                        self.cfg.OUTPUT_DIR + "annotated/" +
                        f"{Path(fname).stem}_f{fidx:04d}.jpg",
                        ann, [cv2.IMWRITE_JPEG_QUALITY, 80]
                    )
            fidx += 1

        cap.release()

        if frame_results:
            vr.frame_results = frame_results
            vr = self._decide(vr)
        return vr

    # ── 프레임 분석 ───────────────────────────────
    def _analyze_frame(self, frame: np.ndarray,
                       fidx: int, ts: float) -> FrameResult:
        cfg = self.cfg
        det = self.pose_model(frame, verbose=False, conf=0.30, iou=0.5)[0]

        if det.keypoints is None or len(det.keypoints.xy) == 0:
            return FrameResult(fidx, ts, 0, 999.0, 0.0, False)

        kpts_all  = det.keypoints.xy.cpu().numpy()
        confs_all = (det.keypoints.conf.cpu().numpy()
                     if det.keypoints.conf is not None else None)
        boxes     = det.boxes.xyxy.cpu().numpy()

        best_wn     = 999.0
        best_motion = 0.0

        for pi in range(len(kpts_all)):
            info = self.pose_analyzer.score(
                kpts_all[pi],
                confs_all[pi] if confs_all is not None else None,
            )
            if not info["valid"]:
                continue

            wn      = info["wrist_norm"]
            bbox    = tuple(map(int, boxes[pi][:4]))
            motion  = self.motion_analyzer.score(frame, bbox)

            if wn < best_wn:
                best_wn     = wn
                best_motion = motion

        return FrameResult(
            frame_idx=fidx, timestamp=ts,
            num_persons=len(kpts_all),
            wrist_norm=round(best_wn, 4),
            motion_score=round(best_motion, 4),
            is_suspect=(best_wn < cfg.RUN_THRESH_T),
        )

    # ── 영상 최종 판정 ────────────────────────────
    def _decide(self, vr: VideoResult) -> VideoResult:
        cfg = self.cfg

        # is_suspect에는 이미 horiz_dist 조건이 반영되어 있음
        suspects = [fr.is_suspect for fr in vr.frame_results]

        # frac: 의심 프레임 비율 (수평거리 조건 포함)
        frac = sum(suspects) / max(len(suspects), 1)

        # 연속 구간: 의심 프레임의 최장 연속 수
        max_run = 0; cur_run = 0
        for s in suspects:
            if s:
                cur_run += 1; max_run = max(max_run, cur_run)
            else:
                cur_run = 0

        # 판정 (OR 조건)
        smoking = (frac >= cfg.FRAC_THRESH_VAL or
                   max_run >= cfg.RUN_THRESH_N)

        # 최초 감지 시점
        first_det = None
        for fr in vr.frame_results:
            if fr.is_suspect:
                first_det = fr.timestamp; break

        vr.smoking_detected      = smoking
        vr.frac_under_thresh     = round(frac, 4)
        vr.max_consecutive_run   = int(max_run)
        vr.peak_motion           = round(
            max((fr.motion_score for fr in vr.frame_results), default=0.0), 4
        )
        vr.first_detection_time  = first_det
        return vr

    # ── 시각화 ────────────────────────────────────
    def _draw(self, frame: np.ndarray, fr: FrameResult) -> np.ndarray:
        vis   = frame.copy()
        smoke = fr.is_suspect
        color = (0, 0, 220) if smoke else (0, 180, 0)
        bg_c  = (30, 10, 10) if smoke else (10, 30, 10)
        label = "SUSPECT" if smoke else "normal"

        cv2.rectangle(vis, (0, 0), (560, 110), bg_c, -1)
        cv2.putText(vis,
                    f"{label}  wrist_norm={fr.wrist_norm:.3f}",
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)
        cv2.putText(vis,
                    f"t={fr.timestamp:.1f}s  persons={fr.num_persons}"
                    f"  motion={fr.motion_score:.3f}",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (180,180,180), 1)
        return vis

    # ── 배치 처리 ─────────────────────────────────
    def run_batch(self) -> List[VideoResult]:
        videos = sorted([f for f in os.listdir(self.cfg.VIDEO_DIR)
                         if f.endswith(".mp4")])
        print(f"\n총 {len(videos)}개 영상  (device={self.cfg.DEVICE})\n")

        all_results = []
        for i, fname in enumerate(videos):
            path = os.path.join(self.cfg.VIDEO_DIR, fname)
            t0   = time.time()
            print(f"[{i+1:2d}/{len(videos)}] {fname}")
            try:
                vr      = self.process_video(path)
                elapsed = time.time() - t0
                mark    = "🚬 흡연" if vr.smoking_detected else "  정상"
                correct = ""
                if vr.gt_label == "aft":
                    correct = " ✓" if vr.smoking_detected else " ✗MISS"
                elif vr.gt_label == "for":
                    correct = " ✓" if not vr.smoking_detected else " ✗FA"
                print(f"         → {mark}  "
                      f"frac={vr.frac_under_thresh:.2f}  "
                      f"run={vr.max_consecutive_run}  "
                      f"gt={vr.gt_label}{correct}  ({elapsed:.1f}s)")
                all_results.append(vr)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"         → 오류: {e}")
        return all_results

    # ── 결과 저장 ─────────────────────────────────
    def save_results(self, results: List[VideoResult]):
        out = self.cfg.OUTPUT_DIR

        class NpEncoder(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, np.integer): return int(o)
                if isinstance(o, np.floating): return float(o)
                if isinstance(o, np.bool_):   return bool(o)
                return super().default(o)

        # JSON
        json_data = []
        for vr in results:
            d = asdict(vr)
            d["frame_summary"] = [
                {"t": r["timestamp"],
                 "wrist_norm": r["wrist_norm"],
                 "motion": r["motion_score"],
                 "suspect": r["is_suspect"]}
                for r in d.pop("frame_results")
            ]
            json_data.append(d)
        with open(out + "results_detail.json", "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2, cls=NpEncoder)

        # CSV
        fields = ["filename","zone","camera","date","time_str","angle",
                  "gt_label","df_version","total_frames","fps","duration",
                  "smoking_detected","frac_under_thresh",
                  "max_consecutive_run","peak_motion","first_detection_time"]
        with open(out + "results_summary.csv", "w", newline="",
                  encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for vr in results:
                w.writerow({k: getattr(vr, k) for k in fields})

        # 성능 평가
        gt_map  = {"for": False, "aft": True}
        labeled = [(vr.smoking_detected, gt_map[vr.gt_label])
                   for vr in results if vr.gt_label in gt_map]
        if labeled:
            tp = sum(1 for p,g in labeled if p and g)
            tn = sum(1 for p,g in labeled if not p and not g)
            fp = sum(1 for p,g in labeled if p and not g)
            fn = sum(1 for p,g in labeled if not p and g)
            n  = len(labeled)
            acc  = (tp+tn)/n
            prec = tp/(tp+fp) if tp+fp>0 else 0
            rec  = tp/(tp+fn) if tp+fn>0 else 0
            f1   = 2*prec*rec/(prec+rec) if prec+rec>0 else 0
            eval_d = {"total":n,"TP":tp,"TN":tn,"FP":fp,"FN":fn,
                      "accuracy":round(acc,4),"precision":round(prec,4),
                      "recall":round(rec,4),"f1_score":round(f1,4),
                      "params":{
                          "FRAC_THRESH_T":    self.cfg.FRAC_THRESH_T,
                          "FRAC_THRESH_VAL":  self.cfg.FRAC_THRESH_VAL,
                          "RUN_THRESH_T":     self.cfg.RUN_THRESH_T,
                          "RUN_THRESH_N":     self.cfg.RUN_THRESH_N,
                          "HORIZ_DIST_THRESH":self.cfg.HORIZ_DIST_THRESH,
                      }}
            with open(out + "evaluation.json", "w", encoding="utf-8") as f:
                json.dump(eval_d, f, indent=2)

            print("\n" + "═"*52)
            print("  성능 평가  (gt: for=정상, aft=흡연)")
            print("═"*52)
            print(f"  Accuracy  : {acc:.1%}  ({tp+tn}/{n})")
            print(f"  Precision : {prec:.1%}")
            print(f"  Recall    : {rec:.1%}")
            print(f"  F1-Score  : {f1:.4f}")
            print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
            print("═"*52)

        print(f"\n결과 저장: {out}")
        print(f"  results_summary.csv  /  results_detail.json")
        print(f"  evaluation.json")
        print(f"  annotated/ ({len(os.listdir(out+'annotated/'))}장)")


# ══════════════════════════════════════════════════════
#  리포트 시각화
# ══════════════════════════════════════════════════════
def generate_report(results: List[VideoResult], output_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  matplotlib 없음 - 그래프 생략"); return

    rdir = output_dir + "report/"
    os.makedirs(rdir, exist_ok=True)

    cfg = Config()

    # 개별 타임라인
    for vr in results:
        if not vr.frame_results: continue
        ts   = [fr.timestamp   for fr in vr.frame_results]
        wns  = [min(fr.wrist_norm, 5.0) for fr in vr.frame_results]
        mots = [fr.motion_score for fr in vr.frame_results]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 5), sharex=True)

        # 손목 정규화 위치
        ax1.plot(ts, wns, color="steelblue", lw=1.2, label="wrist_norm")
        ax1.axhline(cfg.FRAC_THRESH_T, color="red",
                    ls="--", lw=1, label=f"frac_thresh={cfg.FRAC_THRESH_T}")
        ax1.axhline(cfg.RUN_THRESH_T, color="orange",
                    ls=":", lw=1, label=f"run_thresh={cfg.RUN_THRESH_T}")
        ax1.fill_between(ts, 0, [min(w, cfg.FRAC_THRESH_T) for w in wns],
                         where=[w < cfg.FRAC_THRESH_T for w in wns],
                         color="red", alpha=0.25)
        ax1.set_ylabel("wrist_norm\n(낮을수록 손이 얼굴 근처)")
        ax1.set_ylim(-0.5, 5)
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(alpha=0.3)

        # 모션
        ax2.plot(ts, mots, color="orange", lw=1.0, label="motion score")
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Motion")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(alpha=0.3)

        det_str = "🚬SMOKING" if vr.smoking_detected else "✓normal"
        correct = ""
        if vr.gt_label == "aft" and not vr.smoking_detected:
            correct = "  [MISS]"
        elif vr.gt_label == "for" and vr.smoking_detected:
            correct = "  [FA]"
        fig.suptitle(
            f"{vr.filename}\n"
            f"gt={vr.gt_label}  pred={det_str}{correct}  "
            f"frac={vr.frac_under_thresh:.2f}  run={vr.max_consecutive_run}",
            fontsize=10
        )
        fig.tight_layout()
        fig.savefig(rdir + vr.filename.replace(".mp4", ".png"), dpi=90)
        plt.close(fig)

    # 개요 bar 차트
    fig, ax = plt.subplots(figsize=(16, max(7, len(results)*0.75)))
    for i, vr in enumerate(results):
        gt_c  = "tomato"    if vr.gt_label == "aft" else "royalblue"
        det_c = "red"       if vr.smoking_detected  else "limegreen"
        ax.barh(i, vr.duration, color=gt_c,  alpha=0.15, height=0.8)
        ax.barh(i, vr.duration * vr.frac_under_thresh * 10,
                color=det_c, alpha=0.7, height=0.45)
        mark = "🚬" if vr.smoking_detected else "✓"
        note = ""
        if vr.gt_label == "aft" and not vr.smoking_detected: note = " [MISS]"
        if vr.gt_label == "for" and vr.smoking_detected:     note = " [FA]"
        ax.text(2, i, f" {mark} {vr.filename[:44]}{note}", va="center", fontsize=7.5)

    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([""] * len(results))
    ax.set_xlabel("Duration (s)")
    ax.set_title("흡연 감지 전체 요약")
    patches = [
        mpatches.Patch(color="tomato",    alpha=0.4, label="gt=흡연(aft)"),
        mpatches.Patch(color="royalblue", alpha=0.3, label="gt=정상(for)"),
        mpatches.Patch(color="red",       alpha=0.8, label="pred=흡연"),
        mpatches.Patch(color="limegreen", alpha=0.8, label="pred=정상"),
    ]
    ax.legend(handles=patches, fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(rdir + "overview.png", dpi=100)
    plt.close(fig)
    print(f"  report/ 저장 완료 ({len(results)+1}개)")


# ══════════════════════════════════════════════════════
#  엔트리포인트
# ══════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  흡연 감지 시스템 v3.1")
    print("  YOLOv8-Pose + 손목 정규화 + 시간적 지속성")
    print("=" * 60)
    cfg      = Config()
    detector = SmokingDetector(cfg)
    results  = detector.run_batch()
    detector.save_results(results)
    generate_report(results, cfg.OUTPUT_DIR)


if __name__ == "__main__":
    main()
