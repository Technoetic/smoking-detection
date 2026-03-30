<div align="center">

# 흡연 감지 시스템

### 드론 탑뷰 CCTV × YOLOv8-Pose × 규칙 기반 포즈 분석

[![Live Demo](https://img.shields.io/badge/라이브_데모-Railway-blueviolet?style=for-the-badge)](https://smoking-api-production.up.railway.app/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-백엔드-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![YOLOv8](https://img.shields.io/badge/YOLOv8--Pose-Ultralytics-00FFFF?style=for-the-badge)](https://ultralytics.com)
[![Railway](https://img.shields.io/badge/Railway-배포-7B2FBE?style=for-the-badge&logo=railway&logoColor=white)](https://railway.app)

<br/>

**드론 탑뷰 CCTV 영상에서 연기 없이 자세만으로 흡연을 감지합니다.**<br/>
YOLOv8-Pose 키포인트 추적 + 손목 정규화 지표 + 시간적 지속성 앙상블.

[시스템 구조](#-시스템-구조) · [핵심 알고리즘](#-핵심-알고리즘) · [성능](#-성능-평가) · [실행 방법](#-실행-방법)

<br/>

[![Live Demo](docs/demo-btn.gif)](https://smoking-api-production.up.railway.app/)

<br/>

<img src="docs/demo.gif" width="90%" alt="흡연 감지 시스템 데모 — 영상 업로드 → 분석 → 결과 및 스냅샷"/>

</div>

---

## 목차

- [프로젝트 소개](#-프로젝트-소개)
- [핵심 아이디어](#-핵심-아이디어)
- [시스템 구조](#-시스템-구조)
- [핵심 알고리즘](#-핵심-알고리즘)
- [성능 평가](#-성능-평가)
- [파일 구성](#-파일-구성)
- [기술 스택](#-기술-스택)
- [실행 방법](#-실행-방법)

---

## 프로젝트 소개

> [!IMPORTANT]
> **드론 탑뷰에서는 연기가 보이지 않습니다.** — 기존 연기/색상 감지 방식은 원천 불가능합니다.

공원, 금연구역, 건물 외곽의 드론 CCTV에서 흡연자를 자동으로 탐지합니다.
연기 대신 **손이 입으로 향하는 포즈**를 측정하는 완전히 새로운 접근 방식을 사용합니다.

### 문제 정의 & 해결책

```mermaid
graph LR
    subgraph problems["기존 방식의 문제"]
        P1["드론 탑뷰에서\n연기가 보이지 않음"]
        P2["색상/연기 감지\n흰 옷·마스크와 혼동"]
        P3["학습 데이터\n부족 (22개 영상)"]
        P4["다양한 자세\n앉기·서기·움직임"]
    end

    subgraph solutions["해결책"]
        S1["포즈 기반 감지\n손목 위치 추적"]
        S2["wrist_norm 지표\n어깨 기준 정규화"]
        S3["규칙 기반 앙상블\n학습 불필요"]
        S4["코~어깨 거리\n자세 불변 정규화"]
    end

    P1 --> S1
    P2 --> S2
    P3 --> S3
    P4 --> S4

    style P1 fill:#ff6b6b,color:#fff
    style P2 fill:#ff6b6b,color:#fff
    style P3 fill:#ff6b6b,color:#fff
    style P4 fill:#ff6b6b,color:#fff
    style S1 fill:#51cf66,color:#fff
    style S2 fill:#51cf66,color:#fff
    style S3 fill:#51cf66,color:#fff
    style S4 fill:#51cf66,color:#fff
```

---

## 핵심 아이디어

### wrist_norm — 핵심 지표

흡연 동작의 본질은 **손을 입으로 반복적으로 올리는 것**입니다.

$$\text{wrist\_norm} = \frac{\min(\text{wrist}_y) - \text{nose}_y}{\text{shoulder}_y - \text{nose}_y}$$

| wrist_norm 값 | 의미 |
|:---:|:---|
| **< 0.5** | 손이 코보다 위 → 강한 흡연 의심 |
| **0.5 ~ 1.5** | 손이 얼굴~어깨 사이 → 의심 |
| **> 1.5** | 손이 어깨 아래 → 정상 |

> [!NOTE]
> 드론 탑뷰에서는 앉거나 서거나 상관없이 **코~어깨 거리**로 정규화하면 자세 변화에 강인합니다.

### COCO Keypoint 매핑

```
NOSE(0) ─── 기준점
  │
L_SHOULDER(5) ─── R_SHOULDER(6)   ← 정규화 기준
  │                       │
L_ELBOW(7)           R_ELBOW(8)
  │                       │
L_WRIST(9) ◉         ◉ R_WRIST(10)  ← 추적 대상
```

### 판정 로직

```mermaid
graph TD
    A["영상 입력"] --> B["YOLOv8n-pose 추론\n매 프레임"]
    B --> C["키포인트 추출\n17개 COCO landmark"]
    C --> D["wrist_norm 계산\n손목↔코 정규화 거리"]
    D --> E{"wrist_norm < 1.5?"}
    E -->|YES| F["의심 프레임 마킹\nis_suspect = True"]
    E -->|NO| G["정상 프레임"]
    F --> H["frac 계산\n의심 프레임 비율"]
    F --> I["run 계산\n최장 연속 구간"]
    H & I --> J{"frac ≥ 0.02\nOR run ≥ 1?"}
    J -->|YES| K["🚬 흡연 판정"]
    J -->|NO| L["✅ 정상 판정"]

    style K fill:#e03131,color:#fff
    style L fill:#2f9e44,color:#fff
```

---

## 시스템 구조

```mermaid
graph TB
    subgraph browser["브라우저 — index.html"]
        UP["📁 영상 업로드\n드래그&드롭 / 파일 선택"]
        VD["판정 결과\n흡연 / 정상"]
        MT["지표 카드\n의심 비율 · 연속 구간 · 프레임 수"]
        SN["🔍 감지 장면 스냅샷\nbbox + skeleton 오버레이"]
        CH["wrist_norm 타임라인\n실시간 그래프"]
    end

    subgraph server["FastAPI 서버 — app.py"]
        API["POST /analyze\n영상 수신 → 분석 → JSON 반환"]
        SNAP["extract_snapshots()\n의심 프레임 크롭 + base64"]
    end

    subgraph detector["감지 엔진 — smoking_detector.py"]
        YOLO["YOLOv8n-pose\n사람 감지 + 17 keypoint"]
        PA["PoseAnalyzer\nwrist_norm / elbow_norm 계산"]
        MA["MotionAnalyzer\nMOG2 + Farneback Optical Flow"]
        DEC["_decide()\nfrac & run 기반 최종 판정"]
    end

    browser -->|mp4 업로드| server
    server --> detector
    detector --> YOLO --> PA & MA
    PA & MA --> DEC
    DEC -->|VideoResult| SNAP
    SNAP -->|JSON + base64 스냅샷| browser

    style browser fill:#1a1a2e,color:#fff
    style server fill:#16213e,color:#fff
    style detector fill:#0f3460,color:#fff
```

---

## 핵심 알고리즘

### 1. PoseAnalyzer — 손목 정규화

```python
wrist_norm = (min(wrist_y) - nose_y) / (shoulder_y - nose_y)
```

- 드론 탑뷰 특성상 몸 전체가 작게 보임
- 절대 픽셀 거리 대신 **신체 비율로 정규화** → 거리·각도 불변
- 어깨 미검출 시 편측 어깨로 fallback

### 2. MotionAnalyzer — 상체 모션

```mermaid
graph LR
    ROI["상체 ROI\n바운딩박스 상단 55%"] --> MOG["MOG2\n배경 차분\n가중치 60%"]
    ROI --> FLOW["Farneback\nOptical Flow\n가중치 40%"]
    MOG & FLOW --> SCORE["motion_score\n[0, 1]"]
```

흡연 동작의 특징인 **손 상하 반복 운동** → y축 광학 흐름 우세 탐지

### 3. 시간적 판정 (OR 앙상블)

| 단서 | 조건 | 의미 |
|:---:|:---:|:---|
| `frac` | ≥ 0.02 | 전체 프레임의 2% 이상이 의심 포즈 |
| `run` | ≥ 1 | 연속 의심 프레임이 1개 이상 존재 |

- **OR 조건**: 단 1프레임이라도 강한 의심 시 즉시 판정
- 짧은 순간의 흡연 동작도 놓치지 않음 (높은 Recall)

### 4. 스냅샷 — SUSPECT 크롭 확대

```mermaid
graph LR
    FR["의심 프레임\n(wrist_norm 최저 순)"] --> INFER["YOLOv8 재추론\nbbox + keypoint"]
    INFER --> CROP["SUSPECT 인물 크롭\n여백 50% 추가"]
    CROP --> RESIZE["640px 리사이즈"]
    RESIZE --> OVL["bbox · skeleton · 손목 마커\n오버레이"]
    OVL --> B64["base64 JPEG\n→ 브라우저 표시"]
```

---

## 성능 평가

> 22개 영상 (흡연 16개 · 정상 5개 · 세팅 1개), 규칙 기반 — 학습 없음

| 지표 | 값 |
|:---:|:---:|
| **F1-Score** | **0.8333** |
| **Recall** | **93.8%** |
| **Precision** | **75.0%** |
| **Accuracy** | **71.4%** |

```
TP=15  TN=0  FP=5  FN=1
```

> [!NOTE]
> FP 5건은 흰 마스크·모자를 착용한 인물이 손을 올리는 동작으로, 규칙 기반의 구조적 한계입니다.
> FN 1건은 앉은 자세로 인해 키포인트 검출이 불안정한 경우입니다.

### 파라미터

| 파라미터 | 값 | 설명 |
|:---:|:---:|:---|
| `FRAC_THRESH_T` | 0.50 | frac 계산 wrist_norm 임계값 |
| `FRAC_THRESH_VAL` | 0.02 | 흡연 판정 최소 의심 비율 |
| `RUN_THRESH_T` | 1.50 | run 계산 wrist_norm 임계값 |
| `RUN_THRESH_N` | 1 | 최장 연속 구간 최소값 |

---

## 파일 구성

```
smoking-detection/
├── app.py                  # FastAPI 서버 + 스냅샷 생성
├── smoking_detector.py     # 감지 엔진 (PoseAnalyzer · MotionAnalyzer · SmokingDetector)
├── make_videos.py          # 결과 동영상 생성 (overlay + analysis)
├── index.html              # 웹 UI (업로드 · 결과 · 그래프 · 스냅샷)
├── requirements.txt        # Python 의존성
└── Dockerfile              # Railway 배포용
```

---

## 기술 스택

```mermaid
graph LR
    subgraph vision["컴퓨터 비전"]
        YOLO["YOLOv8n-pose\nUltralytics"]
        OCV["OpenCV\nMOG2 + Farneback"]
        NP["NumPy"]
    end

    subgraph backend["백엔드"]
        FAPI["FastAPI"]
        UV["Uvicorn"]
        MP["python-multipart"]
    end

    subgraph frontend["프론트엔드"]
        HTML["Vanilla HTML/CSS/JS"]
        CANVAS["Canvas API\n타임라인 그래프"]
    end

    subgraph infra["인프라"]
        DOCKER["Docker"]
        RAIL["Railway"]
        GH["GitHub"]
    end

    style vision fill:#264653,color:#fff
    style backend fill:#2a9d8f,color:#fff
    style frontend fill:#e9c46a,color:#000
    style infra fill:#e76f51,color:#fff
```

| 분류 | 기술 | 용도 |
|:---:|:---:|:---|
| **포즈 감지** | YOLOv8n-pose | 17개 COCO 키포인트 추출 |
| **모션 분석** | OpenCV MOG2 | 배경 차분 기반 활동량 측정 |
| **광학 흐름** | Farneback | 상체 y축 움직임 측정 |
| **백엔드** | FastAPI + Uvicorn | REST API + 정적 파일 서빙 |
| **프론트엔드** | Vanilla JS + Canvas | 업로드 UI + 타임라인 그래프 |
| **배포** | Railway + Docker | 컨테이너 기반 클라우드 배포 |

---

## 실행 방법

### 사전 요구사항

> [!WARNING]
> CUDA GPU 환경 권장 (CPU에서도 동작하나 영상당 30~60초 소요)

- Python 3.11+
- CUDA 지원 GPU (선택사항)

### 로컬 실행

```bash
# 1. 저장소 클론
git clone https://github.com/Technoetic/smoking-detection.git
cd smoking-detection

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 서버 실행
python app.py

# 4. 브라우저 접속
open http://localhost:8000
```

### Docker 실행

```bash
docker build -t smoking-detection .
docker run -p 8000:8000 smoking-detection
```

### Railway 배포

```bash
railway up
```

### 배포 흐름

```mermaid
graph LR
    DEV["로컬 개발"] -->|git push| GH["GitHub"]
    GH -->|자동 감지| RAIL["Railway"]
    RAIL -->|Dockerfile| BLD["Docker 빌드\nYOLOv8 모델 다운로드"]
    BLD --> SRV["FastAPI 서버 시작\nPORT 환경변수 자동 적용"]
    SRV --> LIVE["🚀 라이브 배포 완료"]

    style DEV fill:#339af0,color:#fff
    style RAIL fill:#7950f2,color:#fff
    style LIVE fill:#40c057,color:#fff
```

---

<div align="center">

**드론 CCTV 인프라를 위한 금연구역 자동 모니터링 솔루션**

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.app)

<br/>

<img src="docs/demo-footer.gif" width="90%" alt="footer"/>

</div>
