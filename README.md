# 🔊 온보이스 (On-Voice) — 발음 교정 모듈

> 청각장애인을 위한 발음 교정 앱 **온보이스**의 AI 발음 분석 백엔드  
> CTC + Greedy Decoding + G2P 기반 음절 단위 발음 채점 시스템

---

## 📁 파일 구성

```
.
├── pronunciation_scorer.py       # 핵심 발음 분석기 (1단계 베이스라인)
├── test_final.py                 # 파일 + 마이크 통합 테스트
├── test_g2p.py                   # G2P 발음 전사 재점수 검증
├── pronunciation_tester.html     # 브라우저 UI 데모 대시보드
├── requirements.txt              # 의존성 목록
│
├── korean_g2p_nomecab.py         # MeCab 없이 동작하는 G2P (Windows 호환)
│
├── [데이터 탐색]
│   ├── 00_unzip_labels.py        # 라벨링 ZIP 압축 해제
│   ├── 01_explore_data.py        # 데이터셋 구조 탐색
│   ├── 02_phoneme_utils.py       # 음소 분리 + 오류 탐지 유틸리티
│   ├── 03_build_error_pairs.py   # 음소 오류 탐지용 데이터셋 구축
│   └── 04_train_model.py         # 음소 오류 탐지 모델 학습
│
├── [파인튜닝 파이프라인]
│   ├── 05_unzip_wavs.py          # 원천 WAV ZIP 압축 해제
│   ├── A_prepare_finetune_data.py # 파인튜닝용 데이터 준비 (G2P 라벨 생성)
│   ├── finetune_simple.py        # ★ 현재 사용 중인 파인튜닝 스크립트
│   └── vad_segment.py            # ★ VAD 기반 오디오 세그멘테이션
│
└── README.md
```

---

## 🗺️ 개발 로드맵

### ✅ 1단계 완료 — CTC + Greedy Decoding 베이스라인

- `w11wo/wav2vec2-xls-r-300m-korean` 베이스라인 채택
- G2P (g2pk, descriptive=True) 발음 전사 채점 기준 통합
- 음절 단위 Diff 시각화 + CER 기반 점수
- 마이크 / 파일 입력, TTS 재생, 연습 모드 구현

### ✅ 2단계 진행 중 — 구음장애 데이터 파인튜닝

#### 완료
- AIHub 구음장애 음성인식 데이터셋 (언어청각장애) 다운로드 및 탐색
- 데이터 구조 파악: WAV 1개 = 세션 전체 녹음 (13~45분)
- MeCab 없이 동작하는 G2P 구현 (`korean_g2p_nomecab.py`)
- WAV 압축 해제 완료 (`TS02_언어청각장애.zip` → 1,283개 WAV)
- 1차 파인튜닝 시도 → CER 0.70 (오디오-텍스트 불일치 문제 확인)
- Silero-VAD 기반 세그멘테이션 완료
  - 1,280개 세션 → **155,453개 세그멘트** 생성
  - train: 127,320개 / val: 11,223개 / test: 16,910개

#### 진행 예정
- 세그멘트 데이터로 파인튜닝 재시도
- ONNX → CoreML 변환 → iOS 탑재

---

## 🚀 2단계 파인튜닝 실행 방법

### 환경 설치
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers datasets accelerate evaluate jiwer librosa soundfile
pip install silero-vad soundfile
```

### 다음 실행 단계

**1. 15초 초과 세그멘트 필터링**
```bash
python -c "
import json
from pathlib import Path
for split in ['train', 'validation', 'test']:
    path = Path('./segmented_dataset') / f'{split}.jsonl'
    lines = path.read_text(encoding='utf-8').splitlines()
    filtered = [l for l in lines if json.loads(l).get('duration', 0) <= 15]
    path.write_text('\n'.join(filtered), encoding='utf-8')
    print(f'{split}: {len(lines):,} → {len(filtered):,}개')
"
```

**2. 파인튜닝 실행**
```bash
python finetune_simple.py \
  --wav_dir ./segmented_dataset/wavs \
  --json_dir ./segmented_dataset \
  --batch_size 4 --grad_accum 8
```

**3. 파인튜닝 모델 적용**
```bash
python pronunciation_scorer.py --practice --model finetuned_model/best
```

---

## 🧠 기술 구조

### 전체 파이프라인
```
구음장애 WAV (세션 녹음)
        ↓ Silero-VAD + 균등 분할
문장 단위 세그멘트 (1~15초)
        ↓ Transcript 순서 매핑
오디오 ↔ G2P 발음 전사 쌍
        ↓ wav2vec2 CTC 파인튜닝
구음장애 특화 ASR 모델
        ↓
입력 음성 → 발음 인식 → G2P 정답 비교 → CER 점수 + 오류 위치
```

### 핵심 설계 결정

**CTC + Greedy Decoding (LM 없음)**
LM이 개입하면 발음 오류를 맞춤법으로 교정해버리기 때문에
음향 신호에만 의존하는 Greedy Decoding 사용.

**G2P 채점 기준 (descriptive=True)**
```python
g2p("같이 해볼까", descriptive=True)  # → "가치 해볼까"
g2p("좋네요", descriptive=True)       # → "존네요"
```
학습 라벨과 채점 기준을 동일하게 맞춰 일관성 확보.

---

## 📊 현재 성능

| 구분 | CER | 비고 |
|------|-----|------|
| 베이스라인 (파인튜닝 전) | ~0.99 | 구음장애 음성 미인식 |
| 1차 파인튜닝 (오디오-텍스트 불일치) | 0.70 | 데이터 매핑 문제 |
| 2차 파인튜닝 예정 (VAD 세그멘테이션) | - | 진행 예정 |

---

## ⚙️ 학습 환경

| 항목 | 사양 |
|------|------|
| GPU | NVIDIA GTX 1660 Super (VRAM 6GB) |
| CUDA | 13.0 (PyTorch cu128 사용) |
| Python | 3.13 |
| transformers | 5.3.0 |
| 학습 데이터 | AIHub 구음장애 음성인식 (언어청각장애, 1,280개 세션) |

---

## 📦 의존성

```
torch>=2.0.0
transformers>=5.0.0
librosa>=0.10.0
jiwer>=3.0.0
sounddevice>=0.4.6
evaluate>=0.4.0
accelerate>=1.0.0
silero-vad
soundfile
g2pk>=0.9.4
gtts>=2.3.0
pyttsx3>=2.90
```

---

## 🔗 참고

- 모델: [w11wo/wav2vec2-xls-r-300m-korean](https://huggingface.co/w11wo/wav2vec2-xls-r-300m-korean)
- G2P: [g2pk](https://github.com/Kyubyong/g2pK)
- VAD: [Silero-VAD](https://github.com/snakers4/silero-vad)
- AI Hub 구음장애 데이터: [aihub.or.kr](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=608)
