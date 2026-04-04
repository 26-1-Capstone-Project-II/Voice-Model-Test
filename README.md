# 🔊 온보이스 (On-Voice) — 발음 교정 모듈

> 청각장애인을 위한 발음 교정 앱 **온보이스**의 AI 발음 분석 백엔드
> Whisper Tiny + G2P 기반 발음 전사 & 자모 레벨 오류 감지 시스템

---

## 📁 파일 구성

```
.
├── [Whisper 발음 전사 파이프라인] ★ 현재 진행 중
│   ├── finetune_whisper.py          # ★ Whisper tiny 발음 전사 파인튜닝
│   ├── pronunciation_evaluator.py   # 발음 평가 엔진 (자모 레벨 비교)
│   └── test_whisper_phonetic.py     # 발음 전사 테스트 + 베이스라인 비교
│
├── [공용 모듈]
│   ├── korean_g2p_nomecab.py        # MeCab 없이 동작하는 G2P (Windows 호환)
│   ├── jamo_utils.py                # 자모 Vocab 생성 + 음절↔자모 변환 유틸리티
│   └── vad_segment.py               # Silero-VAD 기반 오디오 세그멘테이션
│
├── [이전 시도 (자모 CTC)] — 학습 기록용
│   ├── finetune_jamo.py             # 자모 vocab CTC 파인튜닝 (wav2vec2)
│   └── test_jamo_model.py           # 자모 CTC 모델 검증
│
├── [1단계 베이스라인]
│   ├── pronunciation_scorer.py      # CTC + Greedy Decoding 베이스라인
│   ├── test_final.py                # 파일/마이크 통합 테스트
│   ├── test_g2p.py                  # G2P 발음 전사 검증
│   └── pronunciation_tester.html    # 브라우저 UI 데모
│
└── README.md
```

---

## 🗺️ 개발 로드맵

### ✅ 1단계 완료 — CTC + Greedy Decoding 베이스라인

- `w11wo/wav2vec2-xls-r-300m-korean` 베이스라인 채택
- G2P (g2pk, descriptive=True) 발음 전사 채점 기준 통합
- 음절 단위 Diff 시각화 + CER 기반 점수

### ⚠️ 2단계 완료 — 자모 Vocab + wav2vec2 CTC (중단)

| 시도 | 데이터 | 결과 | 원인 |
|------|--------|------|------|
| 음절 vocab | 155k 세그멘트 | loss 3.31 | Vocab OOV 문제 |
| 자모 vocab (1k) | 1,024 매칭 | loss 0.80 | 과적합 |
| 자모 vocab (155k) | 117k 필터링 | CER 0.92 | VAD 균등분할 → 라벨 불일치 |

**중단 사유:**
- CTC는 오디오-라벨 strict alignment 필요 → VAD 균등분할 데이터에서 학습 불가
- wav2vec2-300m (1.2GB) → iOS 온디바이스 배포 불가

### 🔄 3단계 진행 중 — Whisper Tiny 발음 전사 ⭐

**핵심 전환점 1: 모델 아키텍처**
```
wav2vec2 CTC:   label과 audio가 정확히 정렬 필요 → VAD 실패 시 학습 불가
Whisper Seq2Seq: attention으로 자체 정렬 → 유연한 학습 가능

wav2vec2-300m:  1.2GB → iOS ❌
Whisper tiny:   150MB → iOS/CoreML ✅ (양자화 시 ~40MB)
```

**핵심 전환점 2: 학습 데이터 대전환 (구음장애 → 낭독체)**
VAD 정렬 문제를 해결하려다 발견한 프로젝트의 **가장 중요한 인사이트**.
발음 평가를 위해 "구음장애 데이터"가 아닌 **"정상 낭독체(Zeroth-Korean) 데이터"**를 사용해야 합니다.

---

## 🎯 핵심 설계: 소리나는 대로 텍스트 출력

### 왜 기존 ASR을 쓸 수 없는가

```
기존 ASR 모델 (Whisper, CLOVA 등):
  사용자 발화: "가치 머글까?" (소리)
  기존 모델 출력: "같이 먹을까?" ← 맞춤법 보정
  → 발음 오류를 숨김 → 교정 앱에서 사용 불가

우리 모델 (Whisper + G2P 파인튜닝):
  사용자 발화: "가티 머글까?" (소리)
  우리 모델 출력: "가티 머글까?" ← 소리나는 대로
  → 기대 발음 "가치 머글까?"와 비교 → ㅊ→ㅌ 오류 감지!
```

### 앱 전체 흐름

```
┌─────────────────────────────────────────────┐
│  1. 사용자 목표 문장 입력                      │
│     "같이 먹을까?"                            │
│          ↓ G2P (descriptive=True)            │
│     "가치 머글까?" (기대 발음)                  │
│                                              │
│  2. 사용자 발화 녹음 → Whisper tiny            │
│     "가티 머글까?" (실제 발음)                  │
│                                              │
│  3. 자모 레벨 비교                             │
│     기대: ㄱㅏㅊㅣ ㅁㅓㄱㅡㄹㄲㅏ               │
│     실제: ㄱㅏㅌㅣ ㅁㅓㄱㅡㄹㄲㅏ               │
│               ↑                              │
│          ㅊ→ㅌ 오류 감지!                      │
│                                              │
│  4. 피드백: 발음 점수 85%, 오류 위치 표시       │
└─────────────────────────────────────────────┘
```

### 학습 전략: 왜 구음장애가 아닌 '정상 낭독체'를 쓰는가? (💡 핵심 인사이트)

**[오류 시나리오] 구음장애 음성으로 학습할 경우:**
- 오디오: "가티 머거요" (환자의 틀린 발음)
- 라벨(G2P): "가치 머거요" (원래 의도한 정답 발음)
- **문제점:** 모델이 "가티"라는 틀린 소리를 들어도 "가치"로 **자동 교정(Auto-correct)**하는 법을 배우게 됩니다. 앱에서 환자가 틀리게 발음해도 모델이 스스로 빈칸을 채워 정답으로 출력해버려, 발음 오류 추적이 불가능해집니다.

**[성공 시나리오] 정상 낭독체(Zeroth-Korean)로 학습할 경우:**
- 오디오: "가치 머거요" (정상인의 정확한 발음)
- 라벨(G2P): "가치 머거요" 
- **해결책:** 모델은 100% "정확한 소리 = 정확한 발음 기호"의 1:1 매핑만을 학습합니다.
- **결과:** 추론 시 환자가 "가티 머거요"라고 틀리게 말하면, 모델은 교정하는 법을 배운 적이 없기 때문에 자기가 들은 가장 비슷한 소리인 **"가티 머거요"를 날것 그대로(Raw) 출력**합니다! → **발음 오류 감지 대성공!**

---

## 🚀 실행 방법

### 환경 설치
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers datasets accelerate evaluate jiwer librosa soundfile
pip install openai-whisper gtts sounddevice
```

### 데이터 다운로드 및 전처리 (Zeroth-Korean)
```bash
# HuggingFace 데이터 다운로드 및 변환 (51시간 분량 오디오 → WAV/JSONL)
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python prepare_zeroth.py
```

### Whisper 파인튜닝 (서버)
```bash
# 1. 빠른 실험 (10k 샘플, ~2시간)
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python finetune_whisper.py \
    --json_dir zeroth_dataset \
    --apply_g2p \
    --max_samples 10000 --lr 2e-5 --num_epochs 3

# 2. 전체 학습 (~22,000개, ~10시간)
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python finetune_whisper.py \
    --json_dir zeroth_dataset \
    --apply_g2p \
    --lr 2e-5 --num_epochs 3 --batch_size 8 --grad_accum 2
```

### 테스트
```bash
# 발음 전사 + 베이스라인 비교
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python test_whisper_phonetic.py \
    --model_path best_model_whisper/best

# 발음 평가 단독 실행
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python pronunciation_evaluator.py \
    --model_path best_model_whisper/best \
    --audio recording.wav \
    --target "같이 먹을까?"
```

---

## 📊 성능 기록

### wav2vec2 CTC 시도 (중단)

| 구분 | CER | Loss | 비고 |
|------|-----|------|------|
| 베이스라인 (파인튜닝 전) | ~0.99 | - | 구음장애 음성 미인식 |
| 2차 (음절 vocab, VAD 세그멘트) | - | 3.31 | vocab OOV → 수렴 실패 |
| 3차 (자모 vocab, 1k 데이터) | ~0.7 | 0.80 | 과적합 (1,024개 부족) |
| 4차 (자모 vocab, 155k 세그먼트) | 0.92 | 2.78 | VAD 라벨 불일치 → 정체 |

### Whisper Tiny 시도 (진행 중)

### Whisper Tiny 시도 (현재 적용 모델 ⭐)

| 구분 | CER | Loss | 비고 |
|------|-----|------|------|
| **Zeroth-Korean (낭독체 51h) + g2pk 파인튜닝** | **0.088 (8.8%)** | 0.20 | **대성공! 완벽한 소리나는 대로(발음열) 전사 달성** |

---

## 📦 의존성

```
torch>=2.0.0
transformers>=4.30.0
librosa>=0.10.0
jiwer>=3.0.0
sounddevice>=0.4.6
evaluate>=0.4.0
accelerate>=1.0.0
silero-vad
soundfile
g2pk>=0.9.4
gtts>=2.3.0
openai-whisper
```

---

## 🔗 참고

- 모델: [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny)
- 이전 모델: [w11wo/wav2vec2-xls-r-300m-korean](https://huggingface.co/w11wo/wav2vec2-xls-r-300m-korean)
- G2P: [g2pk](https://github.com/Kyubyong/g2pK)
- VAD: [Silero-VAD](https://github.com/snakers4/silero-vad)
- iOS 배포: [WhisperKit](https://github.com/argmaxinc/WhisperKit)
- AI Hub 구음장애 데이터: [aihub.or.kr](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=608)
- HuggingFace Zeroth-Korean 데이터: [Bingsu/zeroth-korean](https://huggingface.co/datasets/Bingsu/zeroth-korean)
