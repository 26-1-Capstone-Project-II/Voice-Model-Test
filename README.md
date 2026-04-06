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
├── [iOS 변환 모델]
│   └── Whisper_CoreML_Model/        # 🍏 ANE 호환 CoreML 변환 완료된 앱 탑재용 모델
│
├── [실패한 파이프라인 (보관용)]
│   └── legacy_pipelines/            # 이전 시도 (자모 CTC, wav2vec2) 및 관련 실험 코드 모음
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

### 테스트 및 검증
```bash
# 발음 전사 성능 및 베이스라인(원본 모델) 대비 교정 문제 극복 검증
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python test_whisper_phonetic.py \
    --model_path best_model_whisper/best

# 파일 단독 평가 및 자모 레벨 피드백 확인
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python pronunciation_evaluator.py \
    --model_path best_model_whisper/best \
    --audio recording.wav \
    --target "같이 먹을까?"
```

---

## 🔥 발음 평가 엔진 동작 사례 (Demo)

스크립트(`test_whisper_phonetic.py`) 실행 시 나타나는 **실제 파인튜닝 모델의 오류 감지 결과**입니다. 
단순한 텍스트 비교가 아닌, 모델 입력을 통해 **사람의 귀처럼** 발음 오류를 정확하게 잡아냅니다.

```text
[성공 케이스 1: 연음의 완벽한 전사]
목표: 좋네요
기대 발음: 존네요
실제 디코딩: 존네요 (정서법 '좋네요'가 아닌 통과음 완벽 전사)
결과: 점수 100%

[성공 케이스 2: 족집게 오류 감지]
목표: 닭볶음
기대 발음: 닥뽀끔
모델이 들은 소리: 박뿌금
오류 피드백 추출: ㄷ→ㅂ, ㅗ→ㅜ, ㄲ→ㄱ
(사용자가 어떤 모음과 자음을 다르게 발음했는지 정확히 도출!)

[성공 케이스 3: 묵음/비음화 오류 감지]
목표: 학교에 갑니다
기대 발음: 학꾜에 감니다
모델이 들은 소리: 학꾜에 갑니다 (비음화 '감니다'가 안 된 소리)
오류 피드백 추출: ㅁ→ㅂ
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
| **Zeroth-Korean (낭독체 51h) + g2pk 파인튜닝** | **0.088 (8.8%)** | 0.20 | **대성공! 발음열 전사 및 상세 자모 피드백 완벽 지원** |

---

## 🍏 iOS (CoreML) 발음 평가 엔진 연동 가이드

서버에서 훈련된 최고 성능의 Whisper 모델을 iOS 앱(아이폰 등)에서 오프라인으로 쌩쌩하게 돌려볼 수 있도록 Apple 전용 `CoreML (mlmodelc)` 포맷으로 압축 변환해 두었습니다!

### 1️⃣ 모델 다운로드 (Mac 환경)
모델은 Git LFS 대용량 스토리지에 저장되어 있습니다. Mac 터미널을 열고 기기로 복사해 옵니다.

```bash
# 1. 깃허브 저장소 클론
git clone https://github.com/26-1-Capstone-Project-II/Voice-Model-Test.git
cd Voice-Model-Test

# 2. LFS 용량 제한으로 잘린 모델 파일 원본 받아오기
git lfs pull
```

### 2️⃣ Xcode UI 세팅 (SwiftUI 앱)
1. Mac에서 Xcode를 켜고 새로운 **iOS App (SwiftUI)** 프로젝트를 생성합니다.
2. `File` > `Add Package Dependencies...` 에서 `https://github.com/argmaxinc/WhisperKit` 을 입력하고 설치합니다.
3. 방금 다운받은 깃허브 폴더 안에 있는 `Whisper_CoreML_Model` 폴더를 통째로 **Xcode 파일 속성 영역에 드래그 앤 드롭** 합니다. (이 폴더가 아이폰의 AI 뇌가 됩니다!)
   - **팝업 주의:** `Create groups` 옵션을 선택하세요.
4. Xcode 프로젝트 설정 탭(Target) `Info` 메뉴에서 **Microphone Usage Description** (마이크 권한)을 "발음 평가를 위해 마이크를 사용합니다." 로 추가합니다.

### 3️⃣ 데모 앱 구동
저장소에 올려둔 `IOS_Whisper_Test_App.swift` 코드를 열고, Xcode 안의 내 기본 뷰(`ContentView.swift` 등)에 전체 복사/붙여넣기 합니다.
이후 시뮬레이터나 본인 아이폰 기기로 연결하여 앱을 빌드(Cmd+R) 하면, **소리나는 대로 모두 잡아내는 세상에서 단 하나뿐인 발음 진단기 앱**이 내 폰 안에서 돌아가는 것을 볼 수 있습니다!

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
