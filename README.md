# 🔊 온보이스 (On-Voice) — 발음 교정 모듈

> 청각장애인을 위한 발음 교정 앱 **온보이스**의 AI 발음 분석 백엔드  
> CTC + Greedy Decoding + G2P 기반 발음 채점 시스템

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
├── [자모 Vocab 파인튜닝] ★ 현재 진행 중
│   ├── jamo_utils.py             # 자모 Vocab 생성 + 음절↔자모 변환 유틸리티
│   ├── finetune_jamo.py          # ★ 자모 기반 CTC 파인튜닝 (핵심)
│   └── test_jamo_model.py        # 파인튜닝 모델 검증 + 베이스라인 비교
│
├── [이전 파인튜닝 시도]
│   ├── finetune_full.py          # 음절 vocab 파인튜닝 (v1, loss 3.31 → 실패)
│   ├── finetune_lora.py          # LoRA 파인튜닝 시도
│   ├── finetune_simple.py        # 간단 파인튜닝 시도
│   └── vad_segment.py            # VAD 기반 오디오 세그멘테이션
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

### 🔄 2단계 진행 중 — 자모 Vocab 기반 파인튜닝

#### 이전 시도 (음절 vocab)
- AIHub 구음장애 음성인식 데이터셋 (언어청각장애) 다운로드 및 탐색
- Silero-VAD 기반 세그멘테이션: 1,280개 세션 → 155,453개 세그멘트
- `finetune_full.py`로 30 에폭 학습 → **Train Loss 3.31 (실패)**
  - 원인: 음절 vocab (1,207개) ↔ G2P 발음 전사 라벨 불일치 (OOV 문제)

#### 현재 진행: 자모 Vocab 재구축 ⭐
- **핵심 변경: 음절 vocab → 자모 vocab (57개)**
  - OOV 완전 해소 — 어떤 발음이든 자모로 분해 가능
  - 발음 오류 자모 레벨 정밀 감지
  - CTC head 부담 감소 (1,207 → 57 클래스)
- `jamo_utils.py`: 자모 분해/재조립 + Processor 생성
- `finetune_jamo.py`: CTC head 교체 + 자모 라벨 학습
- `test_jamo_model.py`: 모델 검증 및 베이스라인 비교

#### 검토 중: HuBERT 기반 전환 (차기 전략)
- `team-lucid/hubert-base-korean`: 음향 충실 출력, 크기 1/3

---

## 🧬 자모 Vocab 핵심 설계

### 왜 자모 vocab인가?

```
기존 (음절 vocab):
  vocab = ["가", "각", "간", ... ]  ← 1,207개
  "가치" → tokenizer → [1, 1100]
  ⚠️ G2P 출력 중 vocab에 없는 음절 → [UNK] → 학습 불가

변경 (자모 vocab):
  vocab = ["ㄱ", "ㅏ", "ㅊ", "ㅣ", ...]  ← 57개
  "가치" → 자모분해 → "ㄱㅏㅊㅣ" → tokenizer → [5, 25, 20, 45]
  ✅ 어떤 발음이든 100% 인코딩 가능
```

### 파이프라인 흐름

```
입력 문장:     "같이 해볼까"
    ↓ G2P (descriptive=True)
발음 전사:     "가치 해볼까"
    ↓ syllable_to_jamo()
자모 라벨:     "ㄱㅏㅊㅣ|ㅎㅐㅂㅗㄹㄲㅏ"    (| = 공백)
    ↓ Wav2Vec2CTCTokenizer
토큰 ID:       [5, 25, 20, 45, 4, 24, 26, ...]
    ↓ CTC 학습
모델 출력:     자모 시퀀스
    ↓ jamo_to_syllable()
최종 텍스트:   "가치 해볼까"    (사용자 피드백용)
```

### 자모 Vocab 구성 (총 57개)

| 구분 | 토큰 수 | 내용 |
|------|---------|------|
| 특수토큰 | 5 | `<pad>`, `<s>`, `</s>`, `<unk>`, `\|` |
| 초성 자음 | 19 | ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ |
| 중성 모음 | 21 | ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ |
| 겹받침 | 11 | ㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ |

---

## 🚀 자모 파인튜닝 실행 방법

### 환경 설치
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers datasets accelerate evaluate jiwer librosa soundfile
```

### 파인튜닝 실행 (서버)
```bash
# 1. 데이터 검증 (dry run)
CUDA_VISIBLE_DEVICES=0 python finetune_jamo.py --dry_run

# 2. 본 학습 (자모 vocab, lr=3e-5)
CUDA_VISIBLE_DEVICES=0 python finetune_jamo.py --lr 3e-5

# 3. 커스텀 설정
CUDA_VISIBLE_DEVICES=0 python finetune_jamo.py \
  --lr 3e-5 \
  --num_epochs 20 \
  --batch_size 2 \
  --grad_accum 8
```

### 모델 검증
```bash
# 파인튜닝 모델 단독 테스트 (gTTS 기반)
python test_jamo_model.py --model ./best_model_jamo/best

# 베이스라인과 비교
python test_jamo_model.py --model ./best_model_jamo/best --compare

# 실제 오디오 파일 테스트
python test_jamo_model.py --model ./best_model_jamo/best --audio my_voice.wav --text "같이 해볼까"
```

### 발음 분석 (1단계 베이스라인)
```bash
# 연습 모드: 문장 듣고 → 따라 말하기 → 분석
python pronunciation_scorer.py --practice

# 마이크 녹음 후 분석
python pronunciation_scorer.py --text "같이 해볼까" --mic

# 자동 테스트 (gTTS)
python pronunciation_scorer.py --test
```

---

## 🧠 기술 구조

### 전체 파이프라인
```
구음장애 WAV (세션 녹음)
        ↓ Silero-VAD + 균등 분할
문장 단위 세그멘트 (1~15초)
        ↓ Transcript → G2P → 자모 분해
오디오 ↔ 자모 라벨 쌍
        ↓ wav2vec2 CTC 파인튜닝 (자모 vocab 57개)
구음장애 특화 ASR 모델
        ↓
입력 음성 → 자모 인식 → 음절 재조립 → G2P 정답 비교 → CER 점수 + 오류 위치
```

### 핵심 설계 결정

**CTC + Greedy Decoding (LM 없음)**
LM이 개입하면 발음 오류를 맞춤법으로 교정해버리기 때문에
음향 신호에만 의존하는 Greedy Decoding 사용.

**자모 단위 Vocab (v2)**
음절 vocab은 G2P 발음 전사와 OOV 불일치 → 학습 실패.
자모 vocab (57개)으로 전환하여 완전한 발음 커버리지 확보.

**G2P 채점 기준 (descriptive=True)**
```python
g2p("같이 해볼까", descriptive=True)  # → "가치 해볼까"
g2p("좋네요", descriptive=True)       # → "존네요"
```
학습 라벨과 채점 기준을 동일하게 맞춰 일관성 확보.

---

## 📊 성능 기록

| 구분 | CER | Loss | 비고 |
|------|-----|------|------|
| 베이스라인 (파인튜닝 전) | ~0.99 | - | 구음장애 음성 미인식 |
| 1차 시도 (음절 vocab, 세션 단위) | 0.70 | - | 오디오-텍스트 불일치 |
| 2차 시도 (음절 vocab, VAD 세그멘트) | - | 3.31 | vocab OOV 문제 → 수렴 실패 |
| **3차 시도 (자모 vocab)** | - | - | **진행 중** ⭐ |

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
- HuBERT (차기): [team-lucid/hubert-base-korean](https://huggingface.co/team-lucid/hubert-base-korean)
