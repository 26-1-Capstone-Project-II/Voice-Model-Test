# 🔊 온보이스 (OnVoice) — 발음 교정 모듈

> 청각장애인을 위한 발음 교정 앱 **온보이스**의 AI 발음 분석 백엔드
> CTC + Greedy Decoding + G2P 기반 음절 단위 발음 채점 시스템

---

## 📁 파일 구성

```
.
├── pronunciation_scorer.py   # 핵심 발음 분석기 (메인 스크립트)
├── test_final.py             # 파일 + 마이크 통합 테스트
├── test_g2p.py               # G2P 발음 전사 재점수 검증
├── pronunciation_tester.html # 브라우저 UI 데모 대시보드
├── requirements.txt          # 의존성 목록
├── INSTALL.md                # 플랫폼별 설치 가이드
└── README.md                 # 이 파일
```

---

## 🚀 빠른 시작

### 1. 설치

```powershell
pip install torch transformers librosa jiwer sounddevice scipy numpy gtts pyttsx3 kss
```

> **Apple Silicon (M1/M2) Mac**
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
> pip install transformers librosa jiwer sounddevice scipy numpy gtts pyttsx3 kss
> ```

> **노이즈 제거 강화 (선택)**
> ```bash
> pip install noisereduce  # 설치만 하면 자동 적용
> ```

> **g2pk 설치 오류가 난다면** → [INSTALL.md](INSTALL.md) 참고

### 2. ★ 연습 모드 (핵심 기능)

```powershell
# 대화형: 문장을 직접 입력하며 연속 연습
python pronunciation_scorer.py --practice

# 단일 문장 지정
python pronunciation_scorer.py --practice --text "같이 해볼까"
```

```
연습할 문장을 입력하세요: 같이 해볼까

══════════════════════════════════════════════════════
  🎯 연습 문장
══════════════════════════════════════════════════════
  목표 문장  :  같이 해볼까
  발음 기준  :  가치 해볼까  ← G2P 변환

  🔊 재생 중: 「같이 해볼까」
  ✅ 재생 완료

  r=다시듣기  Enter=녹음시작 >
  🔴 녹음 중... 5초
  🎛️  전처리 완료  (5.0s → 1.2s, 무음 76% 제거)

  📊 발음 분석 결과
  [오류 1건]
  • '가치' → '가티' 로 발음됨  (ㅊ(파찰음(격)) → ㅌ(파열음(격)))

  다시 연습할까요?  Enter=재도전  q=종료 >
```

### 3. 🔬 자동 테스트 모드 (마이크 없이)

마이크 없이 gTTS 음성으로 전체 파이프라인을 검증할 수 있다.

```powershell
# 기본 테스트셋 5문장 자동 실행
python pronunciation_scorer.py --test

# 특정 문장만 테스트
python pronunciation_scorer.py --test --text "퇴근하고 싶어요"
```

### 4. 배치 연습 모드

```powershell
python pronunciation_scorer.py --batch
```

### 5. 마이크로 단일 분석

```powershell
python pronunciation_scorer.py --text "같이 해볼까" --mic
```

### 6. 오디오 파일로 분석

```powershell
python pronunciation_scorer.py --text "같이 해볼까" --audio my_voice.wav
```

### 7. G2P 변환 미리보기 (모델 로드 없이)

```powershell
python pronunciation_scorer.py --text "같이 해볼까" --preview
# 맞춤법 : 같이 해볼까
# 발음   : 가치 해볼까  ← G2P 변환됨
```

---

## 🧠 기술 구조

### 전체 흐름

```
입력 문장 ("같이 해볼까")
    │
    ├─ [G2P 변환] kss
    │       └─→ 발음 전사 ("가치 해볼까")  ← 채점 기준
    │
    ├─ [TTS 재생] gTTS → pyttsx3 fallback
    │       └─→ 사용자가 목표 발음을 귀로 확인
    │             r 입력 시 반복 재생 가능
    │
    └─ [음성 입력] 마이크 or 파일
            │
            ▼
    [오디오 전처리]
    무음 체크 → 앞뒤 무음 제거 → 노이즈 제거 → RMS 정규화
            │
            ▼
    [wav2vec2 CTC Encoder]
    raw waveform → Transformer → logits [1, T, vocab]
            │
            ▼
    [Greedy Decoding]  ← LM 개입 없음
    argmax per time-step → CTC blank 제거
            │
            ▼
    인식 결과 (raw 그대로 — 후처리 없음)  ← 핵심 설계 원칙
            │
            ▼
    [음절 단위 Diff + 자모 피드백]
    G2P(정답) vs 모델 raw 출력 비교
    SequenceMatcher → equal / replace / delete / insert
    교체 오류 → 자모 분리 → "ㅊ(파찰음) → ㅌ(파열음)" 피드백
            │
            ▼
    점수 (CER 기반 0~100점) + 오류 위치 + 자모 피드백
            │
            ▼
    재도전 여부 선택 (95점 미만 시 자동 제안)
```

### 핵심 설계 결정

#### CTC + Greedy Decoding (LM 없음)

```python
# logits: [batch=1, time_steps, vocab_size]
logits = model(input_values).logits

# Greedy: 각 time-step에서 argmax만 선택
# → beam search 없음, LM rescoring 없음
# → "들리는 소리 그대로" 출력
predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)[0]
```

Language Model이 개입하면 "가치 해볼까"를 "같이 해볼까"로 맞춤법 교정해버린다.
발음 교정 앱에서는 LM 없이 음향 신호에만 의존하는 것이 핵심이다.

#### G2P 채점 기준 (kss)

```python
from kss import Kss
g2p = Kss("g2p")

g2p("같이 해볼까")   # → "가치 해볼까"  ✅ 연음 변환
g2p("좋네요")        # → "존네요"        ✅ 비음화
g2p("않아요")        # → "않아요"        ✅ 과잉 변환 없음
```

g2pk/g2pk2 계열은 Windows에서 MeCab C++ 빌드 오류가 발생하여 kss로 교체했다.
kss는 pecab 백엔드를 사용해 MeCab 없이 크로스플랫폼에서 동작한다.

#### ⚠️ 모델 출력에 G2P를 적용하면 안 되는 이유

```
[잘못된 방식] — 절대 하면 안 됨
  모델 출력 "싶어요" → G2P → "시퍼요"
  G2P 정답 "시퍼요"  vs  후처리 "시퍼요"  → 100점 ❌
  → 사용자가 "시포요"로 잘못 발음했어도 정상으로 오판

[올바른 방식]
  G2P 정답  : "시퍼요"
  모델 출력  : "싶어요"  ← raw 그대로 (발음 오류 정보 보존)
  비교 결과  : 오류 감지 ✅
```

모델 출력은 발음 오류 정보를 담고 있으므로 절대 후처리하지 않는다.
모델이 맞춤법 쪽으로 치우친 출력을 하는 문제는 2단계 파인튜닝으로 해결한다.

#### INT8 동적 양자화

```python
# torch 2.10+: torchao API 우선
from torchao.quantization import quantize_, int8_dynamic_activation_int8_weight
quantize_(model, int8_dynamic_activation_int8_weight())

# 이전 버전 fallback
torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

- 메모리 사용량 ~50% 감소 (~1.3GB → ~650MB)
- CPU 추론 속도 1.5~2x 향상
- CUDA/MPS 환경에서는 적용 안 함 (하드웨어 가속이 더 빠름)

#### 오디오 전처리 파이프라인

```python
def preprocess_audio(audio):
    # 1. 무음 체크 (RMS < 0.01 시 경고)
    # 2. 앞뒤 무음 제거 (librosa.trim, top_db=25)
    # 3. 노이즈 제거 (noisereduce 설치 시 자동 적용, prop_decrease=0.75)
    # 4. RMS 음량 정규화 (target=0.1)
```

#### 자모 단위 피드백

```
음절 교체: "가치" vs "가티"
    ↓ 유니코드 직접 분리 (0xAC00 기반, 외부 라이브러리 불필요)
ㄱ ㅏ ㅊ ㅣ  vs  ㄱ ㅏ ㅌ ㅣ
→ ㅊ(파찰음(격)) → ㅌ(파열음(격))
```

---

## 📊 모델 선정 과정

3종 모델을 동일 조건(gTTS 5문장)으로 비교 테스트했다.

| 모델 | 평균 점수 | 특징 |
|---|---|---|
| **w11wo/wav2vec2-xls-r-300m-korean** ✅ | **69.0점** | "같이→가치" 발음 전사 근접, 긴 문장 100점 |
| kresnik/wav2vec2-large-xlsr-korean | 69.3점 | 짧은 문장 안정적, 발음 전사 부정확 |
| Kkonjeong/wav2vec2-base-korean | 탈락 | 자모 분리 출력 (`ㄱㅏㅇㅣ...`) → 음절 CER 계산 불가 |

**채택 모델**: `w11wo/wav2vec2-xls-r-300m-korean`
- 크기: ~1.3GB (INT8 양자화 후 ~650MB)
- 발음 전사 친화적 인식 (`같이 → 가치`)
- 2단계 파인튜닝 베이스로 적합한 크기

---

## 📈 1단계 성능 결과

G2P 발음 전사 기준 채점으로 전환 후 성과 (gTTS 기준):

| 문장 | 맞춤법 기준 | 발음 기준 | 변화 |
|---|---|---|---|
| 같이 해볼까 | 60점 | **100점** | +40 ✅ |
| 저는 잘 들리지 않아요 | 100점 | **78점** | -22 (과잉 변환 방지) |
| 천천히 말해주세요 | 75점 | **75점** | ±0 |
| 오늘 날씨가 좋네요 | 50점 | **50점** | ±0 (TTS 음질 한계) |

> TTS 기준 점수. 실제 사람 목소리에서는 더 높은 점수 예상.

---

## ⚙️ CLI 옵션 전체

```
--text      목표 문장
--audio     오디오 파일 경로 (.wav / .mp3 / .m4a)
--mic       마이크 실시간 녹음
--practice  ★ 연습 모드: TTS 재생 → 따라 말하기 → 분석 → 재도전
--test      🔬 자동 테스트: 마이크 없이 gTTS로 파이프라인 검증
--duration  녹음 시간 초 (기본: 5)
--save      녹음 파일 저장 경로 (예: out.wav)
--batch     배치 연습 모드 (7문장 연속, TTS 포함)
--preview   G2P 변환 결과만 미리보기 (모델 로드 없음)
--json      JSON 형태로 결과 추가 출력
--model     모델 ID 직접 지정 (기본: w11wo/wav2vec2-xls-r-300m-korean)
```

---

## 🗺️ 개발 로드맵

### ✅ 1단계 완료 — CTC + Greedy Decoding 베이스라인

- [x] Python 로컬 환경 구축
- [x] 3종 한국어 CTC 모델 비교 테스트
- [x] `w11wo/wav2vec2-xls-r-300m-korean` 베이스라인 채택
- [x] kss G2P 발음 전사 채점 기준 통합 (크로스플랫폼)
- [x] 음절 단위 Diff 시각화 + CER 기반 점수
- [x] 오디오 전처리: 무음 제거 + 노이즈 제거 + RMS 정규화
- [x] 자모 단위 피드백: "ㅊ(파찰음) → ㅌ(파열음)" 오류 원인 설명
- [x] INT8 동적 양자화: 로드 시간 ~50% 단축
- [x] TTS 재생 기능 (gTTS → pyttsx3 fallback)
- [x] 연습 모드: 문장 입력 → TTS 듣기 → 녹음 → 분석 → 재도전
- [x] 자동 테스트 모드 (`--test`): 마이크 없이 gTTS로 파이프라인 검증
- [x] 배치 연습 모드 (TTS 포함)
- [ ] 실제 사람 목소리 마이크 테스트 (환경 확보 후 진행)

### 🔜 2단계 예정 — Phonetic Transcription 파인튜닝

- [ ] 연습 문장 세트 정의
- [ ] kss G2P 자동 전사로 표준 발음 라벨 생성 (70%)
- [ ] AI Hub 구음장애 음성 데이터셋 활용 (30%)
- [ ] `wav2vec2-xls-r-300m-korean` 발음 전사 라벨로 파인튜닝
- [ ] ONNX → CoreML 변환 → iOS 탑재

---

## 🔍 알려진 한계 (1단계)

| 한계 | 원인 | 해결 방향 |
|---|---|---|
| `안녕하세요 → 안유아세요` | 모델 음절 인식 불안정 | 2단계 파인튜닝 |
| `오늘 → 모늘` | 초성 혼동 | 2단계 파인튜닝 |
| `싶어요 → 싶어요` (맞춤법 출력) | 모델 vocabulary 편향 | 2단계 발음 전사 라벨 학습 |
| TTS 테스트 한계 | gTTS ≠ 실제 사람 목소리 | 실제 음성 데이터 수집 |

---

## 📦 의존성

```
torch>=2.0.0
transformers>=4.35.0
librosa>=0.10.0
jiwer>=3.0.0
sounddevice>=0.4.6
scipy>=1.11.0
numpy>=1.24.0
kss>=6.0.0
gtts>=2.3.0
pyttsx3>=2.90
noisereduce>=2.0.0   # 선택 — 설치 시 노이즈 제거 자동 적용
```

---

## 🔗 참고

- 모델: [w11wo/wav2vec2-xls-r-300m-korean](https://huggingface.co/w11wo/wav2vec2-xls-r-300m-korean)
- G2P: [kss](https://github.com/hyunwoongko/kss)
- AI Hub 구음장애 데이터: [aihub.or.kr](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=608)
- 원본 앱: [AppleDeveloperAcademy-MC3/Alright](https://github.com/AppleDeveloperAcademy-MC3/Alright)