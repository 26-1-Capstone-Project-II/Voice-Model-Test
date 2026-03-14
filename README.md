# 🔊 온보이스 (On-Voice) — 발음 교정 모듈

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
└── README.md                 # 이 파일
```

---

## 🚀 빠른 시작

### 1. 설치

```powershell
pip install torch transformers librosa jiwer sounddevice scipy g2pk pyttsx3
```

> **Apple Silicon (M1/M2) Mac**
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
> pip install transformers librosa jiwer sounddevice scipy g2pk pyttsx3
> ```

### 2. ★ 연습 모드 (핵심 기능)

문장을 입력하면 TTS로 먼저 들려주고, 따라 말하면 발음을 분석해준다.

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

  r=다시듣기  Enter=녹음시작 >        ← r 로 반복 청취 가능
  🔴 녹음 중... 5초
  ✅ 녹음 완료!

  📊 발음 분석 결과 ...

  다시 연습할까요?  Enter=재도전  q=종료 >
```

### 3. 배치 연습 모드 (TTS 포함)

```powershell
python pronunciation_scorer.py --batch
```

### 4. 마이크로 단일 분석

```powershell
python pronunciation_scorer.py --text "같이 해볼까" --mic
```

### 5. 오디오 파일로 분석

```powershell
python pronunciation_scorer.py --text "같이 해볼까" --audio my_voice.wav
```

### 6. G2P 변환 미리보기 (모델 로드 없이)

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
    ├─ [G2P 변환] g2pk (descriptive=True)
    │       └─→ 발음 전사 ("가치 해볼까")  ← 채점 기준
    │
    ├─ [TTS 재생] gTTS → pyttsx3 fallback
    │       └─→ 사용자가 목표 발음을 귀로 확인
    │             r 입력 시 반복 재생 가능
    │
    └─ [음성 입력] 마이크 or 파일
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
    인식 결과 ("가치 해볼까")
            │
            ▼
    [음절 단위 Diff]
    SequenceMatcher → equal / replace / delete / insert
            │
            ▼
    점수 (CER 기반 0~100점) + 오류 위치 시각화
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

#### G2P 채점 기준 (descriptive=True)

```python
from g2pk import G2p
g2p = G2p()

# descriptive=False (기본): 완전 음운 변환 → 과잉 변환 발생
g2p("않아요")              # → "아나요"  ❌ 과잉 변환
g2p("같이 해볼까")         # → "가치 해볼까"  ✅

# descriptive=True: 실제 화자 발음 기준 → 안정적
g2p("않아요", descriptive=True)          # → "않아요"   ✅ 보존
g2p("같이 해볼까", descriptive=True)     # → "가치 해볼까"  ✅ 연음 변환
g2p("좋네요", descriptive=True)          # → "존네요"   ✅ 비음화
```

`descriptive=True` 를 선택한 이유: 모델이 "않아요"를 그대로 잘 인식하는데,  
`descriptive=False` 로 정답을 "아나요"로 바꾸면 오히려 점수가 낮아지는 역효과 발생.

#### TTS 재생 (gTTS → pyttsx3 fallback)

```python
def play_tts(text):
    if _play_gtts(text):    # gTTS: 인터넷 연결 시 자연스러운 음성
        return
    _play_pyttsx3(text)     # pyttsx3: 오프라인 fallback (Windows SAPI)
```

청각장애인 사용자가 목표 발음을 보청기/인공와우로 먼저 확인한 뒤  
따라 말할 수 있도록 하는 핵심 UX 흐름이다.

---

## 📊 모델 선정 과정

3종 모델을 동일 조건(gTTS 5문장)으로 비교 테스트했다.

| 모델 | 평균 점수 | 특징 |
|---|---|---|
| **w11wo/wav2vec2-xls-r-300m-korean** ✅ | **69.0점** | "같이→가치" 발음 전사 근접, 긴 문장 100점 |
| kresnik/wav2vec2-large-xlsr-korean | 69.3점 | 짧은 문장 안정적, 발음 전사 부정확 |
| Kkonjeong/wav2vec2-base-korean | 탈락 | 자모 분리 출력 (`ㄱㅏㅇㅣ...`) → 음절 CER 계산 불가 |

**채택 모델**: `w11wo/wav2vec2-xls-r-300m-korean`
- 크기: ~1.3GB
- 발음 전사 친화적 인식 (`같이 → 가치`)
- 2단계 파인튜닝 베이스로 적합한 크기

---

## 📈 1단계 성능 결과

G2P 발음 전사 기준 채점으로 전환 후 성과:

| 문장 | 맞춤법 기준 | 발음 기준 | 변화 |
|---|---|---|---|
| 같이 해볼까 | 60점 | **100점** | +40 ✅ |
| 저는 잘 들리지 않아요 | 100점 | **78점** | -22 (descriptive=True 로 안정화) |
| 천천히 말해주세요 | 75점 | **75점** | ±0 |
| 오늘 날씨가 좋네요 | 50점 | **50점** | ±0 (TTS 음질 한계) |

> **TTS 한계**: gTTS가 경음화/연음을 완벽히 재현하지 못해 일부 점수가 낮게 측정됨.  
> 실제 사람 목소리 기준으로는 더 높은 점수 예상.

---

## ⚙️ CLI 옵션 전체

```
--text      목표 문장
--audio     오디오 파일 경로 (.wav / .mp3 / .m4a)
--mic       마이크 실시간 녹음
--practice  ★ 연습 모드: TTS 재생 → 따라 말하기 → 분석 → 재도전
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
- [x] G2P (g2pk, descriptive=True) 발음 전사 채점 기준 통합
- [x] 음절 단위 Diff 시각화 + CER 기반 점수
- [x] 마이크 / 파일 입력 둘 다 지원
- [x] TTS 재생 기능 (gTTS → pyttsx3 fallback)
- [x] 연습 모드: 문장 입력 → TTS 듣기 → 녹음 → 분석 → 재도전
- [x] 배치 연습 모드 (TTS 포함)
- [ ] 실제 사람 목소리 마이크 테스트 (환경 확보 후 진행)

### 🔜 2단계 예정 — Phonetic Transcription 파인튜닝

- [ ] 연습 문장 세트 정의
- [ ] G2P 자동 전사로 표준 발음 라벨 생성
- [ ] AI Hub 구음장애 음성 데이터셋 활용
- [ ] `wav2vec2-xls-r-300m-korean` 발음 전사 라벨로 파인튜닝
- [ ] ONNX → CoreML 변환 → iOS 탑재

---

## 🔍 알려진 한계 (1단계)

| 한계 | 원인 | 2단계 해결 방향 |
|---|---|---|
| `안녕하세요 → 안유아세요` | 모델 음절 인식 불안정 | 파인튜닝으로 정밀도 향상 |
| `오늘 → 모늘` | 초성 혼동 | 구음장애 데이터 혼합 학습 |
| TTS 테스트 한계 | gTTS ≠ 실제 사람 목소리 | 실제 음성 데이터 수집 |
| 음소 레벨 피드백 없음 | 음절 단위만 지원 | 2단계 자모 분리 분석 추가 예정 |

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
g2pk>=0.9.4
gtts>=2.3.0        # TTS 1순위 (인터넷 필요)
pyttsx3>=2.90      # TTS fallback (오프라인)
```

---

## 🔗 참고

- 모델: [w11wo/wav2vec2-xls-r-300m-korean](https://huggingface.co/w11wo/wav2vec2-xls-r-300m-korean)
- G2P: [g2pk](https://github.com/Kyubyong/g2pK)
- AI Hub 구음장애 데이터: [aihub.or.kr](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=608)
- 원본 앱: [AppleDeveloperAcademy-MC3/Alright](https://github.com/AppleDeveloperAcademy-MC3/Alright)


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
└── README.md                 # 이 파일
```

---

## 🚀 빠른 시작

### 1. 설치

```powershell
pip install torch transformers librosa jiwer sounddevice scipy g2pk
```

> **Apple Silicon (M1/M2) Mac**
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
> pip install transformers librosa jiwer sounddevice scipy g2pk
> ```

### 2. G2P 변환 미리보기 (모델 로드 없이)

```powershell
python pronunciation_scorer.py --text "같이 해볼까" --preview
# 맞춤법 : 같이 해볼까
# 발음   : 가치 해볼까  ← G2P 변환됨
```

### 3. 마이크로 발음 분석

```powershell
python pronunciation_scorer.py --text "같이 해볼까" --mic
```

### 4. 오디오 파일로 분석

```powershell
python pronunciation_scorer.py --text "같이 해볼까" --audio my_voice.wav
```

### 5. 배치 연습 모드

```powershell
python pronunciation_scorer.py --batch
```

---

## 🧠 기술 구조

### 전체 흐름

```
입력 문장 ("같이 해볼까")
    │
    ├─ [G2P 변환] g2pk (descriptive=True)
    │       └─→ 발음 전사 ("가치 해볼까")  ← 채점 기준
    │
    └─ [음성 입력] 마이크 or 파일
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
    인식 결과 ("가치 해볼까")
            │
            ▼
    [음절 단위 Diff]
    SequenceMatcher → equal / replace / delete / insert
            │
            ▼
    점수 (CER 기반 0~100점) + 오류 위치 시각화
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

#### G2P 채점 기준 (descriptive=True)

```python
from g2pk import G2p
g2p = G2p()

# descriptive=False (기본): 완전 음운 변환 → 과잉 변환 발생
g2p("않아요")              # → "아나요"  ❌ 과잉 변환
g2p("같이 해볼까")         # → "가치 해볼까"  ✅

# descriptive=True: 실제 화자 발음 기준 → 안정적
g2p("않아요", descriptive=True)          # → "않아요"   ✅ 보존
g2p("같이 해볼까", descriptive=True)     # → "가치 해볼까"  ✅ 연음 변환
g2p("좋네요", descriptive=True)          # → "존네요"   ✅ 비음화
```

`descriptive=True` 를 선택한 이유: 모델이 "않아요"를 그대로 잘 인식하는데,  
`descriptive=False` 로 정답을 "아나요"로 바꾸면 오히려 점수가 낮아지는 역효과 발생.

---

## 📊 모델 선정 과정

3종 모델을 동일 조건(gTTS 5문장)으로 비교 테스트했다.

| 모델 | 평균 점수 | 특징 |
|---|---|---|
| **w11wo/wav2vec2-xls-r-300m-korean** ✅ | **69.0점** | "같이→가치" 발음 전사 근접, 긴 문장 100점 |
| kresnik/wav2vec2-large-xlsr-korean | 69.3점 | 짧은 문장 안정적, 발음 전사 부정확 |
| Kkonjeong/wav2vec2-base-korean | 탈락 | 자모 분리 출력 (`ㄱㅏㅇㅣ...`) → 음절 CER 계산 불가 |

**채택 모델**: `w11wo/wav2vec2-xls-r-300m-korean`
- 크기: ~1.3GB
- 발음 전사 친화적 인식 (`같이 → 가치`)
- 2단계 파인튜닝 베이스로 적합한 크기

---

## 📈 1단계 성능 결과

G2P 발음 전사 기준 채점으로 전환 후 성과:

| 문장 | 맞춤법 기준 | 발음 기준 | 변화 |
|---|---|---|---|
| 같이 해볼까 | 60점 | **100점** | +40 ✅ |
| 저는 잘 들리지 않아요 | 100점 | **78점** | -22 (descriptive=True 로 안정화) |
| 천천히 말해주세요 | 75점 | **75점** | ±0 |
| 오늘 날씨가 좋네요 | 50점 | **50점** | ±0 (TTS 음질 한계) |

> **TTS 한계**: gTTS가 경음화/연음을 완벽히 재현하지 못해 일부 점수가 낮게 측정됨.  
> 실제 사람 목소리 기준으로는 더 높은 점수 예상.

---

## ⚙️ CLI 옵션 전체

```
--text      목표 문장 (정답)
--audio     오디오 파일 경로 (.wav / .mp3 / .m4a)
--mic       마이크 실시간 녹음
--duration  녹음 시간 초 (기본: 5)
--save      녹음 파일 저장 경로 (예: out.wav)
--batch     배치 연습 모드 (7문장 연속)
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
- [x] G2P (g2pk, descriptive=True) 발음 전사 채점 기준 통합
- [x] 음절 단위 Diff 시각화 + CER 기반 점수
- [x] 마이크 / 파일 입력 둘 다 지원
- [x] 배치 연습 모드
- [ ] 실제 사람 목소리 마이크 테스트 (환경 확보 후 진행)

### 🔜 2단계 예정 — Phonetic Transcription 파인튜닝

- [ ] 연습 문장 세트 정의
- [ ] G2P 자동 전사로 표준 발음 라벨 생성
- [ ] AI Hub 구음장애 음성 데이터셋 활용
- [ ] `wav2vec2-xls-r-300m-korean` 발음 전사 라벨로 파인튜닝
- [ ] ONNX → CoreML 변환 → iOS 탑재

---

## 🔍 알려진 한계 (1단계)

| 한계 | 원인 | 2단계 해결 방향 |
|---|---|---|
| `안녕하세요 → 안유아세요` | 모델 음절 인식 불안정 | 파인튜닝으로 정밀도 향상 |
| `오늘 → 모늘` | 초성 혼동 | 구음장애 데이터 혼합 학습 |
| TTS 테스트 한계 | gTTS ≠ 실제 사람 목소리 | 실제 음성 데이터 수집 |
| 음소 레벨 피드백 없음 | 음절 단위만 지원 | 2단계 자모 분리 분석 추가 예정 |

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
g2pk>=0.9.4
```

---

## 🔗 참고

- 모델: [w11wo/wav2vec2-xls-r-300m-korean](https://huggingface.co/w11wo/wav2vec2-xls-r-300m-korean)
- G2P: [g2pk](https://github.com/Kyubyong/g2pK)
- AI Hub 구음장애 데이터: [aihub.or.kr](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=608)
- 원본 앱: [AppleDeveloperAcademy-MC3/Alright](https://github.com/AppleDeveloperAcademy-MC3/Alright)
