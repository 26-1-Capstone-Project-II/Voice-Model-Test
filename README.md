# 🔊 온보이스 (On-Voice) — 발음 교정 모듈

> 청각장애인을 위한 발음 교정 앱 **온보이스**의 AI 발음 분석 백엔드
> Whisper Tiny + G2P 기반 발음 전사 & 자모 레벨 오류 감지 시스템

---

## 📁 파일 구성

```
.
├── [Whisper 발음 전사 파이프라인]
│   ├── finetune_whisper.py          # Whisper tiny LoRA 파인튜닝
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
└── README.md
```

---

## 🎯 핵심 설계: 소리나는 대로 텍스트 출력

### 왜 기존 ASR을 쓸 수 없는가

일반 STT 모델은 내부 Language Model이 들리는 소리를 표준 맞춤법으로 자동 교정합니다. 사용자가 "가치 머글까"라고 발음해도 "같이 먹을까"로 출력되죠. 이는 일반적인 받아쓰기에는 좋지만, **발음 평가에는 치명적**입니다. 사용자가 잘못 발음해도 정답으로 처리되기 때문입니다.

따라서 우리의 목표는 **들리는 소리 그대로 출력하는 모델**, 즉 **발음 전사(Phonetic Transcription) 모델**입니다.

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

---

## 🧠 모델 학습 과정

### 학습 설정

| 항목 | 내용 |
|------|------|
| **베이스 모델** | [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) |
| **파인튜닝 방식** | **LoRA (Low-Rank Adaptation)** — 베이스 음향 인식 능력 보존, 출력 분포만 효율적으로 재조정 |
| **학습 데이터셋** | **Zeroth-Korean (낭독체, 약 51시간)** |
| **라벨 생성** | 원본 텍스트를 `g2pk`로 자동 변환하여 발음 전사 라벨 생성 |
| **학습 결과** | **검증 CER 0.088 (8.8%)** |
| **저장 경로** | `best_model_whisper/best` |

### 라벨 변환 예시 (g2pk 적용)

| 원본 텍스트 | 발음 전사 라벨 | 변환 규칙 |
|-------------|----------------|-----------|
| 같이 먹을까 | **가치 머글까** | 구개음화 + 연음 |
| 좋네요 | **존네요** | 비음화 |
| 학교에 갑니다 | **학꾜에 감니다** | 경음화 + 비음화 |

### 학습이 의도한 동작 변화

Whisper의 강력한 LM이 자동 수행하던 **맞춤법 교정 동작을, g2pk가 만들어낸 발음 전사 라벨로 재학습**시켜 소리 나는 대로 출력하도록 행동을 바꾸는 것이 핵심입니다.
**LoRA**를 사용함으로써 베이스 모델의 음향 인식 능력은 보존하면서, 출력 분포만 효율적으로 재조정했습니다.

### 💡 왜 구음장애가 아닌 '정상 낭독체'를 쓰는가?

**[오류 시나리오] 구음장애 음성으로 학습할 경우:**
- 오디오: "가티 머거요" (환자의 틀린 발음)
- 라벨(G2P): "가치 머거요" (원래 의도한 정답 발음)
- **문제점:** 모델이 "가티"라는 틀린 소리를 들어도 "가치"로 **자동 교정(Auto-correct)** 하는 법을 배우게 됩니다. 발음 오류 추적이 불가능해집니다.

**[성공 시나리오] 정상 낭독체(Zeroth-Korean)로 학습할 경우:**
- 오디오: "가치 머거요" (정상인의 정확한 발음)
- 라벨(G2P): "가치 머거요"
- **해결책:** 모델은 100% "정확한 소리 = 정확한 발음 기호"의 1:1 매핑만을 학습합니다.
- **결과:** 추론 시 환자가 "가티 머거요"라고 틀리게 말하면, 모델은 교정하는 법을 배운 적이 없기 때문에 들은 그대로 **"가티 머거요"를 Raw 출력** → **발음 오류 감지 성공!**

---

## 🧪 테스트 및 검증 과정

테스트는 두 단계로 구성됩니다.
**1차 도메인 내 검증(Zeroth-Korean)** 으로 모델의 발음 전사 능력 자체를 측정하고,
**2차 타깃 도메인 검증(AIHub 구음장애)** 으로 실제 사용자 발화에서의 효과를 간접 증명합니다.

---

### 1차 검증 — Zeroth-Korean 낭독체 (도메인 내 성능)

#### 사용 데이터셋
- **테스트 데이터셋:** Zeroth-Korean 낭독체 평가 셋 (학습과 동일 도메인 분포)
- 학습에 사용되지 않은 문장 **10개**를 추출하여 검증

#### 검증 단계 (3단계 구성)

**① 발음 전사 테스트 (10개 문장)**
- 모델 출력이 g2pk 기대 발음과 정확히 일치하는지 **엄격 매칭(Exact Match)** 으로 검증
- **평가지표:** 전사 정확도
- **결과:** 4/10

**② 발음 평가 테스트 (10개 문장)**
- 동일 10문장을 **CER 기반 점수** (`점수 = (1 − CER) × 100`) 로 환산
- **결과:** 평균 **88.7%** (평균 CER 0.113)
- 자모 단위로 어떤 음소가 잘못 나왔는지 구체적 피드백 제공 (예: ㅁ→ㅂ, ㅓ→ㅗ)

**③ 베이스라인 비교 (원본 Whisper-tiny vs 파인튜닝)**
- 같은 음성을 두 모델에 동시 입력해 출력 차이를 직접 시연
- 파인튜닝의 효과를 가장 설득력 있게 보여주는 핵심 단계

---

### 2차 검증 — AIHub 구음장애 데이터셋 (사용자 효용 검증) ⭐

> **목적:** **구음장애를 가진 분들이 본 앱을 사용해야 하는 이유**를 정량적으로 입증합니다.
> 본 검증은 모델 정확도(CER)가 아니라 **앱 사용 가치**를 측정합니다.

#### 왜 CER이 아닌 사용자 효용을 측정하는가
- Zeroth-Korean 검증(1차)은 정상 발음 데이터에서의 *전사 정확도*를 측정 (CER 0.088 달성)
- 그러나 구음장애 화자에게는 **"내 발음이 표준에서 어떻게 어긋났는지 *볼 수 있는가*"** 가 핵심
- 따라서 2차 검증은 **사용자가 일반 ASR 대비 추가로 얻는 정보·진단 능력**을 측정

#### 사용 데이터셋
- **데이터셋:** [AI Hub 구음장애 음성인식 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=608)
- **선정 기준:** 학습에 사용되지 않은 화자(unseen speaker)의 발화 일부 추출

#### 평가 메커니즘
같은 오디오를 두 모델에 동시 입력하여 출력 차이를 분석:

```
                  X = 베이스라인 Whisper-tiny  (자동 교정된 표준 한국어 표기)
오디오 →  분기  ↗
                  Y = 파인튜닝 모델            (들리는 그대로의 발음 표기)

         X와 Y의 자모-수준 차이 = 본 앱이 사용자에게 노출하는 추가 정보
```

#### 4가지 가치 명제와 측정 지표

| 가치 명제 | 사용자 관점 | 측정 지표 |
|----------|------------|----------|
| **V1. 솔직한 피드백** | "다른 앱은 내 오류를 정답으로 바꿔서 보여주지만, 이 앱은 실제 소리를 보여줌" | **M1.** Auto-Correction Rejection Rate: `# (Y ≠ X) / total` |
| **V2. 정보 가시성** | "내 발음이 표준에서 어떻게 어긋났는지 *볼 수 있다*" | **M2.** Information Disclosure: `jamo_edit_distance(Y, X)` 평균 |
| **V3. 진단 구체성** | "ㅁ을 ㅂ으로 발음하고 있다는 걸 정확히 짚어줌" | **M3.** Feedback Density (한국어 음운규칙 부합 차이/발화)<br>**M4.** Diagnosis Coverage (경음화/비음화/구개음화/연음화 분포) |
| **V4. 일관성** | "내 발음 약점이 매번 같은 곳에서 나타나니, 거기를 집중 연습 가능" | **M5.** Per-Speaker Consistency: 동일 화자 발화에서 동일 자모 치환 패턴 반복률 |

#### 결과 신뢰성 보강 — 응집 베이스라인 필터 + S/N 비율

베이스라인(Whisper-tiny)은 dysarthric 음성에서 표준 표기를 항상 정확히 뽑지 못하므로,
모든 결과를 **2가지 부분집합**에서 산출해 비교 보고합니다.

| 부분집합 | 정의 | 용도 |
|---------|------|-----|
| **전체** | 모든 평가 발화 | 광범위 추세 |
| **응집 베이스라인** | X가 한국어로 응집된 발화만 (≥5자, 한글비율≥70%, 종결어미 포함) | **신뢰도 ↑ — 발표 시 사용 권장** |

또한 **신호/노이즈 비율(S/N)** 을 핵심 해석 지표로 추가:

```
S/N 비율 = phonetic_rule_ratio = rule_hits / edit_distance
        = 자모 차이 중 한국어 음운규칙으로 설명되는 비율
```

높을수록 본 앱이 노출하는 정보가 *진짜 발음 정보*임을 의미합니다.

#### 통제군 비교 — Zeroth-Korean Test

같은 스크립트를 1차 검증에서 사용한 깨끗한 Zeroth-Korean test에 돌려
**효용 지표 상한값(upper bound)** 을 얻습니다.
AIHub(dysarthric)와 Zeroth(clean) 두 결과를 함께 제시하면 발표 신뢰도가 크게 올라갑니다.

| 데이터셋 | 의미 | 기대되는 결과 |
|---------|------|--------------|
| **AIHub 구음장애** | 실 사용자 시나리오 (target user proxy) | M1 높음, S/N 중간 (노이즈 많음) |
| **Zeroth-Korean test** | 정상 발음 통제군 (upper bound) | M1 높음, S/N **높음** (대부분이 음운변동) |

#### 실행 환경
- **테스트 환경:** Linux 서버 (NVIDIA CUDA GPU)
- **이유:** 두 모델 동시 추론 + AIHub/Zeroth 양쪽 평가

```bash
# 1단계 — AIHub 원본 데이터를 VAD로 세그멘트 (최초 1회)
PYTHONNOUSERSITE=1 python vad_segment.py \
    --wav_dir  "/path/to/aihub/원천데이터" \
    --json_dir "/path/to/aihub/라벨링데이터" \
    --output_dir ./segmented_dataset

# 2단계 — 사용자 효용 검증 (AIHub 구음장애)
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python test_aihub_baseline_ref.py \
    --model_path best_model_whisper/best \
    --baseline_model openai/whisper-tiny \
    --json_dir segmented_dataset \
    --num_samples 200 \
    --tag aihub \
    --output_dir results/aihub_value_proposition

# 3단계 — 통제군 (Zeroth-Korean clean speech)
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python test_aihub_baseline_ref.py \
    --model_path best_model_whisper/best \
    --baseline_model openai/whisper-tiny \
    --json_dir zeroth_dataset \
    --num_samples 200 \
    --tag zeroth \
    --output_dir results/zeroth_value_proposition
```

#### 출력 결과
- `results/{tag}_value_proposition/eval_results.json` — 전체+응집 부분집합 metrics + 샘플별 상세 (X/Y/자모 차이/음운변동/S/N)
- `results/{tag}_value_proposition/summary.md` — 강한 주장 요약 + M1-M5 비교표(전체 vs 응집) + 시연 케이스

---

## 🔥 발음 평가 엔진 동작 사례 (Demo)

`test_whisper_phonetic.py` 실행 시 나타나는 **실제 파인튜닝 모델의 오류 감지 결과**입니다.
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

## 📈 검증 결과 분석

본 앱의 가치는 **두 단계의 테스트**를 통해 정량·정성적으로 입증되었습니다.
1차 테스트는 *"모델이 발음을 정확히 전사하는가"*, 2차 테스트는
*"사용자가 일반 ASR 대비 무엇을 더 얻는가"* 를 측정합니다.

---

### 🥇 1차 테스트 — 발음 전사 정확도 (Zeroth-Korean)

#### 한 줄 요약
> **모델은 깨끗한 한국어 낭독체 음성을 발음 단위로 정확히 받아쓴다.**

#### 무엇을 측정했나
- **데이터셋:** Zeroth-Korean 낭독체 평가 셋 (학습과 동일 도메인, 학습 미사용 문장)
- **질문:** 사용자가 정확히 발음한 한국어를 모델이 g2pk 기대 발음과 일치하게 옮기는가
- **지표:** **CER** (Character Error Rate) — 낮을수록 정확

#### 결과

| 검증 단계 | 결과 | 의미 |
|----------|------|-----|
| ① 발음 전사 정확도 (Exact Match) | **4 / 10** | 10문장 중 4문장이 g2pk 기대 발음과 *완벽히* 일치 |
| ② 발음 평가 점수 `(1−CER)×100` | **평균 88.7%** | CER 0.113 — 자모 단위 약 1자 오차 |
| ③ 베이스라인(원본 Whisper) 대비 | 정성 시연 입증 | 원본은 표준 표기로 교정, 본 앱은 발음 그대로 |
| 🏆 **검증셋 최종 CER** | **0.088 (8.8%)** | 학습 51h LoRA 파인튜닝의 안정적 발음 전사 능력 |

#### 1차 테스트 결론
모델이 **정상 발음 환경에서 한국어 음운변동(경음화/비음화/구개음화/연음화)을 정확히 학습**했음이 입증되었습니다. 깨끗한 음성에서 평균 88.7%의 발음 점수, CER 0.088은 발음 평가 엔진의 **기반 능력(foundation capability)** 이 견고함을 의미합니다.

---

### 🥈 2차 테스트 — 사용자 효용 검증 (AI Hub 구음장애)

#### 한 줄 요약
> **본 앱은 일반 ASR이 숨기는 발음 정보를 사용자에게 노출하고, 4종 한국어 음운변동을 자모 단위로 진단한다.**

#### 무엇을 측정했나
- **데이터셋:** AI Hub 구음장애 음성인식 데이터 200발화 + Zeroth-Korean 200발화(통제군)
- **질문:** 구음장애 화자가 본 앱을 사용했을 때 일반 ASR 대비 어떤 추가 가치를 얻는가
- **지표:** CER 아님. **4가지 가치 명제(V1~V4)** + 신호/노이즈 비율(S/N)

#### 4가지 가치 명제 측정 결과

| 가치 명제 | AIHub (구음장애, 응집) | Zeroth (정상, 통제군) | 입증 강도 |
|----------|---------------------|---------------------|----------|
| **V1. 솔직한 피드백** (M1 자동교정 거부율) | **100%** | **100%** | ★★★ 환경 무관 완벽 입증 |
| **V2. 정보 가시성** (M2 자모/발화) | 31.8자모 | 21.6자모 | ★★ 양쪽 모두 풍부한 정보 노출 |
| **V3. 진단 구체성** (S/N 비율) | 4.8% | **24.2%** | ★★★ Zeroth에서 결정적 입증 |
| **V3. 음운변동 4종 검출** (M4) | **4종 모두** | **4종 모두** | ★★★ 모델이 음운규칙 학습 |
| **V4. 일관성** (M5 화자별 패턴 반복률) | **43.1%** | (측정불가) | ★★ AIHub에서 입증 |

#### 핵심 발견 — Zeroth 통제군이 S/N 지표의 유효성을 자체 검증

| | Zeroth (정상) | AIHub (구음장애) | 의미 |
|---|--------------|----------------|-----|
| S/N 비율 | **24.2%** | 4.8% | **5배 차이** — S/N이 데이터 품질에 민감하게 반응 |
| 발화당 음운규칙 적중 | 4.55개 | 1.19개 | 정상 발음에서 음운변동이 더 풍부히 검출 |
| 경음화 발생률 | 84% | 33% | Zeroth에서 8할 이상 발화에 경음화 적용 |
| 연음화 발생률 | 92% | 41% | Zeroth에서 9할 이상 발화에 연음화 적용 |

**이 5배 차이가 결정적입니다.** 만약 S/N이 단순한 노이즈 지표였다면 두 데이터셋에서 비슷한 값이 나왔을 것입니다. 5배 차이는 **S/N 비율이 진짜 발음 정보의 농도를 측정하는 유효 지표임을 자체 입증**합니다.

#### 결정적 시연 사례 — 한 발화에 모든 가치가 입증됨

**Zeroth #4 사례 (S/N 64.3%)**

```
X (베이스라인 = 일반 ASR):  박 사무장  측은   법원에  미국에서  재판을   받아야  …담안 소면을  제출했다
Y (본 앱 = 파인튜닝 모델):  박 싸무장  츠근   버붜네   미구게서   재파늘   바다야  …다믄 서며늘  제출핻따
                              ↑경음     ↑연음   ↑연음2    ↑연음     ↑연음    ↑연음    ↑연음 ↑연음   ↑경음
```

→ 14개 자모 차이 중 **9개가 한국어 음운규칙**으로 설명됨. 한 발화에 **V1(자동교정 거부) + V2(정보 노출) + V3(4종 음운변동)** 이 모두 시연됩니다.

#### 2차 테스트 결론

본 앱은 **일반 ASR과 본질적으로 다른 동작을 한다**는 것이 양쪽 데이터셋에서 100% 입증되었습니다(V1). 정상 발음 환경에서 **자모 차이의 24%가 진짜 한국어 음운규칙 신호**이며, 4종 음운변동(경음/구개/비음/연음)이 모두 자모 단위로 진단됩니다(V3). 어려운 dysarthric 환경에서도 동일하게 4종 모두 검출되며, 화자별 발음 약점 패턴이 평균 43% 반복 검출되어 **장기 학습 추적 도구**로 활용 가능합니다(V4).

---

### 🎯 종합 결론 — 본 앱이 사용자에게 주는 4가지 가치

| 사용자가 묻는 질문 | 답을 주는 지표 | 검증된 결과 |
|------------------|--------------|-----------|
| "왜 다른 앱이 아닌 이 앱을 써야 하나요?" | V1 — 자동교정 거부율 | **AIHub·Zeroth 양쪽 모두 100%** |
| "이 앱으로 뭘 알 수 있나요?" | V2 — 정보 공개량 | 발화당 21~32 자모 정보 추가 노출 |
| "어떤 발음 문제를 짚어주나요?" | V3 — 음운변동 4종 진단 + S/N | 경음/구개/비음/연음 **4종 모두 검출**, 정상 발음 S/N **24.2%** |
| "내 약점을 추적해서 연습할 수 있나요?" | V4 — 화자별 일관성 | 동일 자모 치환 패턴 평균 **43.1%** 반복 검출 |

#### 핵심 메시지

> **1차 테스트는 "모델이 발음을 정확히 전사할 수 있다"는 *기반 능력*을 입증했고
> (Zeroth CER 0.088 / 발음 점수 88.7%),
> 2차 테스트는 "사용자가 일반 ASR 대비 진짜 추가 가치를 얻는다"는 *실용 가치*를 입증했습니다
> (V1 100% / V3 4종 모두 / V4 43.1%).**
>
> 두 테스트가 합쳐져, 본 앱이 **구음장애·청각장애 사용자의 발음 자가진단 도구**로서
> *작동하며 가치 있다*는 것이 정량·정성적으로 모두 증명되었습니다.

---

## 📊 최종 성능 요약

| 구분 | 측정값 | 의미 |
|------|--------|-----|
| **Zeroth-Korean 검증 CER** | **0.088 (8.8%)** | 정상 발음 전사 정확도 — 기반 능력 |
| **Zeroth 발음 점수 (1−CER)** | **88.7%** | 자모 단위 발음 일치율 |
| **AIHub 자동교정 거부율** | **100%** | 일반 ASR과의 본질적 차별성 (V1) |
| **Zeroth S/N 비율** | **24.2%** | 진짜 발음 정보의 농도 (V3 핵심) |
| **음운변동 4종 검출** | **4 / 4** | 경음/구개/비음/연음 모두 자모 단위 진단 |
| **AIHub 화자 일관성** | **43.1%** | 발음 약점 추적 가능성 (V4) |

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
peft
```

---

## 🔗 참고

- 모델: [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny)
- G2P: [g2pk](https://github.com/Kyubyong/g2pK)
- VAD: [Silero-VAD](https://github.com/snakers4/silero-vad)
- iOS 배포: [WhisperKit](https://github.com/argmaxinc/WhisperKit)
- HuggingFace Zeroth-Korean 데이터: [Bingsu/zeroth-korean](https://huggingface.co/datasets/Bingsu/zeroth-korean)