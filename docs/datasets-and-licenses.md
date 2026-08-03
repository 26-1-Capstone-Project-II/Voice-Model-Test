# 사용 데이터셋 명시 및 상업적 이용 가능성 검토

작성일: 2026-07-31
대상: 현재 배포 모델 v3 (`best_model_vocab/best` → `Whisper_CoreML_Model`, 커밋 `ba75ec91`)

> 이 문서는 **코드에 실제로 배선된 소스**를 기준으로 작성했다(README 서술이 아니라 코드가 진실).
> 법률 자문이 아니며, 상업화 전 각 제공기관 약관 원문 확인 및 필요한 경우 법무 검토가 필요하다.

---

## 1. 사용 자산 전체 목록

### 1-1. 기반 모델

| 자산 | 출처 | 코드 근거 | 라이선스 | 상업적 이용 |
|---|---|---|---|---|
| Whisper base (사전학습 가중치) | `openai/whisper-base` | [run_server_pipeline.sh:93](run_server_pipeline.sh#L93), [run_vocab_pipeline.sh:204](run_vocab_pipeline.sh#L204) | MIT | ✅ 가능 (저작권 고지 포함) |
| WhisperKit (iOS 추론 런타임) | argmaxinc/WhisperKit | [convert_to_coreml.sh](convert_to_coreml.sh) 변환 대상 | MIT (OSS 버전) | ✅ 가능 |

### 1-2. 학습 데이터 (모델 가중치에 직접 반영됨)

| # | 데이터셋 | 역할 | 코드 근거 | 라이선스 | 상업적 이용 |
|---|---|---|---|---|---|
| D1 | **Zeroth-Korean** (`Bingsu/zeroth-korean`, OpenSLR-40, 51.6h) | 주 학습 코퍼스 (train) + val/test 기준 | [prepare_zeroth.py:31](prepare_zeroth.py#L31) | CC BY 4.0 | ✅ 가능 (출처 표기 필수) |
| D2 | **Zeroth-Korean test 스플릿** | 경쟁 화자/babble 증강의 **간섭 화자 풀** (held-out) | [finetune_whisper.py](finetune_whisper.py) → `FarFieldAugmentor(speech_files=...)` | CC BY 4.0 | ✅ 가능 |
| D3 | **KsponSpeech** (AI Hub 한국어 음성) | 실 자유발화 축 — 외래어/구어 어휘 확장 | [prepare_kspon.py](prepare_kspon.py), [run_vocab_pipeline.sh:145](run_vocab_pipeline.sh#L145) | AI Hub 이용정책 | ⚠️ **조건부 가능** — 학습 모델 배포는 허용, 단 원본/재가공 데이터 유출·배포 금지 + 출처 표기 (§2-B1) |
| D4 | **gTTS 합성음** (Google 번역 TTS) | 외래어/저빈도어 타깃 합성셋 | [prepare_loanword_tts.py](prepare_loanword_tts.py), `TTS_BACKEND=gtts` | 명시적 라이선스 없음 | ❌ **불가/고위험 (블로커)** |
| D5 | **facebook/mms-tts-kor 합성음** | 위와 동일 (스크립트 기본값에 포함) | [prepare_loanword_tts.py:71-72](prepare_loanword_tts.py#L71-L72), `TTS_BACKEND` 기본값 `gtts,mms` | **CC BY-NC 4.0** | ❌ **불가 (사용 시 블로커)** |
| D6 | **MUSAN noise** (OpenSLR-17) | 배경 소음 증강 | [augment.py:10](augment.py#L10), `MUSAN_NOISE_DIR` | CC BY 4.0 | ✅ 가능 (출처 표기) |
| D7 | **RIRS_NOISES** (OpenSLR-28) | 잔향(RIR) 증강 | [augment.py:11](augment.py#L11), `RIR_DIR` | Apache 2.0 | ✅ 가능 |
| D8 | **외래어 시드 어휘/문장** | 합성 원문 텍스트 | [loanword_corpus.py](loanword_corpus.py) | 자체 작성 | ✅ 가능 |

> **D1 참고 — HF 미러 2종:** `Bingsu/zeroth-korean` 과 `kresnik/zeroth_korean` 은 **같은 OpenSLR-40 원본**
> (제작: Lucas Jo, Wonkyum Lee)의 서로 다른 업로드다. 라이선스·내용·시간 모두 동일하고 **컬럼 구성만 다르다**:
> Bingsu = `audio`, `text` / kresnik = `id`, `speaker_id`, `chapter_id`, `path`, `audio`, `text`.
> → CLAUDE.md 가 *"Bingsu 에 화자 ID 가 없어서 공식 test 스플릿을 간섭 풀로 쓴다"* 고 적은 제약은
> **kresnik 미러로 바꾸면 사라진다**(화자 단위 배타 분할을 직접 구성 가능). 라이선스 영향은 없다.

> **D4/D5 확인 필요:** 배포 v3 체크포인트 커밋(`7904dfa5`) 메시지에는 *"외래어 TTS(gtts, melo OOD 평가)"* 로 적혀 있어
> **학습에는 gtts만** 들어갔을 가능성이 높다. 그러나 [run_vocab_pipeline.sh:49](run_vocab_pipeline.sh#L49) 기본값은
> `TTS_BACKEND=gtts,mms` 다. **서버의 실제 실행 로그/`loanword_dataset` 메타로 mms 포함 여부를 확정**해야 한다.
> mms가 포함됐다면 NC 라이선스 오염이므로 재학습이 불가피하다.

### 1-3. 평가 전용 데이터 (가중치 미반영, 결과 공개 시 의무 발생)

| # | 데이터셋 | 역할 | 코드 근거 | 상업적 이용 |
|---|---|---|---|---|
| E1 | **AI Hub 구음장애 음성인식 데이터** (dataSetSn=608) | 2차 타깃 도메인 검증 | [test_aihub_baseline_ref.py](test_aihub_baseline_ref.py), [segment_aihub_silence.py](segment_aihub_silence.py) | ⚠️ 비상업 연구 전제 — 상업 제품 마케팅 자료로 성능 수치를 쓰려면 협의 필요 |
| E2 | **MeloTTS Korean** (`myshell-ai/MeloTTS-Korean`) | OOD 평가셋 합성 (학습 미사용) | `TTS_EVAL_BACKEND=melo` | ✅ MIT — 문제 없음 |

### 1-4. 도구/라이브러리

| 자산 | 라이선스 | 비고 |
|---|---|---|
| g2pk (한국어 G2P) | Apache 2.0 | [korean_g2p_nomecab.py:220](korean_g2p_nomecab.py#L220)에서 optional import |
| `korean_g2p_nomecab.py` (폴백 G2P) | 자체 구현 | MeCab 미설치 환경용 |
| transformers / datasets / accelerate | Apache 2.0 | |
| silero-vad | MIT | 세그멘테이션 전처리 |
| Coqui XTTS-v2 | CPML (비상업) | **코드에 옵션으로만 존재** — 실제 미사용. 상업화 시 절대 사용 금지 |

---

## 2. 판정 요약

### ✅ 문제 없는 축 (상업적 이용 가능)
Whisper base(MIT), WhisperKit(MIT), Zeroth-Korean(CC BY 4.0), MUSAN(CC BY 4.0), RIRS_NOISES(Apache 2.0), MeloTTS(MIT).
→ **출처 표기(attribution)만 하면 상업적 배포 가능.** CC BY는 파생물(=모델 가중치) 배포도 허용한다.

### ⚠️ 조건부 가능 — 의무사항 준수 필요

**B1. KsponSpeech (AI Hub)** — *블로커 아님. 단 지켜야 할 선이 명확하다.*

AI Hub 이용정책 페이지와 FAQ의 문구가 서로 다르게 읽히는데, **적용 대상이 다르다**:

| 대상 | 규정 | 근거 |
|---|---|---|
| **데이터 자체**의 판매·거래 | 수행기관과 별도 협의 필요 | 이용정책 (AI데이터 거래소 맥락) |
| **학습된 모델·서비스** | *"자유롭게 배포, 활용하실 수 있습니다"* | FAQ |
| **데이터 원본 파일** | *"외부(국내, 국외)로 유출하실 수 없습니다"* | FAQ |
| **재가공 데이터 배포** | NIA·구축기업 사전 협의 없이는 불가 | FAQ |
| **출처 표기** | 2차적 저작물에도 사업결과임 표기 | 이용정책 |

→ **모델 가중치를 앱에 넣어 유료로 파는 것까지 FAQ 문언상 허용된다.** 금지되는 것은 *데이터*를 내보내는 행위다.

지켜야 할 것:
1. `kspon_dataset/`(정규화된 wav+jsonl)은 **재가공 데이터**다. 공개 GitHub·서버 외부·해외 클라우드에 올리면 안 된다.
   현재 `.gitignore` 에 `kspon_dataset/`·`loanword_dataset/` 항목이 **없다** — 생성 경로가 리포 밖(`$WS`)이라 우연히 안전할 뿐이므로 명시적으로 추가할 것.
2. 앱/리포에 AI Hub 출처 표기.
3. KsponSpeech는 개별 데이터셋 약관이 따로 있을 수 있으므로, 유료화 확정 전 **AI Hub에 1회 서면 질의**로 확인 기록을 남기는 것을 권장(비용 0, 분쟁 시 방어 근거).

### ❌ 실제 블로커 — 1건 (+1 확인 필요)

**B2. gTTS 합성음**
- gTTS는 Google Cloud TTS(상업 라이선스 명확)가 **아니라** translate.google.com 프론트엔드 비공식 엔드포인트를 호출한다
- 해당 출력물의 상업적 이용 권한을 부여하는 약관 자체가 존재하지 않음 → 권리 근거 없음
- gTTS 저장소 자체도 공개/상업 프로젝트 사용에 대한 법적 주의를 이슈로 안내 중

**B3(확인 필요). facebook/mms-tts-kor 합성음**
- CC BY-NC 4.0 = **비상업 전용**. 출력물(합성 오디오)까지 NC가 미치므로 학습 데이터로 쓰면 모델이 오염된다
- v3 학습에 실제로 포함됐는지 서버 로그로 확정 필요

---

## 3. 상업화 경로 (권장 순서)

**핵심: 유일한 필수 작업은 TTS 백엔드 교체 후 재학습이다.** KsponSpeech는 유지해도 된다.

1. **실제 v3 학습 구성 확정** — 서버의 `loanword_dataset` 생성 로그/지문에서 `backend=` 값을 확인해 mms 포함 여부 결론.
2. **TTS 백엔드 교체 후 재학습** (필수) — gtts/mms를 걷어내고 상업 이용 가능한 엔진으로:
   - MeloTTS Korean (MIT) — 현재 OOD 평가에 쓰던 엔진, 즉시 전환 가능
   - Google Cloud TTS / Azure / Amazon Polly 등 유료 API — 약관상 출력물 상업 이용 명시
   - `--backend melo,<유료엔진>` 조합이면 "다중 백엔드 병합" 원칙도 유지된다
   - ⚠️ 단, MeloTTS는 v3에서 **OOD 평가셋**이었다. 학습에 넣으면 평가 맹점이 생기므로
     평가 엔진을 학습에 안 쓴 다른 것(유료 API 중 하나)으로 교체해야 한다
   - KsponSpeech는 그대로 유지 → 자유발화 축을 잃지 않으므로 §9 게이트 재통과 가능성이 높다
3. **데이터 유출 방지 정비** — `.gitignore` 에 `kspon_dataset/`, `loanword_dataset/` 추가.
   AI Hub 원본·재가공 데이터는 서버 밖으로 내보내지 않는다(모델 가중치만 반출).
4. **AI Hub 서면 확인** (권장) — KsponSpeech 개별 약관 기준 상업 배포 가능 여부를 문의하고 답변 보관.
5. **AI Hub 구음장애 평가 결과 취급 결정** — 상업 제품 홍보에 수치를 쓸지, 내부 검증용으로만 남길지 결정.
   공개 시 "한국지능정보사회진흥원 사업결과" 표기 의무.
6. **NOTICE 파일 작성** — 앱/리포에 아래 고지 포함.

### 참고: 무료 앱 배포도 "상업적"인가

- **CC BY-NC (mms-tts-kor)**: NC는 *"주로 상업적 이익이나 금전적 보상을 목적으로 하지 않는"* 이용을 뜻한다.
  광고·인앱결제·구독이 없는 순수 무료 앱은 NC 범위 안이라고 볼 여지가 있으나, App Store라는
  상업 플랫폼 배포라 회색지대다. **나중에 수익화하면 소급해서 위반이 되고 재학습 외엔 정정 불가**이므로
  처음부터 배제하는 것이 옳다.
- **gTTS**: 무료/유료와 무관하게 **애초에 이용 권한을 부여하는 약관이 없다**. 무료 앱이어도 근거가 없다.
- **AI Hub**: 위 표대로 모델 배포는 유·무료 모두 허용.

---

## 4. 배포 시 포함할 출처 고지 (초안)

```
This product includes models trained on the following resources:

- Whisper (openai/whisper-base), OpenAI — MIT License
- WhisperKit, Argmax Inc. — MIT License
- Zeroth-Korean (OpenSLR-40), Lucas Jo & Wonkyum Lee, Atlas Guide Inc. / Gooiny Inc.
  — CC BY 4.0 (https://openslr.org/40/)
- MUSAN corpus (OpenSLR-17), D. Snyder, G. Chen, D. Povey — CC BY 4.0
  (https://openslr.org/17/)
- Room Impulse Response and Noise Database (OpenSLR-28), T. Ko et al. — Apache 2.0
  (https://openslr.org/28/)
- MeloTTS, MyShell.ai — MIT License   ※ 학습 전환 시 추가
```

KsponSpeech를 계속 쓸 경우 아래를 추가하고, 사전에 상업 이용 승인을 받아야 한다:

```
- KsponSpeech (한국어 음성), AI Hub — 한국지능정보사회진흥원 사업결과
```

---

## 5. 근거 링크

- [Zeroth Korean — OpenSLR 40](https://openslr.org/40/) / [Bingsu/zeroth-korean](https://huggingface.co/datasets/Bingsu/zeroth-korean)
- [MUSAN — OpenSLR 17](https://www.openslr.org/17/)
- [RIRS_NOISES — OpenSLR 28](https://www.openslr.org/28/)
- [AI Hub 이용정책](https://www.aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105) / [AI Hub FAQ (모델 배포·데이터 유출 규정)](https://www.aihub.or.kr/aihubnews/faq/list.do)
- [AI Hub 한국어 음성(KsponSpeech)](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=123)
- [AI Hub 구음장애 음성인식 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=608)
- [facebook/mms-tts-kor (CC BY-NC 4.0)](https://huggingface.co/facebook/mms-tts-kor)
- [gTTS — 공개/상업 프로젝트 법적 이슈 안내](https://github.com/pndurette/gTTS/issues/309)
- [myshell-ai/MeloTTS-Korean (MIT)](https://huggingface.co/myshell-ai/MeloTTS-Korean)
- [openai/whisper LICENSE (MIT)](https://github.com/openai/whisper/blob/main/LICENSE)
- [argmaxinc/WhisperKit LICENSE (MIT)](https://github.com/argmaxinc/WhisperKit/blob/main/LICENSE)
- [Coqui XTTS-v2 CPML 비상업 제한](https://github.com/coqui-ai/TTS/discussions/4304)
