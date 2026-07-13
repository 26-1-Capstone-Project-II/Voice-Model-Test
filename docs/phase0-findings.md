# Phase 0 — 코드베이스 확인 결과 (Voice-Model-Test 부분)

> 마스터 플랜: `docs/speaker-isolation-plan.md` §4/§7/§8.
> 이 문서는 Phase 3(경쟁 화자 증강) 구현 착수 **전** 코드 확인 결과다.
> On-Voice(iOS) 부분은 별도 리포이므로 여기서는 다루지 않는다.

## 1. §8 코드 맵 — Voice-Model-Test 행 채움

| # | 관심사 | 실제 경로·심볼 | 비고 |
|---|---|---|---|
| 9 | FarFieldAugmentor | `augment.py` → `class FarFieldAugmentor` | 스테이지·순서·config → 아래 §2 |
| 10 | 데이터 로드 / 화자 ID | `finetune_whisper.py:load_data` (71), `WhisperPhoneticDataset` (175); 준비 `prepare_zeroth.py` | **화자 ID 없음 — §3 블로커** |
| 11 | 자모 손실 오류율 평가 | `diagnose_farfield_baseline.py`(환각률/삽입률/CER/**JER**/**다화자 조건**), `jamo_utils.py`(자모 분해), `pronunciation_evaluator.py` | §9 구현 완료: `jer()` + `comp_sir*` 조건(held-out test 페어링). 학습 중 지표는 여전히 CER(`make_compute_metrics`) |

## 2. FarFieldAugmentor 실제 구조 (플랜 §7 순서와 대조)

현재 `__call__` 파형 증강 순서 (`augment.py:134-147`):

```
clean → (p_reverb) reverb[RIR] → (p_noise) +noise[MUSAN]@SNR → (p_gain) gain → (p_tail) 비음성 꼬리 → 클리핑 방지
```

- SpecAugment는 파형이 아니라 **모델 내장**으로 적용됨: `finetune_whisper.py:467` `model.config.apply_spec_augment=True` (mel 단계, eval 미적용).
- 따라서 플랜 §7의 "경쟁 화자 믹스 → RIR → MUSAN → SpecAugment"는:
  **경쟁 믹스(신규, 파형 맨 앞단) → reverb(RIR) → noise(MUSAN) → …(gain/tail) → [모델] SpecAugment** 로 성립.
- config 주입: 생성자 인자 + `finetune_whisper.py`의 CLI 인자 + `run_server_pipeline.sh` env. 하드코딩 없음(플랜 §1-8 준수 경로 존재).
- 증강은 **train split에만** 적용(`WhisperPhoneticDataset(augmentor=...)`), val/test는 clean(`augmentor=None`). noise-only 샘플엔 미적용(`finetune_whisper.py:266` 가드).

## 3. 블로커 — 화자 ID 부재 (플랜 §7 held-out 요구와 충돌)

- 데이터 소스 `Bingsu/zeroth-korean`의 스키마는 `{audio, text}` **뿐**. speaker/spk/화자 필드 없음
  (HF datasets-server info/first-rows로 확인. train 22,263 / test 457).
- `prepare_zeroth.py`는 `text`/`audio`만 읽고, wav를 셔플 후 순번(`train_000123.wav`)으로 추출 →
  화자 정보가 파일명에도 없음.
- 결과: **현재 파이프라인만으로는 "화자 단위 배타적 held-out 간섭 스플릿"(§7) 생성 불가.**

### 해결 옵션 (사용자 결정 필요 — §1 충돌 중단 규칙)

- **A. 원본 OpenSLR-40에서 화자 ID 복원 후 재준비**
  원본 Zeroth는 파일명에 화자 인코딩(`spk_utt`). 학습 화자 집합을 target vs held-out-interferer로
  화자 단위 분할 → 대규모·정합 간섭 풀. 플랜 §7에 가장 충실하나 서버 재다운로드·재추출 비용.
- **B. Zeroth 공식 test 스플릿을 held-out 간섭 풀로 사용** *(추천)*
  OpenSLR-40 train/test는 **화자 배타적**(test 화자 ⊄ train). test.jsonl 발화를 간섭원으로 쓰면
  화자 단위 배타 요건을 재준비 0으로 충족. 라벨 무결성도 자동(test 전사는 학습 라벨이 된 적 없음).
  비용: test가 이중 용도(간섭 + `run_server_pipeline.sh` 전후 기준선). §9 평가용 간섭 풀도 test와
  "동일 집합"이라 플랜과 일관. 간섭 다양성은 457발화×랜덤 크롭/온셋으로 확보.
- **C. 학습셋 자기혼합(비-held-out)**
  최소 작업이나 **플랜 §7 held-out 요구 위반** — 간섭 화자 암기·라벨 오염 위험. 비권장.

### 결정 (2026-07-09)

**옵션 B 채택 — Zeroth 공식 test 스플릿을 held-out 간섭 풀로 사용.**
`finetune_whisper.py` 가 `splits["test"]` 의 `wav_path` 목록을 `FarFieldAugmentor(speech_files=...)`
로 전달한다. train 발화는 간섭원으로 쓰지 않으므로 화자 단위 배타·라벨 무결성 요건을 충족한다.
남은 유의점: test 가 간섭원 + `run_server_pipeline.sh` 전후 기준선으로 이중 사용됨(문서화된 수용).

## 4. 그 외 확인 사항

- 브랜치 컨벤션: 플랜 §11 → `feat/competing-speaker-aug`. 현재 브랜치 `whisper-base-zeroth`.
- 재학습은 성숙 모델에서 이어서(`--init_model best_model_zeroth_aug/best`, 낮은 LR)가 기존 관례.
- 재학습 후 macOS에서 `./convert_to_coreml.sh` → WhisperKit CoreML 교체(가중치 Git LFS).
