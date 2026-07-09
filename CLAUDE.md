# Voice-Model-Test — Claude 작업 지침

이 리포는 Whisper 한국어 **발음 전사** 모델의 학습/증강/CoreML 변환(Python) 쪽이다.
(iOS 앱 On-Voice 는 별도 리포. 화자 게이트 등 앱 측 코드는 여기에 없다.)

## 경쟁 화자 증강(Competing-Speaker Augmentation) 워크스트림

목적: 다화자·소음 환경에서 모델이 전경(타깃) 화자에 집중하도록 학습.
On-Voice 앱의 화자 게이트(프론트엔드 방어)에 상보적인 **학습 측 방어 계층**이다.
마스터 플랜: On-Voice 리포 `docs/speaker-isolation-plan.md` §7 (Phase 3).
이 리포의 Phase 0 확인 결과·설계 결정: `docs/phase0-findings.md`.

### 절대 원칙 (위반 금지 — 충돌 시 구현 중단·질문)

- **라벨 무결성:** 라벨은 **타깃 화자 전사만**. 간섭 화자의 전사는 어떤 형태로도
  라벨·프롬프트에 끼면 안 된다. 증강기는 오디오만 입출력한다.
- **간섭원은 held-out 화자.** MUSAN speech 금지(라벨 부재·도메인 불일치).
  간섭 화자 집합은 학습 타깃 집합과 **화자 단위로 배타적**이어야 한다.
  → 데이터 소스 `Bingsu/zeroth-korean` 에 화자 ID 가 없으므로,
  **Zeroth 공식 test 스플릿**(OpenSLR-40 train/test 는 화자 배타적)을 간섭 풀로 사용한다.
  (`finetune_whisper.py` 가 `splits["test"]` wav 를 `FarFieldAugmentor(speech_files=...)` 로 전달)
- **증강 순서 고정:** 경쟁 화자 믹스 → RIR → MUSAN → SpecAugment.
  신규 스테이지는 `FarFieldAugmentor.__call__` **맨 앞단**. SpecAugment 는 파형이 아니라
  모델 내장(`model.config.apply_spec_augment=True`, mel 단계).
- **레시피:** SIR(타깃−간섭) 5~20dB 균등 위주 + 전체 ~10%는 0~5dB 하드.
  **부분 겹침(onset/offset) 위주, full-overlap 없음.** 시드 고정(`seed=`), test 스플릿(=간섭 풀) 커밋.
- **모델 전체성:** 프로덕션 모델은 `openai/whisper-base` **full fine-tuning**
  (whisper-tiny 아님, LoRA 아님 — README 구버전 기재에 의지 말 것. **코드가 진실**).
- **평가:** 다화자 조건 자모/문자 손실 오류율 + 깨끗한 발화 절대 무회귀. **DER 사용 금지.**
- **하드코딩 금지:** 확률·SIR·임계값은 생성자 인자/CLI/env(`COMPETING_PROB`)로 분리.
- 재학습 후 macOS 에서 `./convert_to_coreml.sh` (coremltools) → WhisperKit CoreML 교체,
  가중치는 Git LFS.

### 실행

```bash
# 서버 원샷 (경쟁 화자 포함): 기본 COMPETING_PROB=0.3
# 옵션 COMPETING_OWN_RIR_PROB>0 → 간섭 화자를 타깃과 다른 RIR 로(공간 분리, 플랜 §7 옵션)
COMPETING_PROB=0.3 COMPETING_OWN_RIR_PROB=0.0 ./run_server_pipeline.sh
# 직접 호출
python finetune_whisper.py --json_dir zeroth_dataset --apply_g2p --competing_prob 0.3 ...
# 단위 테스트
python test_competing_speaker.py

# §9 다화자 평가 (재학습 전후 대조): comp_sir* 조건 + 자모 손실 오류율(JER)
python diagnose_farfield_baseline.py --model_path best_model_whisper/best \
    --json_dir zeroth_dataset --apply_g2p --num_samples 200 \
    --competing_sir_list 0,5,10,15,20 --competing_overlaps 0.5,1.0
```

### 필수 테스트 (`test_competing_speaker.py`)

- 라벨 무결성: 경쟁 믹싱이 타깃을 제거/치환하지 않고 더하기만 함(겹침 밖 바이트 동일)
- SIR 분포: 복원 SIR 이 설정 범위 안 + 하드(<5dB) 비율이 p_hard_sir 에 수렴
- full-overlap 없음 / 간섭 풀 없으면 자동 비활성

### 컨벤션

- 브랜치: `feat/competing-speaker-aug`
- 커밋: `feat:`/`fix:` 접두. 커밋에 Co-Authored-By Claude 넣지 않음.
