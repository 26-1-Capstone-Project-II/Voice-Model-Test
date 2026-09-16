# archive — 현역이 아닌 자산 보관소

현재 배포 모델(`best_model_vocab_e/best` → `Whisper_CoreML_Model`)의 학습·평가·변환에
**필요하지 않은** 것들을 여기 모았습니다. 지우지 않은 이유는 계보 추적(모델 A/B 대조)과
과거 검증 결과의 재현 가능성 때문입니다.

| 폴더 | 내용 | 왜 아카이브인가 |
|------|------|-----------------|
| `models/` | 구버전 체크포인트 5종 | 계보상 `best_model_vocab_e` 로 대체됨 (아래 표) |
| `eval_aihub/` | AIHub 구음장애 사용자 효용 검증 워크스트림 | 1회성 검증(README §2차 검증). 현재 재학습 루프에서 호출되지 않음 |
| `diagnostics/` | 일회성 진단 스크립트 | 정렬/데이터 점검용. 상시 지표는 `diagnose_farfield_baseline.py` 가 담당 |
| `legacy_pipelines/` | 초기 학습 실험(jamo vocab, LoRA, 샘플링) | `finetune_whisper.py` full fine-tuning 으로 통합됨 |
| `results/` | 경쟁 화자·babble 재학습 전후 평가 산출물 | 과거 런의 기록. 새 평가는 `results/` 에 새로 생성됨 |
| `logs/` | CoreML 변환 로그 | 참고용 (gitignore 대상이라 추적되지 않음) |

## 모델 계보

| 체크포인트 | 무엇 | 상태 |
|------------|------|------|
| `models/best_model` | 초기 자모(jamo) vocab 실험 설정(가중치 없음, vocab 1207) | 폐기 |
| `models/finetuned_model_lora` | LoRA 어댑터 시절 산출물 | 폐기(full fine-tuning 으로 전환) |
| `models/best_model_whisper` | Zeroth 발음 전사 → 경쟁 화자·babble 증강까지 누적된 배포본 | 대체됨 |
| `models/best_model_zeroth_aug` | 원거리/소음(RIR·MUSAN·tail·long-form) 증강 재학습본 | 대체됨 |
| `models/best_model_vocab` | 외래어 어휘 확장 v3 | 대체됨 |
| **`../best_model_vocab_e`** | **외래어 어휘 확장 런 E — 현재 배포본** | **현역** |

## 여기 있는 스크립트를 실행하려면

`korean_g2p_nomecab.py`, `jamo_utils.py` 같은 공용 모듈을 import 하므로 **리포 루트에서
`PYTHONPATH=.` 를 붙여** 실행합니다.

```bash
PYTHONPATH=. python archive/eval_aihub/test_aihub_dysarthria.py \
    --model_path best_model_vocab_e/best --json_dir segmented_dataset
```
