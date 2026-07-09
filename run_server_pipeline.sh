#!/usr/bin/env bash
# ============================================================
# 원거리·소음 후반부 환각 재학습 — 서버(Linux) 원샷 파이프라인
# ============================================================
# 순서: (0) 재학습 전 기준선 → (1) 증강 재학습 → (2) 재학습 후 기준선 → 비교 안내
# CoreML 변환은 macOS 전용이라 여기 포함하지 않음 (학습 후 Mac에서 ./convert_to_coreml.sh).
#
# 사용:
#   # 빠른 배선 확인 (소량/1에폭, 수 분)
#   SMOKE=1 ./run_server_pipeline.sh
#
#   # 본 학습
#   ./run_server_pipeline.sh
#
# 환경변수(기본값):
#   GPU=0                          CUDA_VISIBLE_DEVICES
#   DATA_DIR=zeroth_dataset        학습/평가 JSONL 디렉터리
#   BASELINE_MODEL=best_model_zeroth_aug/best   재학습 전(현재 배포) 모델
#   OUT_DIR=best_model_whisper     재학습 결과 저장 (best/ 하위에 모델)
#   MUSAN_NOISE_DIR=/data/musan/noise
#   RIR_DIR=/data/RIRS_NOISES/simulated_rirs
#   RESULTS=results                기준선 결과 출력 루트
#   SMOKE=0                        1이면 소량 빠른 점검
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

GPU="${GPU:-0}"
DATA_DIR="${DATA_DIR:-zeroth_dataset}"
BASELINE_MODEL="${BASELINE_MODEL:-best_model_zeroth_aug/best}"
# 기본: 성숙한 모델에서 *이어서* 학습 (깨끗한 정확도 보존 + 증강만 추가).
# 생짜(openai/whisper-base)에서 처음부터 학습하려면 INIT_MODEL=scratch 로.
INIT_MODEL="${INIT_MODEL:-$BASELINE_MODEL}"
OUT_DIR="${OUT_DIR:-best_model_whisper}"
RESULTS="${RESULTS:-results}"
export MUSAN_NOISE_DIR="${MUSAN_NOISE_DIR:-/data/musan/noise}"
export RIR_DIR="${RIR_DIR:-/data/RIRS_NOISES/simulated_rirs}"
# 경쟁 화자 증강(플랜 §7): held-out(test) 다른 화자를 타깃 우세 SIR 로 부분 겹침 혼합.
# 간섭 풀은 DATA_DIR/test.jsonl(train 과 화자 배타적). 0 이면 비활성.
COMPETING_PROB="${COMPETING_PROB:-0.3}"
SMOKE="${SMOKE:-0}"

PY="PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$GPU python"
NEW_MODEL="$OUT_DIR/best"

# 이어서 학습(권장)이면 낮은 LR 권장. 생짜면 2e-5.
[ "$INIT_MODEL" != "scratch" ] && DEFAULT_LR="1e-5" || DEFAULT_LR="2e-5"
LR="${LR:-$DEFAULT_LR}"
EPOCHS="${EPOCHS:-3}"

if [ "$SMOKE" = "1" ]; then
    echo "🧪 SMOKE 모드 — 소량/1에폭 빠른 배선 점검"
    DIAG_N=8; TRAIN_ARGS="--max_samples 200 --num_epochs 1 --lr $LR"; NOISE_CLIPS=4
else
    echo "🚀 본 학습 모드"
    DIAG_N=200; TRAIN_ARGS="--num_epochs $EPOCHS --lr $LR"; NOISE_CLIPS=50
fi

echo "  GPU=$GPU  DATA_DIR=$DATA_DIR"
echo "  BASELINE_MODEL=$BASELINE_MODEL  →  OUT=$NEW_MODEL"
echo "  MUSAN_NOISE_DIR=$MUSAN_NOISE_DIR  RIR_DIR=$RIR_DIR  COMPETING_PROB=$COMPETING_PROB"

# ── 사전 점검 ──────────────────────────────────────────────
[ -f "$DATA_DIR/test.jsonl" ] || { echo "❌ $DATA_DIR/test.jsonl 없음 (prepare_zeroth.py 먼저)"; exit 1; }
[ -f "$BASELINE_MODEL/config.json" ] || echo "⚠️ $BASELINE_MODEL 없음 — STEP 0(전 기준선) 건너뜀"
[ -d "$MUSAN_NOISE_DIR" ] || echo "⚠️ MUSAN_NOISE_DIR 없음 — 소음/꼬리 증강이 무음으로 대체됨"
[ -d "$RIR_DIR" ] || echo "⚠️ RIR_DIR 없음 — 잔향 증강 생략됨"

# ── STEP 0: 재학습 전 기준선 ──────────────────────────────
if [ -f "$BASELINE_MODEL/config.json" ]; then
    echo ""; echo "════════ STEP 0: 재학습 전 기준선 ════════"
    eval $PY diagnose_farfield_baseline.py \
        --model_path "$BASELINE_MODEL" \
        --json_dir "$DATA_DIR" --num_samples $DIAG_N --apply_g2p \
        --tail_sec 8 --noise_only_clips $NOISE_CLIPS \
        --output_dir "$RESULTS/baseline_before"
fi

# ── STEP 1: 증강 재학습 (꼬리 + long-form + 경쟁화자 + 앱 일치 디코딩) ──
echo ""; echo "════════ STEP 1: 증강 재학습 ════════"
# init_model: 성숙한 모델에서 이어서 학습(권장). INIT_MODEL=scratch 면 생짜에서.
INIT_ARG=""
if [ "$INIT_MODEL" != "scratch" ] && [ -f "$INIT_MODEL/config.json" ]; then
    INIT_ARG="--init_model $INIT_MODEL"
    echo "  ↪️  $INIT_MODEL 에서 이어서 학습"
else
    echo "  ↪️  openai/whisper-base 에서 처음부터 학습"
fi
eval $PY finetune_whisper.py \
    --json_dir "$DATA_DIR" --apply_g2p \
    --p_tail 0.3 --longform_prob 0.3 --noise_only_prob "${NOISE_ONLY_PROB:-0.05}" \
    --competing_prob "$COMPETING_PROB" \
    --batch_size 8 --grad_accum 2 \
    --output_dir "$OUT_DIR" $INIT_ARG $TRAIN_ARGS

[ -f "$NEW_MODEL/config.json" ] || { echo "❌ 재학습 결과 $NEW_MODEL 없음 — 학습 실패"; exit 1; }

# ── STEP 2: 재학습 후 기준선 (개선 비교) ──────────────────
echo ""; echo "════════ STEP 2: 재학습 후 기준선 ════════"
eval $PY diagnose_farfield_baseline.py \
    --model_path "$NEW_MODEL" \
    --json_dir "$DATA_DIR" --num_samples $DIAG_N --apply_g2p \
    --tail_sec 8 --noise_only_clips $NOISE_CLIPS \
    --output_dir "$RESULTS/baseline_after"

# ── 결과 안내 ──────────────────────────────────────────────
echo ""; echo "✅ 파이프라인 완료"
echo "  재학습 전 : $RESULTS/baseline_before/summary.md"
echo "  재학습 후 : $RESULTS/baseline_after/summary.md"
echo "  새 모델   : $NEW_MODEL"
echo ""
echo "두 summary.md 의 환각률/삽입률/CER 을 비교하세요 (꼬리·소음 조건에서 감소 기대)."
echo "다음: 새 모델을 Mac으로 가져가 ./convert_to_coreml.sh $NEW_MODEL 로 CoreML 변환."
