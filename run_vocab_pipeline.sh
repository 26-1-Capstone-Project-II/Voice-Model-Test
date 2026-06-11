#!/usr/bin/env bash
# ============================================================
# 외래어/저빈도어 어휘 확장 재학습 — 서버(Linux) 원샷 파이프라인
# ============================================================
# 문제: Zeroth(순수 한국어 낭독체)로 학습한 모델이 외래어/저빈도어 OOV 오인식
#       (데시벨→대시베르, 스톤 프로젝트→캠핑카 소개팅). 음향 증강이 아니라
#       '어휘 도메인 확장'이 필요 → TTS 타깃 합성 + 실제 자유대화(KsponSpeech).
#
# 순서:
#   (A) 외래어 TTS 합성셋 준비   (없으면 생성)
#   (B) KsponSpeech 정규화셋 준비 (경로 주면 생성, 아니면 스킵 — 수동 다운로드 전제)
#   (0) 재학습 전 기준선  : 외래어 평가셋(OOV) + Zeroth 평가셋(회귀)
#   (1) 어휘 확장 재학습  : 성숙 모델에서 *이어서* + 추가 소스 병합(+기존 음향 증강 유지)
#   (2) 재학습 후 기준선  : 동일 평가 → 전후 비교
# CoreML 변환은 macOS 전용(여기 미포함): 학습 후 Mac에서 ./convert_to_coreml.sh.
#
# 사용:
#   # 빠른 배선 점검 (소량/1에폭, gtts 합성)
#   SMOKE=1 ./run_vocab_pipeline.sh
#
#   # 본 학습 (TTS=mms, KsponSpeech 포함)
#   KSPON_AUDIO_ROOT=/data/KsponSpeech KSPON_TRN=/data/KsponSpeech_scripts/train.trn \
#       ./run_vocab_pipeline.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

GPU="${GPU:-0}"
WS="${WS:-$HOME/mingly_workspace/Voice-Model-Test}"
ZEROTH_DIR="${ZEROTH_DIR:-zeroth_dataset}"                    # 주(主) 코퍼스 (val/test 기준)
LOANWORD_DIR="${LOANWORD_DIR:-$WS/loanword_dataset}"          # 외래어 TTS 합성셋
KSPON_DIR="${KSPON_DIR:-$WS/kspon_dataset}"                   # 실제 자유대화 정규화셋
BASELINE_MODEL="${BASELINE_MODEL:-best_model_zeroth_aug/best}"  # 현재 배포(이어서 학습 시작점)
INIT_MODEL="${INIT_MODEL:-$BASELINE_MODEL}"
OUT_DIR="${OUT_DIR:-best_model_vocab}"
RESULTS="${RESULTS:-results_vocab}"
OVERSAMPLE="${OVERSAMPLE:-3}"                                 # TTS 타깃셋 노출 배수
TTS_BACKEND="${TTS_BACKEND:-mms}"
export MUSAN_NOISE_DIR="${MUSAN_NOISE_DIR:-/data/musan/noise}"
export RIR_DIR="${RIR_DIR:-/data/RIRS_NOISES/simulated_rirs}"
SMOKE="${SMOKE:-0}"

PY="PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=$GPU python"
NEW_MODEL="$OUT_DIR/best"

# 이어서 학습이면 낮은 LR (성숙 모델 보존). 생짜면 2e-5.
[ "$INIT_MODEL" != "scratch" ] && DEFAULT_LR="1e-5" || DEFAULT_LR="2e-5"
LR="${LR:-$DEFAULT_LR}"
EPOCHS="${EPOCHS:-3}"

if [ "$SMOKE" = "1" ]; then
    echo "🧪 SMOKE 모드 — 소량/1에폭 빠른 배선 점검"
    TTS_BACKEND="${TTS_BACKEND_SMOKE:-gtts}"
    TTS_ARGS="--per_word 2 --limit 40"
    DIAG_N=8; TRAIN_ARGS="--max_samples 200 --num_epochs 1 --lr $LR"
else
    echo "🚀 본 학습 모드"
    TTS_ARGS="--per_word 6 --perturb"
    DIAG_N=200; TRAIN_ARGS="--num_epochs $EPOCHS --lr $LR"
fi

echo "  GPU=$GPU  ZEROTH_DIR=$ZEROTH_DIR"
echo "  LOANWORD_DIR=$LOANWORD_DIR  KSPON_DIR=$KSPON_DIR"
echo "  INIT=$INIT_MODEL  →  OUT=$NEW_MODEL  (oversample ×$OVERSAMPLE)"

# ── 사전 점검 ──────────────────────────────────────────────
[ -f "$ZEROTH_DIR/test.jsonl" ] || { echo "❌ $ZEROTH_DIR/test.jsonl 없음 (prepare_zeroth.py 먼저)"; exit 1; }
[ -f "$BASELINE_MODEL/config.json" ] || echo "⚠️ $BASELINE_MODEL 없음 — STEP 0(전 기준선) 건너뜀"

# ── STEP A: 외래어 TTS 합성셋 준비 ─────────────────────────
echo ""; echo "════════ STEP A: 외래어 TTS 합성셋 ════════"
if [ -f "$LOANWORD_DIR/train.jsonl" ]; then
    echo "  ✅ 이미 존재 → $LOANWORD_DIR (재생성 원하면 디렉터리 삭제 후 재실행)"
else
    eval $PY prepare_loanword_tts.py --backend "$TTS_BACKEND" \
        --output_dir "$LOANWORD_DIR" $TTS_ARGS
fi
[ -f "$LOANWORD_DIR/test.jsonl" ] && LOAN_EVAL="$LOANWORD_DIR" || LOAN_EVAL=""

# ── STEP B: KsponSpeech 정규화셋 준비(선택) ────────────────
echo ""; echo "════════ STEP B: KsponSpeech 정규화셋 ════════"
if [ -f "$KSPON_DIR/train.jsonl" ]; then
    echo "  ✅ 이미 존재 → $KSPON_DIR"
elif [ -n "${KSPON_AUDIO_ROOT:-}" ] && [ -n "${KSPON_TRN:-}" ]; then
    eval $PY prepare_kspon.py --audio_root "$KSPON_AUDIO_ROOT" --trn "$KSPON_TRN" \
        --output_dir "$KSPON_DIR" --max_utts "${KSPON_MAX_UTTS:-60000}"
else
    echo "  ⏭️  KSPON_AUDIO_ROOT/KSPON_TRN 미지정 — KsponSpeech 생략(TTS만 사용)."
    echo "     (KsponSpeech 는 AIHub 수동 다운로드 전제. 받은 뒤 위 두 env 지정해 재실행)"
fi

# ── 추가 소스 목록 구성 ────────────────────────────────────
EXTRA=""
[ -f "$LOANWORD_DIR/train.jsonl" ] && EXTRA="$LOANWORD_DIR"
[ -f "$KSPON_DIR/train.jsonl" ] && EXTRA="${EXTRA:+$EXTRA,}$KSPON_DIR"
[ -n "$EXTRA" ] || { echo "❌ 추가 학습 소스가 하나도 없음 — 중단"; exit 1; }
echo ""; echo "  ➕ 학습 추가 소스: $EXTRA"

# ── 평가 헬퍼: 외래어 OOV(clean) + Zeroth 회귀 ─────────────
eval_sets () {  # $1=model_path  $2=결과 하위폴더
    local M="$1" TAG="$2"
    if [ -n "$LOAN_EVAL" ]; then
        echo "  · 외래어 OOV 평가 ($TAG)"
        eval $PY diagnose_farfield_baseline.py --model_path "$M" \
            --json_dir "$LOAN_EVAL" --split test --num_samples $DIAG_N --apply_g2p \
            --snr_list 10 --tail_sec 0 --noise_only_clips 0 \
            --output_dir "$RESULTS/$TAG/loanword"
    fi
    echo "  · Zeroth 회귀 평가 ($TAG)"
    eval $PY diagnose_farfield_baseline.py --model_path "$M" \
        --json_dir "$ZEROTH_DIR" --split test --num_samples $DIAG_N --apply_g2p \
        --tail_sec 8 --noise_only_clips 0 \
        --output_dir "$RESULTS/$TAG/zeroth"
}

# ── STEP 0: 재학습 전 기준선 ──────────────────────────────
if [ -f "$BASELINE_MODEL/config.json" ]; then
    echo ""; echo "════════ STEP 0: 재학습 전 기준선 ════════"
    eval_sets "$BASELINE_MODEL" "before"
fi

# ── STEP 1: 어휘 확장 재학습 (이어서 + 추가 소스 + 기존 음향 증강 유지) ──
echo ""; echo "════════ STEP 1: 어휘 확장 재학습 ════════"
INIT_ARG=""
if [ "$INIT_MODEL" != "scratch" ] && [ -f "$INIT_MODEL/config.json" ]; then
    INIT_ARG="--init_model $INIT_MODEL"
    echo "  ↪️  $INIT_MODEL 에서 이어서 학습 (+어휘 확장)"
else
    echo "  ↪️  openai/whisper-base 에서 처음부터 학습"
fi
# 음향 증강(p_tail/longform/noise_only)은 기존 강건성 보존을 위해 그대로 유지.
eval $PY finetune_whisper.py \
    --json_dir "$ZEROTH_DIR" --apply_g2p \
    --extra_json_dirs "$EXTRA" --oversample "$OVERSAMPLE" \
    --p_tail 0.3 --longform_prob 0.3 --noise_only_prob "${NOISE_ONLY_PROB:-0.05}" \
    --batch_size 8 --grad_accum 2 \
    --output_dir "$OUT_DIR" $INIT_ARG $TRAIN_ARGS

[ -f "$NEW_MODEL/config.json" ] || { echo "❌ 재학습 결과 $NEW_MODEL 없음 — 학습 실패"; exit 1; }

# ── STEP 2: 재학습 후 기준선 ──────────────────────────────
echo ""; echo "════════ STEP 2: 재학습 후 기준선 ════════"
eval_sets "$NEW_MODEL" "after"

# ── 결과 안내 ──────────────────────────────────────────────
echo ""; echo "✅ 파이프라인 완료"
echo "  외래어 OOV 전/후 : $RESULTS/before/loanword/summary.md  vs  $RESULTS/after/loanword/summary.md"
echo "  Zeroth 회귀 전/후 : $RESULTS/before/zeroth/summary.md   vs  $RESULTS/after/zeroth/summary.md"
echo "  새 모델          : $NEW_MODEL"
echo ""
echo "기대: 외래어 OOV CER 큰 폭 감소, Zeroth clean/소음 CER 회귀 없음."
echo "다음: 새 모델을 Mac으로 가져가 ./convert_to_coreml.sh $NEW_MODEL 로 CoreML 변환."
