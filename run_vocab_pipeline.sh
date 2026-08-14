#!/usr/bin/env bash
# ============================================================
# 외래어/저빈도어 어휘 확장 재학습 — 서버(Linux) 원샷 파이프라인 (v3)
# ============================================================
# 문제: Zeroth(순수 한국어 낭독체)로 학습한 모델이 외래어/저빈도어 OOV 오인식
#       (데시벨→대시베르, 캡스톤 프로젝트→캠핑카 소개팅). 음향 증강이 아니라
#       '어휘 도메인 확장'이 필요 → TTS 타깃 합성 + 실제 자유대화(KsponSpeech).
#
# v2 학습에서 검증된 문제 4가지와 v3 대응:
#   1. TTS 단일 도메인(gtts만)         → (v3.1 갱신: 라이선스 문제로 gtts/mms 전면 금지,
#                                        melo(MIT) 단독으로 전환 — 상세는 아래 라이선스 메모)
#   2. 받침/연음 신호 부족             → KsponSpeech(실발화) 미지정 시 명시적 실패
#   3. 장문 열화                       → 문단 낭독 평가셋(test_paragraph) 신설
#   4. 평가 맹점(학습 도메인 held-out) → 학습 미사용 TTS 엔진으로 test 합성(OOD)
#                                        ※ v3.1: melo가 학습 백엔드로 들어가면서
#                                        held-out TTS 엔진이 없음 — 임시로 OOD 분리
#                                        비활성(TTS_EVAL_BACKEND 미지정 시 학습 풀과
#                                        동일 도메인으로 평가). 실기기 테스트가 그동안
#                                        유일한 진짜 OOD 신호. 후속: 실녹음 eval셋으로 대체 예정.
#
# v3.2 사고(2026-08-13 발견): --extra_json_dirs / --oversample 가 finetune_whisper.py
# 호출부에서 train() 으로 전달되지 않아, 외래어·KsponSpeech 가 통째로 빠진 채 Zeroth
# 단독으로 재학습되고 있었다. 학습 로그에 "➕ 추가 소스 병합" 줄이 없으면 그 상태다.
# → 전달 배선 복구 + 병합 0개면 명시적 실패 + 소스별 배수(LOAN_OVERSAMPLE/KSPON_OVERSAMPLE).
#
# 라이선스 메모(2026-07-31): gTTS(translate.google.com 비공식 엔드포인트, 상업 이용
# 근거 없음)·facebook/mms-tts-kor(CC BY-NC 4.0, 비상업 전용) 둘 다 상업 배포 블로커라
# v3.1부터 학습 TTS는 melo(MIT, myshell-ai/MeloTTS-Korean) 단독만 사용한다.
# xtts(Coqui, CPML 비상업)도 동일 사유로 학습·평가 어느 쪽에도 사용 금지.
#
# 순서:
#   (A) 외래어 TTS 합성셋 준비   (캐시는 코퍼스 지문 일치 시에만 재사용)
#   (B) KsponSpeech 정규화셋 준비 (미지정 시 실패 — TTS 단독 학습 금지)
#   (0) 재학습 전 기준선  : 외래어 OOD + 문단 낭독 + Zeroth 회귀(다화자/babble 포함)
#   (1) 어휘 확장 재학습  : 현재 배포본(babble)에서 *이어서* + 경쟁화자/babble 증강 유지
#   (2) 재학습 후 기준선  : 동일 평가 → 전후 비교
# CoreML 변환은 macOS 전용(여기 미포함): 학습 후 Mac에서 ./convert_to_coreml.sh.
#
# 사용:
#   # 빠른 배선 점검 (소량/1에폭)
#   SMOKE=1 ./run_vocab_pipeline.sh
#
#   # 본 학습 (TTS=melo 단독, KsponSpeech 포함)
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
# 현재 배포본(babble 증강, On-Voice #268 = 6efa494c 산출물)에서 이어서 학습.
# 화자 격리(경쟁 화자/babble) 학습 측 방어를 보존한 채 어휘만 확장한다.
BASELINE_MODEL="${BASELINE_MODEL:-best_model_whisper/best}"
INIT_MODEL="${INIT_MODEL:-$BASELINE_MODEL}"
OUT_DIR="${OUT_DIR:-best_model_vocab}"
RESULTS="${RESULTS:-results_vocab}"
OVERSAMPLE="${OVERSAMPLE:-3}"                                 # TTS 타깃셋 노출 배수
# 소스별 배수(미지정 시 OVERSAMPLE 과 동일 = 기존 동작). 외래어셋은 KsponSpeech 6만에
# 비해 1천 단위라 같은 배수를 쓰면 학습 데이터의 2% 안팎에 그친다 — 어휘 확장 신호를
# 키우려면 LOAN_OVERSAMPLE 만 올린다(단일 변수 A/B).
LOAN_OVERSAMPLE="${LOAN_OVERSAMPLE:-$OVERSAMPLE}"
KSPON_OVERSAMPLE="${KSPON_OVERSAMPLE:-$OVERSAMPLE}"
# TTS: melo(MIT) 단독 — gtts/mms/xtts는 상업 이용 불가라 전면 배제(위 라이선스 메모).
# EVAL 백엔드는 학습에 쓰지 않는 엔진일 때만 진짜 OOD 인데, 지금은 melo 외에 상업
# 이용 가능한 대체 엔진이 없어 미지정(임시) — test/문단이 학습 풀과 같은 도메인이
# 된다. 진짜 일반화 확인은 실기기 테스트로 대체. 후속: 실녹음 eval셋 도입 예정.
TTS_BACKEND="${TTS_BACKEND:-melo}"
TTS_EVAL_BACKEND="${TTS_EVAL_BACKEND:-}"
PARAGRAPHS="${PARAGRAPHS:-60}"                                # 문단 낭독 평가 클립 수
# 기존 음향/화자 증강 유지 (직전 배포본과 동일 기본값 — run_server_pipeline.sh 참조)
COMPETING_PROB="${COMPETING_PROB:-0.3}"
COMPETING_OWN_RIR_PROB="${COMPETING_OWN_RIR_PROB:-0.0}"
# 경쟁 화자 중 하드 SIR(0~5dB) 비율. 평가의 comp_sir0 는 이 구간의 경계값이라,
# 그 조건만 회귀할 때 올린다(기본 0.1 = 기존 동작).
COMPETING_HARD_PROB="${COMPETING_HARD_PROB:-0.1}"
BABBLE_PROB="${BABBLE_PROB:-0.3}"
export MUSAN_NOISE_DIR="${MUSAN_NOISE_DIR:-/data/musan/noise}"
export RIR_DIR="${RIR_DIR:-/data/RIRS_NOISES/simulated_rirs}"
SMOKE="${SMOKE:-0}"
# TTS 단독 학습(원칙 위반)을 의식적으로 허용할 때만 1 (배선 점검 외 사용 금지)
ALLOW_TTS_ONLY="${ALLOW_TTS_ONLY:-0}"

PY="PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$GPU python"
NEW_MODEL="$OUT_DIR/best"

# 이어서 학습이면 낮은 LR (성숙 모델 보존). 생짜면 2e-5.
[ "$INIT_MODEL" != "scratch" ] && DEFAULT_LR="1e-5" || DEFAULT_LR="2e-5"
LR="${LR:-$DEFAULT_LR}"
EPOCHS="${EPOCHS:-3}"

if [ "$SMOKE" = "1" ]; then
    echo "🧪 SMOKE 모드 — 소량/1에폭 빠른 배선 점검"
    TTS_BACKEND="${TTS_BACKEND_SMOKE:-melo}"
    TTS_EVAL_BACKEND="${TTS_EVAL_BACKEND_SMOKE:-}"
    PARAGRAPHS=4
    TTS_ARGS="--per_word 2 --limit 40"
    DIAG_N=8; NOISE_CLIPS=4; TRAIN_ARGS="--max_samples 200 --num_epochs 1 --lr $LR"
else
    echo "🚀 본 학습 모드"
    TTS_ARGS="--per_word 6"
    # PERTURB=0 이면 화자 섭동(pitch/tempo) 비활성 — melo 오디오 특성과 안 맞아
    # 품질을 깎았을 가능성 검증용 비교 런에 사용(기본값 1=기존 동작 유지).
    [ "${PERTURB:-1}" = "1" ] && TTS_ARGS="$TTS_ARGS --perturb"
    # §9: 고정 조건 비교 — Zeroth test 전량(457) + 시드 고정(diagnose 내부 고정 시드)
    DIAG_N="${DIAG_N:-457}"; NOISE_CLIPS=50; TRAIN_ARGS="--num_epochs $EPOCHS --lr $LR"
fi
TTS_ARGS="$TTS_ARGS --paragraphs $PARAGRAPHS"
[ -n "$TTS_EVAL_BACKEND" ] && TTS_ARGS="$TTS_ARGS --eval_backend $TTS_EVAL_BACKEND"

echo "  GPU=$GPU  ZEROTH_DIR=$ZEROTH_DIR"
echo "  LOANWORD_DIR=$LOANWORD_DIR  KSPON_DIR=$KSPON_DIR"
echo "  INIT=$INIT_MODEL  →  OUT=$NEW_MODEL  (oversample ×$OVERSAMPLE)"
echo "  TTS=$TTS_BACKEND  (OOD 평가: ${TTS_EVAL_BACKEND:-없음})  문단=$PARAGRAPHS"
echo "  증강 유지: COMPETING_PROB=$COMPETING_PROB (하드 SIR 비율 $COMPETING_HARD_PROB) BABBLE_PROB=$BABBLE_PROB"

# ── 사전 점검 ──────────────────────────────────────────────
[ -f "$ZEROTH_DIR/test.jsonl" ] || { echo "❌ $ZEROTH_DIR/test.jsonl 없음 (prepare_zeroth.py 먼저)"; exit 1; }
[ -f "$BASELINE_MODEL/config.json" ] || echo "⚠️ $BASELINE_MODEL 없음 — STEP 0(전 기준선) 건너뜀"

# 소음/잔향 코퍼스 — 없으면 증강이 조용히 no-op 이 된다(로그의 "noise=0 rir=0").
# 학습에선 소음·잔향·비음성 꼬리 증강이, 평가에선 noise/reverb 조건이 통째로 무의미해져
# clean 과 같은 수치가 나오므로, 몇 시간짜리 런을 태우기 전에 여기서 막는다.
# 의식적으로 소음 없이 돌릴 때만 ALLOW_NO_NOISE=1.
MISSING_AUG=""
[ -d "$MUSAN_NOISE_DIR" ] || MISSING_AUG="MUSAN_NOISE_DIR=$MUSAN_NOISE_DIR"
[ -d "$RIR_DIR" ] || MISSING_AUG="${MISSING_AUG:+$MISSING_AUG, }RIR_DIR=$RIR_DIR"
if [ -n "$MISSING_AUG" ]; then
    if [ "${ALLOW_NO_NOISE:-0}" = "1" ] || [ "$SMOKE" = "1" ]; then
        echo "⚠️ 소음/잔향 코퍼스 없음 ($MISSING_AUG) — 소음·잔향·꼬리 증강 없이 진행합니다."
        echo "   평가의 noise/reverb 조건도 clean 과 동일해집니다(비교 무의미)."
    else
        echo "❌ 소음/잔향 코퍼스 경로 없음: $MISSING_AUG"
        echo "   이대로 두면 학습의 소음·잔향·비음성 꼬리 증강이 조용히 비활성화되고,"
        echo "   평가의 noise/reverb 조건도 clean 과 같은 값이 나와 비교가 무의미해집니다."
        echo "   실제 경로를 지정해 재실행하세요:"
        echo "     MUSAN_NOISE_DIR=/경로/musan/noise RIR_DIR=/경로/RIRS_NOISES/simulated_rirs ./run_vocab_pipeline.sh"
        echo "   의식적으로 소음 없이 돌릴 때만 ALLOW_NO_NOISE=1."
        exit 1
    fi
fi

# ── 코퍼스 지문 — 캐시 재사용은 지문이 일치할 때만 ─────────
# v2 사고: 코퍼스를 고쳐도 STEP A 가 "이미 존재 → 스킵"으로 옛 데이터셋을 재사용.
# 코퍼스 생성기 + 합성 설정을 지문으로 남기고, 불일치 시 명시적으로 실패한다.
hash_stdin () {  # sha256 헥사 지문 (Linux=sha256sum, macOS=shasum 둘 다 지원)
    if command -v sha256sum >/dev/null 2>&1; then sha256sum
    else shasum -a 256; fi | cut -d' ' -f1
}
fingerprint_loanword () {
    { cat loanword_corpus.py prepare_loanword_tts.py 2>/dev/null;
      echo "backend=$TTS_BACKEND eval=$TTS_EVAL_BACKEND args=$TTS_ARGS"; } | hash_stdin
}
fingerprint_kspon () {
    { cat prepare_kspon.py 2>/dev/null;
      echo "trn=${KSPON_TRN:-} max=${KSPON_MAX_UTTS:-60000}"; } | hash_stdin
}
check_cache () {  # $1=디렉터리 $2=기대 지문 $3=라벨 → 0=재사용 가능, 1=캐시 없음
    local DIR="$1" WANT="$2" TAG="$3"
    [ -s "$DIR/train.jsonl" ] || return 1
    local GOT=""
    [ -f "$DIR/.corpus_fingerprint" ] && GOT="$(cat "$DIR/.corpus_fingerprint")"
    if [ "$GOT" != "$WANT" ]; then
        echo "❌ $TAG 캐시($DIR)가 현재 코퍼스/설정과 불일치 (지문 ${GOT:-없음} ≠ $WANT)"
        echo "   옛 데이터셋으로 조용히 학습하는 사고 방지를 위해 중단합니다."
        echo "   재생성: rm -rf $DIR 후 재실행"
        exit 1
    fi
    return 0
}

# ── STEP A: 외래어 TTS 합성셋 준비 ─────────────────────────
echo ""; echo "════════ STEP A: 외래어 TTS 합성셋 ════════"
LOAN_FP="$(fingerprint_loanword)"
if check_cache "$LOANWORD_DIR" "$LOAN_FP" "loanword"; then
    echo "  ✅ 캐시 재사용(지문 일치) → $LOANWORD_DIR"
else
    eval $PY prepare_loanword_tts.py --backend "$TTS_BACKEND" \
        --output_dir "$LOANWORD_DIR" $TTS_ARGS
    echo "$LOAN_FP" > "$LOANWORD_DIR/.corpus_fingerprint"
fi
[ -s "$LOANWORD_DIR/test.jsonl" ] && LOAN_EVAL="$LOANWORD_DIR" || LOAN_EVAL=""

# ── STEP B: KsponSpeech 정규화셋 준비 ──────────────────────
# v2 사고: env 미지정 시 조용히 스킵 → TTS 단독 학습이 되어 받침/연음 등
# 자연 발화 신호가 빠졌다. 실발화 없이는 진행하지 않는다(명시적 실패).
echo ""; echo "════════ STEP B: KsponSpeech 정규화셋 ════════"
KSPON_FP="$(fingerprint_kspon)"
if check_cache "$KSPON_DIR" "$KSPON_FP" "kspon"; then
    echo "  ✅ 캐시 재사용(지문 일치) → $KSPON_DIR"
elif [ -n "${KSPON_AUDIO_ROOT:-}" ] && [ -n "${KSPON_TRN:-}" ]; then
    eval $PY prepare_kspon.py --audio_root "$KSPON_AUDIO_ROOT" --trn "$KSPON_TRN" \
        --output_dir "$KSPON_DIR" --max_utts "${KSPON_MAX_UTTS:-60000}"
    echo "$KSPON_FP" > "$KSPON_DIR/.corpus_fingerprint"
elif [ "$SMOKE" = "1" ] || [ "$ALLOW_TTS_ONLY" = "1" ]; then
    echo "  ⚠️ KsponSpeech 없음 — SMOKE/ALLOW_TTS_ONLY 라서 TTS 단독으로 계속(배선 점검 전용)"
else
    echo "❌ KSPON_AUDIO_ROOT/KSPON_TRN 미지정 — TTS 단독 학습은 금지합니다."
    echo "   (v2 검증: TTS 낭독체만으로는 받침 약화/연음 등 자연 발화 신호가 빠져"
    echo "    실기기 오류가 받침 주변에 집중되는 문제가 해결되지 않음)"
    echo "   KsponSpeech(AIHub 수동 다운로드) 후 두 env 를 지정해 재실행하세요:"
    echo "     KSPON_AUDIO_ROOT=/data/KsponSpeech KSPON_TRN=.../train.trn ./run_vocab_pipeline.sh"
    echo "   배선 점검만 하려면 SMOKE=1, 의식적 예외는 ALLOW_TTS_ONLY=1."
    exit 1
fi

# ── 추가 소스 목록 구성 ────────────────────────────────────
EXTRA=""
EXTRA_MULT=""     # EXTRA 와 같은 순서의 소스별 oversample 목록
if [ -s "$LOANWORD_DIR/train.jsonl" ]; then
    EXTRA="$LOANWORD_DIR"
    EXTRA_MULT="$LOAN_OVERSAMPLE"
fi
if [ -s "$KSPON_DIR/train.jsonl" ]; then
    EXTRA="${EXTRA:+$EXTRA,}$KSPON_DIR"
    EXTRA_MULT="${EXTRA_MULT:+$EXTRA_MULT,}$KSPON_OVERSAMPLE"
fi
[ -n "$EXTRA" ] || { echo "❌ 추가 학습 소스가 하나도 없음 — 중단"; exit 1; }
echo ""; echo "  ➕ 학습 추가 소스: $EXTRA  (oversample: $EXTRA_MULT)"

# ── 평가 헬퍼: 외래어 OOD(clean) + 문단 낭독 + Zeroth 회귀(전 조건) ──
eval_sets () {  # $1=model_path  $2=결과 하위폴더
    local M="$1" TAG="$2"
    if [ -n "$LOAN_EVAL" ]; then
        if [ -n "$TTS_EVAL_BACKEND" ]; then
            echo "  · 외래어 OOD 평가 ($TAG) — 학습 미사용 TTS($TTS_EVAL_BACKEND)"
        else
            echo "  · 외래어 평가 ($TAG) — [경고] 학습 풀과 동일 도메인(진짜 OOD 아님, 임시 — 실기기로 대체 확인)"
        fi
        eval $PY diagnose_farfield_baseline.py --model_path "$M" \
            --json_dir "$LOAN_EVAL" --split test --num_samples $DIAG_N --apply_g2p \
            --snr_list 10 --tail_sec 0 --noise_only_clips 0 --no_competing --no_babble \
            --output_dir "$RESULTS/$TAG/loanword"
        if [ -s "$LOAN_EVAL/test_paragraph.jsonl" ]; then
            echo "  · 문단 낭독 평가 ($TAG) — 연속 운율 장문"
            eval $PY diagnose_farfield_baseline.py --model_path "$M" \
                --json_dir "$LOAN_EVAL" --split test_paragraph --num_samples $DIAG_N --apply_g2p \
                --snr_list 10 --tail_sec 0 --noise_only_clips 0 --no_competing --no_babble \
                --output_dir "$RESULTS/$TAG/paragraph"
        fi
        if [ -s "$LOAN_EVAL/heldout.jsonl" ]; then
            echo "  · 미지-어휘 held-out 평가 ($TAG) — 학습 WORDS 와 disjoint(진짜 일반화)"
            eval $PY diagnose_farfield_baseline.py --model_path "$M" \
                --json_dir "$LOAN_EVAL" --split heldout --num_samples $DIAG_N --apply_g2p \
                --snr_list 10 --tail_sec 0 --noise_only_clips 0 --no_competing --no_babble \
                --output_dir "$RESULTS/$TAG/heldout"
        fi
    fi
    echo "  · Zeroth 회귀 평가 ($TAG) — clean/소음 + 경쟁화자/babble 지표 유지 확인"
    eval $PY diagnose_farfield_baseline.py --model_path "$M" \
        --json_dir "$ZEROTH_DIR" --split test --num_samples $DIAG_N --apply_g2p \
        --tail_sec 8 --noise_only_clips $NOISE_CLIPS \
        --output_dir "$RESULTS/$TAG/zeroth"
}

# ── STEP 0: 재학습 전 기준선 ──────────────────────────────
if [ -f "$BASELINE_MODEL/config.json" ]; then
    echo ""; echo "════════ STEP 0: 재학습 전 기준선 ════════"
    eval_sets "$BASELINE_MODEL" "before"
fi

# ── STEP 1: 어휘 확장 재학습 (배포본 이어서 + 추가 소스 + 증강 전체 유지) ──
echo ""; echo "════════ STEP 1: 어휘 확장 재학습 ════════"
INIT_ARG=""
if [ "$INIT_MODEL" != "scratch" ] && [ -f "$INIT_MODEL/config.json" ]; then
    INIT_ARG="--init_model $INIT_MODEL"
    echo "  ↪️  $INIT_MODEL 에서 이어서 학습 (+어휘 확장)"
else
    echo "  ↪️  openai/whisper-base 에서 처음부터 학습"
fi
# 음향/화자 증강(p_tail/longform/noise_only/competing/babble)은 직전 배포본과
# 동일하게 유지 — 화자 격리 학습 측 방어를 보존한 채 어휘만 더한다.
eval $PY finetune_whisper.py \
    --json_dir "$ZEROTH_DIR" --apply_g2p \
    --extra_json_dirs "$EXTRA" --oversample "$OVERSAMPLE" \
    --extra_oversample "$EXTRA_MULT" \
    --p_tail 0.3 --longform_prob 0.3 --noise_only_prob "${NOISE_ONLY_PROB:-0.05}" \
    --competing_prob "$COMPETING_PROB" --competing_own_rir_prob "$COMPETING_OWN_RIR_PROB" \
    --competing_hard_prob "$COMPETING_HARD_PROB" \
    --babble_prob "$BABBLE_PROB" \
    --batch_size 8 --grad_accum 2 \
    --output_dir "$OUT_DIR" $INIT_ARG $TRAIN_ARGS

[ -f "$NEW_MODEL/config.json" ] || { echo "❌ 재학습 결과 $NEW_MODEL 없음 — 학습 실패"; exit 1; }

# ── STEP 2: 재학습 후 기준선 ──────────────────────────────
echo ""; echo "════════ STEP 2: 재학습 후 기준선 ════════"
eval_sets "$NEW_MODEL" "after"

# ── 결과 안내 ──────────────────────────────────────────────
echo ""; echo "✅ 파이프라인 완료"
echo "  외래어 전/후 (${TTS_EVAL_BACKEND:-학습 풀과 동일, OOD 아님}) : $RESULTS/before/loanword/summary.md   vs  $RESULTS/after/loanword/summary.md"
echo "  문단 낭독 전/후  : $RESULTS/before/paragraph/summary.md  vs  $RESULTS/after/paragraph/summary.md"
echo "  Zeroth 회귀 전/후 : $RESULTS/before/zeroth/summary.md    vs  $RESULTS/after/zeroth/summary.md"
echo "  새 모델          : $NEW_MODEL"
echo ""
echo "배포 게이트(§9, docs/whisper-model-setup.md):"
echo "  1) 외래어·문단 낭독 개선 — TTS_EVAL_BACKEND 미설정 시 학습 풀과 동일 도메인이라 OOD 아님. 반드시 실기기 테스트로 교차 검증 후 배포 판단"
echo "  2) Zeroth clean 회귀 없음"
echo "  3) babble_snr0 / comp_sir0 이 직전 배포본(before/zeroth) 대비 유지"
echo "  판별 케이스: '캡스톤 프로젝트', '데시벨' + 받침 문장(없어요/않아요/보내겠습니다)"
echo "다음: 새 모델을 Mac으로 가져가 ./convert_to_coreml.sh $NEW_MODEL 로 CoreML 변환."