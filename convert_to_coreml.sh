#!/usr/bin/env bash
# ============================================================
# whisper-base 파인튜닝 모델 → CoreML 변환 → 앱 번들(Whisper_CoreML_Model) 교체
# ============================================================
# 재학습 후 한 번에: WhisperKit 변환 + .mlmodelc 3종을 앱 폴더로 복사한다.
# (Mac 에서 실행. wkt 가상환경에 whisperkit/coremltools 설치돼 있어야 함)
#
# 사용:
#   ./convert_to_coreml.sh [MODEL_DIR] [OUT_DIR] [DEST_DIR]
#
# 기본값:
#   MODEL_DIR = best_model_zeroth_aug/best   (HF 체크포인트: config.json/model.safetensors 포함)
#   OUT_DIR   = coreml_out                   (변환 중간 산출물, .gitignore 됨)
#   DEST_DIR  = Whisper_CoreML_Model         (앱이 로드하는 폴더)
#
# 검증된 변환 품질 기준: PSNR 40+ (torch2coreml). 미만이면 변환 이상.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODEL_DIR="${1:-best_model_zeroth_aug/best}"
OUT_DIR="${2:-coreml_out}"
DEST_DIR="${3:-Whisper_CoreML_Model}"
GEN="wkt/bin/whisperkit-generate-model"

# ── 사전 점검 ──────────────────────────────────────────────
[ -f "$MODEL_DIR/config.json" ] || { echo "❌ $MODEL_DIR/config.json 없음 (HF 체크포인트 경로 확인)"; exit 1; }
[ -x "$GEN" ] || { echo "❌ $GEN 없음 (wkt 가상환경/whisperkit 설치 확인)"; exit 1; }

echo "🎛️  변환 모델 : $MODEL_DIR"
echo "📦 출력      : $OUT_DIR"
echo "🍏 앱 번들   : $DEST_DIR"

# ── 1) 변환 (테스트=변환이므로 --disable-default-tests 쓰지 말 것) ──
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
echo "🔄 WhisperKit CoreML 변환 중... (수 분 소요)"
"$GEN" --model-version "$MODEL_DIR" --output-dir "$OUT_DIR"

# ── 2) 산출물 위치 탐색 (whisperkit 은 model-version 이름으로 하위폴더 생성) ──
SRC="$(dirname "$(find "$OUT_DIR" -type d -name 'AudioEncoder.mlmodelc' | head -1)")"
[ -n "$SRC" ] && [ -d "$SRC" ] || { echo "❌ 변환 산출물(AudioEncoder.mlmodelc)을 못 찾음"; exit 1; }
echo "📂 산출물 위치: $SRC"

# 3종 mlmodelc 모두 존재 확인
for m in AudioEncoder MelSpectrogram TextDecoder; do
    [ -d "$SRC/$m.mlmodelc" ] || { echo "❌ $m.mlmodelc 누락 — 변환 실패"; exit 1; }
done

# ── 3) 앱 번들 폴더로 교체 복사 ──
mkdir -p "$DEST_DIR"
rm -rf "$DEST_DIR"/*.mlmodelc "$DEST_DIR"/*.mlcomputeplan.json
cp -R "$SRC"/AudioEncoder.mlmodelc "$SRC"/MelSpectrogram.mlmodelc "$SRC"/TextDecoder.mlmodelc "$DEST_DIR"/
# compute-plan json 은 있으면 함께 복사(없어도 앱 동작엔 무관)
cp "$SRC"/*.mlcomputeplan.json "$DEST_DIR"/ 2>/dev/null || true

echo ""
echo "✅ 완료 — $DEST_DIR 갱신됨:"
du -sh "$DEST_DIR"/*.mlmodelc
echo ""
echo "다음: Xcode 에서 기존 $DEST_DIR 를 빼고 갱신된 폴더를 다시 드래그(Create groups)."
echo "      git LFS 로 .mlmodelc 가중치가 추적됩니다. (필요 시 git add $DEST_DIR)"
