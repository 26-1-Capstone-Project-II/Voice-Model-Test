"""
test_jamo_model.py — 자모 파인튜닝 모델 검증 스크립트
=====================================================
파인튜닝된 모델이 "소리나는 대로" 자모 단위로 출력하는지 검증.

테스트 방식:
  1. gTTS로 표준 발음 음성 생성
  2. 파인튜닝 모델로 추론 → 자모 출력
  3. 자모 → 음절 재조립
  4. G2P 정답과 비교 → CER 측정

실행:
    # 파인튜닝 모델 테스트
    python test_jamo_model.py --model ./best_model_jamo/best

    # 베이스라인과 비교
    python test_jamo_model.py --model ./best_model_jamo/best --compare

    # gTTS 없이 오디오 파일로 테스트
    python test_jamo_model.py --model ./best_model_jamo/best --audio test.wav --text "같이 해볼까"
"""

import argparse
import sys
import io
import time
import torch
import numpy as np
from pathlib import Path

# 🚨 서버 GPU cuDNN 초기화 실패 방지
torch.backends.cudnn.enabled = False

# ── 설정 ──────────────────────────────────────
TARGET_SR = 16000

TEST_CASES = [
    # (문장, G2P 발음 전사 기대값)
    ("같이 해볼까",       "가치 해볼까"),
    ("좋네요",            "존네요"),
    ("먹어요",            "머거요"),
    ("안녕하세요",        "안녕하세요"),
    ("오늘 날씨가 좋네요", "오늘 날씨가 존네요"),
    ("천천히 말해주세요",  "천천히 말해주세요"),
    ("저는 잘 들리지 않아요", "저는 잘 들리지 않아요"),
]

C = {
    "green":  "\033[92m", "red":    "\033[91m",
    "yellow": "\033[93m", "blue":   "\033[94m",
    "cyan":   "\033[96m", "bold":   "\033[1m",
    "dim":    "\033[2m",  "reset":  "\033[0m",
}


# ── CER 계산 ──────────────────────────────────
def calc_cer(ref: str, hyp: str) -> float:
    """Edit distance 기반 CER."""
    r = list(ref.replace(" ", ""))
    h = list(hyp.replace(" ", ""))
    m, n = len(r), len(h)
    if m == 0:
        return 1.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = (dp[i - 1][j - 1] if r[i - 1] == h[j - 1]
                        else 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]))
    return dp[m][n] / max(m, 1)


# ── 자모 모델 로드 ────────────────────────────
def load_jamo_model(model_path: str):
    """자모 vocab 파인튜닝 모델 로드."""
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    print(f"\n{C['bold']}[모델 로드]{C['reset']} {model_path}")
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    model.eval()

    device = (
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)

    vocab_size = processor.tokenizer.vocab_size
    print(f"  ✅ 로드 완료 (device: {device}, vocab: {vocab_size}개)")

    return processor, model, device


def load_baseline_model():
    """베이스라인 모델 (음절 vocab) 로드."""
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    model_id = "w11wo/wav2vec2-xls-r-300m-korean"
    print(f"\n{C['bold']}[베이스라인 로드]{C['reset']} {model_id}")
    processor = Wav2Vec2Processor.from_pretrained(model_id)
    model = Wav2Vec2ForCTC.from_pretrained(model_id)
    model.eval()

    device = (
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = model.to(device)
    print(f"  ✅ 로드 완료 (device: {device}, vocab: {processor.tokenizer.vocab_size}개)")

    return processor, model, device


# ── 추론 ──────────────────────────────────────
def transcribe(audio, processor, model, device) -> str:
    """CTC Greedy Decoding."""
    inputs = processor(audio, sampling_rate=TARGET_SR,
                       return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
    ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(ids)[0].strip()


# ── gTTS 음성 생성 ────────────────────────────
def generate_audio(text: str):
    """gTTS로 표준 발음 음성 생성."""
    import librosa
    from gtts import gTTS

    buf = io.BytesIO()
    gTTS(text=text, lang="ko").write_to_fp(buf)
    buf.seek(0)
    audio, _ = librosa.load(buf, sr=TARGET_SR, mono=True)

    # 앞뒤 무음 제거
    trimmed, _ = librosa.effects.trim(audio, top_db=25)
    return trimmed


# ── 단일 테스트 ───────────────────────────────
def test_single(text, expected, processor, model, device, is_jamo=False):
    """단일 문장 테스트."""
    from jamo_utils import jamo_to_syllable

    audio = generate_audio(text)
    raw_output = transcribe(audio, processor, model, device)

    if is_jamo:
        # 자모 모델: 출력을 음절로 변환
        syllable_output = jamo_to_syllable(raw_output)
    else:
        # 베이스라인: 이미 음절 출력
        syllable_output = raw_output

    cer = calc_cer(expected, syllable_output)
    score = max(0.0, round((1 - cer) * 100, 1))

    return {
        "text": text,
        "expected": expected,
        "raw_output": raw_output,
        "syllable_output": syllable_output,
        "cer": cer,
        "score": score,
    }


# ── 전체 테스트 ───────────────────────────────
def run_test(model_path: str, compare: bool = False):
    """전체 테스트 실행."""
    # 자모 모델 테스트
    processor, model, device = load_jamo_model(model_path)

    print(f"\n{'═' * 60}")
    print(f"  {C['bold']}🔬 자모 모델 테스트 (gTTS 자동 생성){C['reset']}")
    print(f"{'═' * 60}\n")

    jamo_results = []
    for text, expected in TEST_CASES:
        r = test_single(text, expected, processor, model, device, is_jamo=True)
        jamo_results.append(r)

        sc = C["green"] if r["score"] >= 80 else C["yellow"] if r["score"] >= 50 else C["red"]
        icon = "✅" if r["score"] >= 80 else "🔍" if r["score"] >= 50 else "❌"

        print(f"  ▶ 문장: {C['blue']}{text}{C['reset']}")
        print(f"    기대: {C['cyan']}{expected}{C['reset']}")
        print(f"    자모: {C['dim']}{r['raw_output'][:50]}...{C['reset']}")
        print(f"    복원: {r['syllable_output']}")
        print(f"    {icon} 점수: {sc}{C['bold']}{r['score']}점{C['reset']}  (CER {r['cer']:.3f})")
        print()

    jamo_avg = sum(r["score"] for r in jamo_results) / len(jamo_results)
    jamo_cer = sum(r["cer"] for r in jamo_results) / len(jamo_results)

    # 베이스라인 비교
    if compare:
        base_processor, base_model, base_device = load_baseline_model()

        print(f"\n{'═' * 60}")
        print(f"  {C['bold']}📊 베이스라인 비교{C['reset']}")
        print(f"{'═' * 60}\n")

        base_results = []
        for text, expected in TEST_CASES:
            r = test_single(text, expected, base_processor, base_model, base_device, is_jamo=False)
            base_results.append(r)

            sc = C["green"] if r["score"] >= 80 else C["yellow"] if r["score"] >= 50 else C["red"]
            icon = "✅" if r["score"] >= 80 else "🔍" if r["score"] >= 50 else "❌"

            print(f"  ▶ {C['blue']}{text}{C['reset']}  →  {r['syllable_output']}")
            print(f"    {icon} {sc}{r['score']}점{C['reset']}")

        base_avg = sum(r["score"] for r in base_results) / len(base_results)
        base_cer = sum(r["cer"] for r in base_results) / len(base_results)

        # 비교 요약
        print(f"\n{'═' * 60}")
        print(f"  {C['bold']}📋 비교 요약{C['reset']}")
        print(f"{'═' * 60}")
        print(f"  {'':18} {'평균 점수':>10} {'평균 CER':>10}")
        print(f"  {'-' * 40}")

        jsc = C["green"] if jamo_avg >= 70 else C["yellow"]
        bsc = C["green"] if base_avg >= 70 else C["yellow"]
        print(f"  {'자모 파인튜닝':16} {jsc}{jamo_avg:>8.1f}점{C['reset']}  {jamo_cer:>8.3f}")
        print(f"  {'베이스라인':16} {bsc}{base_avg:>8.1f}점{C['reset']}  {base_cer:>8.3f}")

        diff = jamo_avg - base_avg
        icon = "📈" if diff > 0 else "📉"
        print(f"\n  {icon} 개선: {diff:+.1f}점")
    else:
        # 자모 모델만 요약
        print(f"{'═' * 60}")
        sc = C["green"] if jamo_avg >= 70 else C["yellow"] if jamo_avg >= 50 else C["red"]
        print(f"  {C['bold']}📋 자모 모델 요약{C['reset']}")
        print(f"  평균 점수: {sc}{jamo_avg:.1f}점{C['reset']}  (CER {jamo_cer:.3f})")
        print(f"  목표 CER: < 0.3 {'✅ 달성!' if jamo_cer < 0.3 else '⚠️ 미달성'}")

    print(f"{'═' * 60}\n")


# ── 단일 파일 테스트 ──────────────────────────
def test_audio_file(model_path: str, audio_path: str, text: str):
    """오디오 파일로 직접 테스트."""
    import librosa
    from jamo_utils import jamo_to_syllable, syllable_to_jamo
    from korean_g2p_nomecab import load_g2p

    processor, model, device = load_jamo_model(model_path)

    audio, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    raw_output = transcribe(audio, processor, model, device)
    syllable_output = jamo_to_syllable(raw_output)

    # G2P 정답
    g2p = load_g2p()
    expected = g2p(text, descriptive=True)

    cer = calc_cer(expected, syllable_output)
    score = max(0.0, round((1 - cer) * 100, 1))

    print(f"\n{'═' * 56}")
    print(f"  {C['bold']}📊 발음 분석 결과 (자모 모델){C['reset']}")
    print(f"{'═' * 56}")
    print(f"  목표 문장  :  {C['blue']}{text}{C['reset']}")
    print(f"  발음 기준  :  {C['cyan']}{expected}{C['reset']}")
    print(f"  자모 출력  :  {C['dim']}{raw_output}{C['reset']}")
    print(f"  음절 복원  :  {syllable_output}")
    print(f"  점수       :  {score}점  (CER {cer:.3f})")
    print(f"{'═' * 56}\n")


# ── 메인 ──────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="자모 파인튜닝 모델 검증",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True,
                        help="파인튜닝 모델 경로 (예: ./best_model_jamo/best)")
    parser.add_argument("--compare", action="store_true",
                        help="베이스라인 모델과 비교")
    parser.add_argument("--audio", type=str, default=None,
                        help="테스트할 오디오 파일 경로")
    parser.add_argument("--text", type=str, default=None,
                        help="목표 문장 (--audio 와 함께 사용)")
    args = parser.parse_args()

    if args.audio:
        if not args.text:
            parser.error("--audio 사용 시 --text 도 지정하세요.")
        test_audio_file(args.model, args.audio, args.text)
    else:
        run_test(args.model, compare=args.compare)


if __name__ == "__main__":
    main()
