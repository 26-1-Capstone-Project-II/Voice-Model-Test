"""
올라잇 1단계 — 최종 통합 테스트
베이스라인 모델: w11wo/wav2vec2-xls-r-300m-korean

테스트 항목:
  [A] 파일 모드  — gTTS 음성으로 자동 검증
  [B] 마이크 모드 — 직접 말하며 실시간 검증

실행:
    python test_final.py          # A(파일) + B(마이크) 순서로 진행
    python test_final.py --file   # A만
    python test_final.py --mic    # B만
"""

import argparse
import sys
import io
import time
import difflib
import torch
import numpy as np
import librosa
from gtts import gTTS
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# ── 설정 ──────────────────────────────────────────────────
MODEL_ID  = "w11wo/wav2vec2-xls-r-300m-korean"
TARGET_SR = 16000

FILE_TEST_CASES = [
    ("같이 해볼까",          "가치 해 볼까"),   # 연음 — 모델이 발음 전사에 가까울수록 좋음
    ("안녕하세요",           "안녕하세요"),     # 표준 발음
    ("오늘 날씨가 좋네요",    "오늘 날씨가 좋네요"),
    ("저는 잘 들리지 않아요", "저는 잘 들리지 않아요"),
    ("천천히 말해주세요",     "천천히 말해주세요"),
]

MIC_TEST_CASES = [
    "같이 해볼까",
    "안녕하세요 반갑습니다",
    "오늘 날씨가 좋네요",
    "저는 잘 들리지 않아요",
]

C = {
    "green":  "\033[92m", "red":    "\033[91m",
    "yellow": "\033[93m", "blue":   "\033[94m",
    "cyan":   "\033[96m", "bold":   "\033[1m",
    "dim":    "\033[2m",  "reset":  "\033[0m",
}

# ── 유틸 ──────────────────────────────────────────────────
def calc_cer(ref: str, hyp: str) -> float:
    r = list(ref.replace(" ", ""))
    h = list(hyp.replace(" ", ""))
    m, n = len(r), len(h)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = (dp[i-1][j-1] if r[i-1]==h[j-1]
                        else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]))
    return dp[m][n] / max(m, 1)

def syllable_diff(ref: str, hyp: str) -> tuple[str, str]:
    """음절 단위 컬러 diff 두 줄 반환"""
    rs = list(ref.replace(" ", ""))
    hs = list(hyp.replace(" ", ""))
    matcher = difflib.SequenceMatcher(None, rs, hs, autojunk=False)
    ref_vis = "  정답: "
    hyp_vis = "  발음: "
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        r_s = "".join(rs[i1:i2])
        h_s = "".join(hs[j1:j2])
        if tag == "equal":
            ref_vis += C["green"]  + r_s          + C["reset"]
            hyp_vis += C["green"]  + h_s          + C["reset"]
        elif tag == "replace":
            ref_vis += C["red"]    + f"[{r_s}]"   + C["reset"]
            hyp_vis += C["yellow"] + f"[{h_s}]"   + C["reset"]
        elif tag == "delete":
            ref_vis += C["red"]    + f"[{r_s}↓]"  + C["reset"]
            hyp_vis += C["dim"]    + "[·]"         + C["reset"]
        elif tag == "insert":
            ref_vis += C["dim"]    + "[·]"         + C["reset"]
            hyp_vis += C["yellow"] + f"[+{h_s}]"  + C["reset"]
    return ref_vis, hyp_vis

def print_single_result(ref: str, hyp: str):
    cer   = calc_cer(ref, hyp)
    score = max(0.0, round((1 - cer) * 100, 1))
    sc    = C["green"] if score >= 80 else C["yellow"] if score >= 50 else C["red"]
    bar   = "█" * int(score/5) + "░" * (20 - int(score/5))
    icon  = "✅" if score >= 80 else "🔍" if score >= 50 else "❌"

    ref_vis, hyp_vis = syllable_diff(ref, hyp)
    print(f"  {icon} 점수: {sc}{C['bold']}{score}점{C['reset']}  [{sc}{bar}{C['reset']}]")
    print(ref_vis)
    print(hyp_vis)

# ── 모델 로드 ──────────────────────────────────────────────
def load():
    print(f"\n{C['bold']}[모델 로드]{C['reset']} {MODEL_ID}")
    print("  (이미 다운로드됐다면 캐시에서 즉시 로드)")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model     = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model.eval()
    device = ("mps"  if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available()          else "cpu")
    model  = model.to(device)
    print(f"  ✅ 로드 완료  (device: {device})\n")
    return processor, model, device

def transcribe(audio: np.ndarray, processor, model, device) -> str:
    inputs = processor(audio, sampling_rate=TARGET_SR,
                       return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to(device)).logits
    ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(ids)[0].strip()

# ── [A] 파일 모드 ──────────────────────────────────────────
def test_file(processor, model, device):
    print(f"{'═'*56}")
    print(f"  {C['bold']}[A] 파일 모드 테스트  (gTTS 자동 생성){C['reset']}")
    print(f"{'═'*56}\n")

    scores = []
    for ref, expected_hyp in FILE_TEST_CASES:
        print(f"  ▶ 문장: {C['blue']}{ref}{C['reset']}")
        print(f"    예상 인식: {C['dim']}{expected_hyp}{C['reset']}")

        # gTTS 생성
        buf = io.BytesIO()
        gTTS(text=ref, lang="ko").write_to_fp(buf)
        buf.seek(0)
        audio, _  = librosa.load(buf, sr=TARGET_SR, mono=True)
        trimmed,_ = librosa.effects.trim(audio, top_db=25)

        result = transcribe(trimmed, processor, model, device)
        print_single_result(ref, result)

        cer   = calc_cer(ref, result)
        score = max(0.0, round((1-cer)*100, 1))
        scores.append(score)
        print()

    avg = sum(scores) / len(scores)
    sc  = C["green"] if avg >= 70 else C["yellow"] if avg >= 50 else C["red"]
    print(f"  📊 파일 모드 평균: {sc}{C['bold']}{avg:.1f}점{C['reset']}\n")
    return avg

# ── [B] 마이크 모드 ────────────────────────────────────────
def test_mic(processor, model, device, duration: int = 5):
    try:
        import sounddevice as sd
    except ImportError:
        print("  ❌ pip install sounddevice scipy 후 재실행")
        return

    print(f"{'═'*56}")
    print(f"  {C['bold']}[B] 마이크 모드 테스트  (직접 발화){C['reset']}")
    print(f"  각 문장을 {duration}초 안에 읽어주세요")
    print(f"{'═'*56}\n")

    scores = []
    for i, ref in enumerate(MIC_TEST_CASES, 1):
        print(f"  [{i}/{len(MIC_TEST_CASES)}] {C['cyan']}「 {ref} 」{C['reset']}")
        cmd = input("       Enter=녹음시작  s=건너뛰기  q=종료 > ").strip().lower()
        if cmd == "q": break
        if cmd == "s": print("       ⏭ 건너뜀\n"); continue

        print(f"  🔴 녹음 중...", end="", flush=True)
        audio = sd.rec(int(duration * TARGET_SR), samplerate=TARGET_SR,
                       channels=1, dtype="float32")
        for i in range(duration, 0, -1):
            time.sleep(1)
            print(f"\r  🔴 {i-1}초 남음...  ", end="", flush=True)
        sd.wait()
        print(f"\r  ✅ 녹음 완료!           ")

        audio     = audio.flatten()
        trimmed,_ = librosa.effects.trim(audio, top_db=25)
        result    = transcribe(trimmed, processor, model, device)

        print_single_result(ref, result)
        cer   = calc_cer(ref, result)
        score = max(0.0, round((1-cer)*100, 1))
        scores.append(score)
        print()

    if scores:
        avg = sum(scores) / len(scores)
        sc  = C["green"] if avg >= 70 else C["yellow"] if avg >= 50 else C["red"]
        print(f"  📊 마이크 모드 평균: {sc}{C['bold']}{avg:.1f}점{C['reset']}\n")

# ── 메인 ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="올라잇 1단계 최종 통합 테스트")
    parser.add_argument("--file", action="store_true", help="파일 모드만 실행")
    parser.add_argument("--mic",  action="store_true", help="마이크 모드만 실행")
    parser.add_argument("--duration", type=int, default=5)
    args = parser.parse_args()

    run_file = args.file or (not args.file and not args.mic)
    run_mic  = args.mic  or (not args.file and not args.mic)

    processor, model, device = load()

    file_avg = None
    if run_file:
        file_avg = test_file(processor, model, device)

    if run_mic:
        test_mic(processor, model, device, args.duration)

    # 최종 요약
    print(f"{'═'*56}")
    print(f"  {C['bold']}📋 1단계 테스트 완료 요약{C['reset']}")
    print(f"{'═'*56}")
    print(f"  모델    : {MODEL_ID}")
    print(f"  방식    : CTC + Greedy Decoding (LM 없음)")
    if file_avg is not None:
        sc = C["green"] if file_avg >= 70 else C["yellow"]
        print(f"  파일 avg: {sc}{file_avg:.1f}점{C['reset']}")
    print(f"\n  → 1단계 베이스라인 확정 ✅")
    print(f"  → 다음: 2단계 파인튜닝으로 발음 전사 정밀도 향상")
    print(f"{'═'*56}")

if __name__ == "__main__":
    main()