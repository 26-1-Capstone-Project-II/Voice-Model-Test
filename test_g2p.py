"""
올라잇 1단계 — G2P 기반 발음 전사 재점수
핵심 변경: 정답 기준을 맞춤법 → 표준 발음 전사로 교체

흐름:
  입력 문장 ("같이 해볼까")
      ↓ g2pk
  발음 전사 ("가치 해볼까")   ← 새로운 정답 기준
      ↓ 모델 인식
  인식 결과 ("가치 해볼까")
      ↓ diff
  점수 / 오류 위치

실행:
    python test_g2p.py
"""

import io
import difflib
import torch
import numpy as np
import librosa
from gtts import gTTS
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# ── 설정 ──────────────────────────────────────────────────
MODEL_ID  = "w11wo/wav2vec2-xls-r-300m-korean"
TARGET_SR = 16000

C = {
    "green":  "\033[92m", "red":    "\033[91m",
    "yellow": "\033[93m", "blue":   "\033[94m",
    "cyan":   "\033[96m", "bold":   "\033[1m",
    "dim":    "\033[2m",  "reset":  "\033[0m",
}

# ── G2P 로드 ───────────────────────────────────────────────
def load_g2p():
    try:
        from g2pk import G2p
        g2p = G2p()
        print("  ✅ g2pk 로드 완료")
        return g2p
    except ImportError:
        print("  ❌ g2pk 미설치: pip install g2pk")
        import sys; sys.exit(1)

def to_phonetic(text: str, g2p) -> str:
    """
    맞춤법 표기 → 표준 발음 전사
    예) "같이 해볼까" → "가치 해볼까"
        "닭볶음"     → "닥뽁끔"
        "좋네요"     → "존네요"
    """
    return g2p(text)

# ── 모델 로드 ──────────────────────────────────────────────
def load_model():
    print(f"\n{C['bold']}[모델 로드]{C['reset']} {MODEL_ID}")
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

# ── CER + Diff ─────────────────────────────────────────────
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

def syllable_diff_vis(ref: str, hyp: str) -> tuple[str, str]:
    rs = list(ref.replace(" ", ""))
    hs = list(hyp.replace(" ", ""))
    matcher = difflib.SequenceMatcher(None, rs, hs, autojunk=False)
    ref_vis = "      발음 정답: "
    hyp_vis = "      모델 인식: "
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        r_s = "".join(rs[i1:i2])
        h_s = "".join(hs[j1:j2])
        if tag == "equal":
            ref_vis += C["green"]  + r_s         + C["reset"]
            hyp_vis += C["green"]  + h_s         + C["reset"]
        elif tag == "replace":
            ref_vis += C["red"]    + f"[{r_s}]"  + C["reset"]
            hyp_vis += C["yellow"] + f"[{h_s}]"  + C["reset"]
        elif tag == "delete":
            ref_vis += C["red"]    + f"[{r_s}↓]" + C["reset"]
            hyp_vis += C["dim"]    + "[·]"        + C["reset"]
        elif tag == "insert":
            ref_vis += C["dim"]    + "[·]"        + C["reset"]
            hyp_vis += C["yellow"] + f"[+{h_s}]" + C["reset"]
    return ref_vis, hyp_vis

def _grade(score: float) -> str:
    if score >= 95: return "🏆 완벽해요!"
    if score >= 85: return "🌟 훌륭해요!"
    if score >= 70: return "👍 잘 하셨어요"
    if score >= 50: return "💬 조금 더 연습해봐요"
    return "💪 함께 연습해봐요"

# ── 테스트 케이스 ──────────────────────────────────────────
TEST_SENTENCES = [
    "같이 해볼까",
    "안녕하세요",
    "오늘 날씨가 좋네요",
    "저는 잘 들리지 않아요",
    "천천히 말해주세요",
    "닭볶음이 맛있어요",       # 연음/경음화 테스트
    "국밥 한 그릇 주세요",     # 받침 연음 테스트
]

# ── 메인 ──────────────────────────────────────────────────
def main():
    print(f"{'═'*60}")
    print(f"  {C['bold']}올라잇 1단계 — G2P 기반 발음 전사 재점수{C['reset']}")
    print(f"  정답 기준: 맞춤법 표기 → 표준 발음 전사 (g2pk)")
    print(f"{'═'*60}\n")

    # 로드
    g2p = load_g2p()
    processor, model, device = load_model()

    # G2P 변환 미리보기
    print(f"  {C['bold']}[G2P 변환 미리보기]{C['reset']}")
    print(f"  {'맞춤법':<18} {'발음 전사(정답 기준)'}")
    print(f"  {'─'*18} {'─'*20}")
    for s in TEST_SENTENCES:
        phonetic = to_phonetic(s, g2p)
        changed  = "← 변환됨" if phonetic != s else ""
        print(f"  {s:<18} {C['cyan']}{phonetic}{C['reset']}  {C['dim']}{changed}{C['reset']}")
    print()

    # 테스트 실행
    print(f"  {C['bold']}[테스트 실행]{C['reset']}\n")

    old_scores = []  # 기존 방식 (맞춤법 기준)
    new_scores = []  # 새 방식 (발음 전사 기준)

    for sentence in TEST_SENTENCES:
        phonetic = to_phonetic(sentence, g2p)

        # gTTS 음성 생성
        buf = io.BytesIO()
        gTTS(text=sentence, lang="ko").write_to_fp(buf)
        buf.seek(0)
        audio, _   = librosa.load(buf, sr=TARGET_SR, mono=True)
        trimmed, _ = librosa.effects.trim(audio, top_db=25)

        # 모델 인식
        result = transcribe(trimmed, processor, model, device)

        # 점수 계산 (두 가지 기준)
        cer_old  = calc_cer(sentence, result)   # 맞춤법 기준
        cer_new  = calc_cer(phonetic, result)    # 발음 전사 기준
        score_old = max(0.0, round((1 - cer_old) * 100, 1))
        score_new = max(0.0, round((1 - cer_new) * 100, 1))
        old_scores.append(score_old)
        new_scores.append(score_new)

        # 점수 변화
        delta = score_new - score_old
        delta_str = (f"{C['green']}+{delta:.0f}{C['reset']}" if delta > 0
                     else f"{C['dim']}{delta:.0f}{C['reset']}" if delta < 0
                     else f"{C['dim']}±0{C['reset']}")

        sc = C["green"] if score_new >= 80 else C["yellow"] if score_new >= 50 else C["red"]
        icon = "✅" if score_new >= 80 else "🔍" if score_new >= 50 else "❌"

        print(f"  {icon} {C['blue']}{sentence}{C['reset']}")
        print(f"      맞춤법 기준: {score_old:.0f}점  →  발음 기준: "
              f"{sc}{C['bold']}{score_new:.0f}점{C['reset']}  ({delta_str})")
        print(f"      모델 출력:   {C['dim']}{result}{C['reset']}")

        ref_vis, hyp_vis = syllable_diff_vis(phonetic, result)
        print(ref_vis)
        print(hyp_vis)
        print()

    # 최종 요약
    avg_old = sum(old_scores) / len(old_scores)
    avg_new = sum(new_scores) / len(new_scores)
    avg_delta = avg_new - avg_old

    print(f"{'═'*60}")
    print(f"  {C['bold']}📊 최종 요약{C['reset']}")
    print(f"{'═'*60}")
    print(f"  기존 (맞춤법 기준) 평균  : {avg_old:.1f}점")
    sc = C["green"] if avg_new >= 70 else C["yellow"]
    print(f"  신규 (발음 전사 기준) 평균: {sc}{C['bold']}{avg_new:.1f}점{C['reset']}  "
          f"({C['green']}+{avg_delta:.1f}{C['reset']})")
    print()
    print(f"  {C['bold']}[문장별 비교]{C['reset']}")
    for i, s in enumerate(TEST_SENTENCES):
        phonetic = to_phonetic(s, g2p)
        bar_o = "█" * int(old_scores[i]/10) + "░" * (10 - int(old_scores[i]/10))
        bar_n = "█" * int(new_scores[i]/10) + "░" * (10 - int(new_scores[i]/10))
        sc = C["green"] if new_scores[i] >= 80 else C["yellow"] if new_scores[i] >= 50 else C["red"]
        print(f"  {s}")
        print(f"    맞춤법: [{bar_o}] {old_scores[i]:.0f}점")
        print(f"    발  음: [{sc}{bar_n}{C['reset']}] {sc}{new_scores[i]:.0f}점{C['reset']}  "
              f"{_grade(new_scores[i])}")
    print()
    print(f"  → {C['bold']}발음 전사 기준이 올라잇의 올바른 점수 기준!{C['reset']}")
    print(f"  → pronunciation_scorer.py 에 g2pk 통합 예정")
    print(f"{'═'*60}")

if __name__ == "__main__":
    main()
