"""
온보이스 발음 교정 — 1단계 완성본
CTC + Greedy Decoding (LM 없음) + G2P 발음 전사 기반 채점

[베이스라인 모델] w11wo/wav2vec2-xls-r-300m-korean
  - 3종 모델 비교 테스트 결과 채택
  - "같이 → 가치" 발음 전사에 가장 근접한 인식

[G2P 채점 방식]
  입력 문장 ("같이 해볼까")
      ↓ g2pk (descriptive=True — 과잉 변환 방지)
  발음 전사 ("가치 해볼까")  ← 정답 기준
      ↓ 모델 인식 결과와 비교
  점수 / 음절 오류 위치

[연습 모드 흐름]
  문장 입력 → TTS 재생 (듣기) → 녹음 → 분석 및 평가
  r 입력으로 TTS 다시 듣기 가능

사용법:
    # ★ 연습 모드: 문장 듣고 → 따라 말하기 → 분석 (핵심 기능)
    python pronunciation_scorer.py --practice

    # 마이크 녹음 후 분석 (5초)
    python pronunciation_scorer.py --text "같이 해볼까" --mic

    # 오디오 파일 분석
    python pronunciation_scorer.py --text "같이 해볼까" --audio my_voice.wav

    # 배치 모드: 여러 문장 연속 연습 (TTS 포함)
    python pronunciation_scorer.py --batch

    # 녹음 저장 옵션
    python pronunciation_scorer.py --text "안녕하세요" --mic --save recording.wav

    # G2P 변환 결과 미리보기
    python pronunciation_scorer.py --text "같이 해볼까" --preview
"""

import argparse
import sys
import os
import difflib
import json
from pathlib import Path

# ═══════════════════════════════════════════════════════
# 0. 의존성 체크
# ═══════════════════════════════════════════════════════

REQUIRED = {
    "torch":        "torch",
    "transformers": "transformers",
    "librosa":      "librosa",
    "jiwer":        "jiwer",
    "numpy":        "numpy",
    "kss":          "kss",        # G2P 발음 전사 — MeCab 불필요, 크로스플랫폼
}
MIC_REQUIRED = {
    "sounddevice": "sounddevice",
    "scipy":       "scipy",
}

def check_dependencies(need_mic: bool = False):
    missing = []
    pkgs = {**REQUIRED, **(MIC_REQUIRED if need_mic else {})}
    for pkg, pip_name in pkgs.items():
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print(f"\n[❌ 패키지 설치 필요]\n    pip install {' '.join(missing)}")
        sys.exit(1)

# ═══════════════════════════════════════════════════════
# 1. 모델 로드 (싱글톤 캐시 + fallback)
# ═══════════════════════════════════════════════════════

import torch
import numpy as np

_model_cache: dict = {}

# ═══════════════════════════════════════════════════════
# 1-b. G2P: 맞춤법 → 표준 발음 전사
# ═══════════════════════════════════════════════════════

_g2p_instance = None

def get_g2p():
    """
    kss G2P 싱글톤.

    g2pk/g2pk2/g2pkk → Windows에서 MeCab/eunjeon C++ 빌드 오류 발생
    kss              → pecab 백엔드 사용, MeCab 불필요, 크로스플랫폼 ✅
    """
    global _g2p_instance
    if _g2p_instance is None:
        from kss import Kss
        _g2p_instance = Kss("g2p")
    return _g2p_instance

def to_phonetic(text: str) -> str:
    """
    맞춤법 표기 → 표준 발음 전사 (정답 기준 생성)

    kss g2p 변환 예시:
      "같이 해볼까" → "가치 해볼까"  (연음)
      "좋네요"     → "존네요"        (비음화)
      "않아요"     → "않아요"        (과잉 변환 없음)
    """
    g2p = get_g2p()
    return g2p(text)

# ── 한국어 CTC 모델 목록 ──────────────────────────────
# 3종 비교 테스트 결과 (2025.03 기준):
#   1위 w11wo/wav2vec2-xls-r-300m-korean  69.0점 — "같이→가치" 발음전사 근접, 긴 문장 100점
#   2위 kresnik/wav2vec2-large-xlsr-korean 69.3점 — 짧은 문장 안정적이나 발음전사 부정확
#   탈락 Kkonjeong/wav2vec2-base-korean    자모분리 출력 → 음절 CER 계산 불가
MODEL_OPTIONS = [
    "w11wo/wav2vec2-xls-r-300m-korean",    # 1순위: 발음 전사 최적 (베이스라인 채택)
    "kresnik/wav2vec2-large-xlsr-korean",  # 2순위: fallback
]

def load_model(model_id: str = MODEL_OPTIONS[0]):
    if model_id in _model_cache:
        return _model_cache[model_id]

    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    print(f"\n[모델 로드 중] {model_id}")
    print("  ※ 첫 실행 시 ~1.3GB 자동 다운로드 (이후 캐시 사용)")

    try:
        processor = Wav2Vec2Processor.from_pretrained(model_id)
        model     = Wav2Vec2ForCTC.from_pretrained(model_id)
        model.eval()

        # 가속 디바이스 자동 감지: Apple Silicon > CUDA > CPU
        device = (
            "mps"  if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available()          else
            "cpu"
        )
        model = model.to(device)
        print(f"  ✅ 로드 완료  (device: {device})")
        _model_cache[model_id] = (processor, model, device)
        return processor, model, device

    except Exception as e:
        if model_id == MODEL_OPTIONS[0]:
            print(f"  ⚠️  실패: {e}")
            print(f"  → fallback 모델 시도: {MODEL_OPTIONS[1]}")
            return load_model(MODEL_OPTIONS[1])
        raise

# ═══════════════════════════════════════════════════════
# 2. 오디오 입력
# ═══════════════════════════════════════════════════════

TARGET_SR = 16_000  # wav2vec2 요구 샘플레이트

def load_audio_file(path: str) -> np.ndarray:
    import librosa
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"파일 없음: {p}")
    audio, _ = librosa.load(str(p), sr=TARGET_SR, mono=True)
    print(f"  📁 파일: {p.name}  ({len(audio)/TARGET_SR:.1f}초)")
    return audio


def record_from_mic(duration: int = 5, save_path: str | None = None) -> np.ndarray:
    try:
        import sounddevice as sd
    except ImportError:
        print("[❌] pip install sounddevice scipy")
        sys.exit(1)

    print(f"\n  🎤  준비되면 Enter 키를 누르세요 ({duration}초 녹음)...")
    input()
    print(f"  ● 녹음 중...", end="", flush=True)

    audio = sd.rec(int(duration * TARGET_SR), samplerate=TARGET_SR,
                   channels=1, dtype="float32")
    import time
    for i in range(duration, 0, -1):
        time.sleep(1)
        print(f"\r  ● 녹음 중... {i-1}초 남음  ", end="", flush=True)
    sd.wait()
    print("\r  ✅ 녹음 완료!                      ")
    audio = audio.flatten()

    if save_path:
        _save_wav(audio, save_path)
        print(f"  💾 저장됨: {save_path}")

    return audio


def _save_wav(audio: np.ndarray, path: str):
    from scipy.io.wavfile import write as wav_write
    wav_write(path, TARGET_SR, (audio * 32767).astype(np.int16))

# ═══════════════════════════════════════════════════════
# 2-c. 오디오 전처리 (정확도 향상)
# ═══════════════════════════════════════════════════════

def preprocess_audio(audio: np.ndarray, verbose: bool = True) -> np.ndarray:
    """
    모델 입력 전 음성 품질 개선 파이프라인.

    처리 순서:
      1. 무음 체크  — 너무 조용하면 경고
      2. 앞뒤 무음 제거 (librosa.trim)
      3. 노이즈 제거 (noisereduce) — 설치된 경우에만
      4. RMS 음량 정규화 — 일정한 입력 레벨 보장

    정확도 향상 효과:
      - 무음 제거: 모델이 침묵을 이상한 소리로 해석하는 현상 방지
      - 노이즈 제거: 배경 소음이 자음 인식을 방해하는 현상 방지
      - 음량 정규화: 너무 작게 말해도 안정적으로 인식
    """
    import librosa

    original_len = len(audio)

    # ── 1. 무음 체크 ─────────────────────────────────────
    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < 0.01:
        if verbose:
            print(f"  ⚠️  음성이 너무 작습니다 (RMS: {rms:.4f}) — 마이크를 가까이 하세요")

    # ── 2. 앞뒤 무음 제거 ────────────────────────────────
    trimmed, _ = librosa.effects.trim(audio, top_db=25)
    if len(trimmed) < TARGET_SR * 0.3:
        # 너무 짧아지면 원본 사용
        trimmed = audio

    # ── 3. 노이즈 제거 (noisereduce 선택적 적용) ─────────
    try:
        import noisereduce as nr
        # 앞 0.3초를 노이즈 프로파일로 사용
        noise_sample = trimmed[:int(TARGET_SR * 0.3)]
        denoised = nr.reduce_noise(
            y=trimmed,
            sr=TARGET_SR,
            y_noise=noise_sample,
            prop_decrease=0.75,   # 75% 노이즈 감소 (과도하면 음성 손상)
            stationary=False,
        )
    except ImportError:
        denoised = trimmed   # noisereduce 없으면 그대로 사용

    # ── 4. RMS 음량 정규화 ────────────────────────────────
    target_rms = 0.1
    current_rms = float(np.sqrt(np.mean(denoised ** 2)))
    if current_rms > 1e-6:
        denoised = denoised * (target_rms / current_rms)
    denoised = np.clip(denoised, -1.0, 1.0)

    trimmed_rate = (1 - len(denoised) / original_len) * 100
    if verbose:
        print(f"  🎛️  전처리 완료  "
              f"({original_len/TARGET_SR:.1f}s → {len(denoised)/TARGET_SR:.1f}s, "
              f"무음 {trimmed_rate:.0f}% 제거, RMS {rms:.3f}→{target_rms:.3f})")

    return denoised



def _play_gtts(text: str) -> bool:
    """
    gTTS로 MP3 생성 → pygame 또는 sounddevice로 재생.
    성공 시 True, 실패 시 False 반환.
    """
    try:
        import io
        from gtts import gTTS
        import sounddevice as sd
        import librosa

        buf = io.BytesIO()
        gTTS(text=text, lang="ko").write_to_fp(buf)
        buf.seek(0)
        audio, sr = librosa.load(buf, sr=22050, mono=True)
        sd.play(audio, samplerate=sr)
        sd.wait()
        return True
    except Exception:
        return False


def _play_pyttsx3(text: str) -> bool:
    """
    pyttsx3 (Windows SAPI) 오프라인 fallback.
    성공 시 True, 실패 시 False 반환.
    """
    try:
        import pyttsx3
        engine = pyttsx3.init()
        # 한국어 목소리 설정 시도
        voices = engine.getProperty("voices")
        for v in voices:
            if "korean" in v.name.lower() or "ko" in v.id.lower():
                engine.setProperty("voice", v.id)
                break
        engine.setProperty("rate", 150)   # 말하기 속도 (기본 200보다 느리게)
        engine.say(text)
        engine.runAndWait()
        return True
    except Exception:
        return False


def play_tts(text: str):
    """
    TTS 재생 진입점.
    gTTS(인터넷) 우선 시도 → 실패 시 pyttsx3(오프라인) fallback.

    사용자가 목표 문장을 귀로 먼저 확인하고 따라 말할 수 있게 해주는 핵심 기능.
    청각장애인 앱 특성상 보청기/인공와우 착용 상태에서의 청취를 보조.
    """
    print(f"  🔊 재생 중: 「{text}」", end="", flush=True)
    if _play_gtts(text):
        print(f"\r  ✅ 재생 완료               ")
        return
    print(f"\r  ⚠️  gTTS 실패 → pyttsx3 시도...", end="", flush=True)
    if _play_pyttsx3(text):
        print(f"\r  ✅ 재생 완료 (pyttsx3)      ")
        return
    print(f"\r  ❌ TTS 재생 실패             ")
    print("     pip install gtts sounddevice  또는  pip install pyttsx3")

# ═══════════════════════════════════════════════════════
# 3. 추론: CTC + Greedy Decoding (LM 없음)
# ═══════════════════════════════════════════════════════

def transcribe_greedy(audio: np.ndarray, processor, model, device: str) -> str:
    """
    핵심 설계 포인트:

    1. processor(audio)       : raw waveform → 정규화된 input_values
                                (MFCC나 mel-spectrogram 아님 — wav2vec2 특징)
    2. model(input_values)    : Transformer encoder → logits [1, T, vocab_size]
    3. torch.argmax(dim=-1)   : 각 time-step t마다 argmax(logits[0,t,:])
                                → 이것이 Greedy Decoding.
                                  beam search 없음, LM rescoring 없음.
    4. batch_decode           : CTC blank 제거 + 연속 중복 collapse
                                → 최종 텍스트 ("들리는 소리 그대로")
    """
    inputs = processor(audio, sampling_rate=TARGET_SR,
                       return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits       # [1, T, V]

    predicted_ids = torch.argmax(logits, dim=-1)  # [1, T] Greedy
    return processor.batch_decode(predicted_ids)[0].strip()

# ═══════════════════════════════════════════════════════
# 4. 음절 단위 Diff + 자모 단위 피드백
# ═══════════════════════════════════════════════════════

# ── 자모 분리 (외부 라이브러리 없이 유니코드 직접 계산) ──
_CHO  = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
_JUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
_JONG = list(" ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")

# 자모 → 발음 분류 매핑
_JAMO_CLASS = {
    # 파열음
    "ㄱ": "파열음(연)", "ㄲ": "파열음(경)", "ㅋ": "파열음(격)",
    "ㄷ": "파열음(연)", "ㄸ": "파열음(경)", "ㅌ": "파열음(격)",
    "ㅂ": "파열음(연)", "ㅃ": "파열음(경)", "ㅍ": "파열음(격)",
    # 파찰음
    "ㅈ": "파찰음(연)", "ㅉ": "파찰음(경)", "ㅊ": "파찰음(격)",
    # 마찰음
    "ㅅ": "마찰음(연)", "ㅆ": "마찰음(경)", "ㅎ": "마찰음",
    # 비음
    "ㄴ": "비음", "ㅁ": "비음", "ㅇ": "비음",
    # 유음
    "ㄹ": "유음",
    # 모음
    "ㅏ": "저모음", "ㅓ": "중모음", "ㅗ": "고모음(원순)",
    "ㅜ": "고모음(원순)", "ㅡ": "고모음", "ㅣ": "고모음(전설)",
    "ㅐ": "중모음(전설)", "ㅔ": "중모음(전설)",
    "ㅑ": "이중모음", "ㅕ": "이중모음", "ㅛ": "이중모음", "ㅠ": "이중모음",
}

def _decompose_char(ch: str) -> list[str]:
    """한글 음절 → 자모 리스트. 비한글은 그대로 반환."""
    code = ord(ch) - 0xAC00
    if code < 0 or code > 11171:
        return [ch]
    cho  = code // (21 * 28)
    jung = (code % (21 * 28)) // 28
    jong = code % 28
    result = [_CHO[cho], _JUNG[jung]]
    if jong > 0:
        result.append(_JONG[jong])
    return result

def _decompose_text(text: str) -> list[str]:
    """문장 → 자모 시퀀스."""
    result = []
    for ch in text.replace(" ", ""):
        result.extend(_decompose_char(ch))
    return result

def _jamo_feedback(ref_syl: str, hyp_syl: str) -> str:
    """
    음절 교체 오류에서 구체적인 자모 차이를 찾아 피드백 생성.
    예) '가치' vs '가티' → "ㅊ(파찰음(격)) → ㅌ(파열음(격))"
    """
    ref_jamo = _decompose_text(ref_syl)
    hyp_jamo = _decompose_text(hyp_syl)

    if len(ref_jamo) != len(hyp_jamo):
        return ""

    diffs = [(r, h) for r, h in zip(ref_jamo, hyp_jamo) if r != h]
    if not diffs:
        return ""

    parts = []
    for r, h in diffs[:2]:   # 최대 2개만 표시
        r_cls = _JAMO_CLASS.get(r, "")
        h_cls = _JAMO_CLASS.get(h, "")
        r_str = f"{r}({r_cls})" if r_cls else r
        h_str = f"{h}({h_cls})" if h_cls else h
        parts.append(f"{r_str} → {h_str}")

    return "  ".join(parts)


def analyze_pronunciation(original: str, hyp: str) -> dict:
    """
    original : 사용자가 입력한 목표 문장 (맞춤법)
    hyp      : 모델이 인식한 결과
    phonetic : G2P로 변환한 발음 전사 → 실제 채점 기준

    반환값에 jamo_feedback 추가:
      음절 교체 오류에서 자모 단위 차이를 분석해 구체적 피드백 제공
      "ㅊ(파찰음(격)) → ㅌ(파열음(격))" 같은 형태
    """
    from jiwer import cer as compute_cer

    phonetic = to_phonetic(original)

    ref_syl = list(phonetic.replace(" ", ""))
    hyp_syl = list(hyp.replace(" ", ""))

    matcher = difflib.SequenceMatcher(None, ref_syl, hyp_syl, autojunk=False)
    diff = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        ref_seg = "".join(ref_syl[i1:i2])
        hyp_seg = "".join(hyp_syl[j1:j2])
        entry = {
            "type": tag,
            "ref":  ref_seg,
            "hyp":  hyp_seg,
        }
        # 교체 오류에 자모 피드백 추가
        if tag == "replace" and len(ref_seg) == len(hyp_seg):
            fb = _jamo_feedback(ref_seg, hyp_seg)
            if fb:
                entry["jamo_feedback"] = fb
        diff.append(entry)

    ref_flat   = phonetic.replace(" ", "")
    hyp_flat   = hyp.replace(" ", "")
    error_rate = compute_cer(ref_flat, hyp_flat) if ref_flat else 0.0
    score      = max(0.0, round((1 - error_rate) * 100, 1))

    return {
        "original": original,
        "phonetic": phonetic,
        "hyp":      hyp,
        "score":    score,
        "cer":      round(error_rate * 100, 1),
        "diff":     diff,
        "n_ref":    len(ref_syl),
    }

# ═══════════════════════════════════════════════════════
# 5. 터미널 출력
# ═══════════════════════════════════════════════════════

C = {
    "green":  "\033[92m", "red":    "\033[91m",
    "yellow": "\033[93m", "blue":   "\033[94m",
    "cyan":   "\033[96m", "bold":   "\033[1m",
    "dim":    "\033[2m",  "reset":  "\033[0m",
}

def _grade(score: float) -> str:
    if score >= 95: return "🏆 완벽해요!"
    if score >= 85: return "🌟 훌륭해요!"
    if score >= 70: return "👍 잘 하셨어요"
    if score >= 50: return "💬 조금 더 연습해봐요"
    return "💪 함께 연습해봐요"

def print_result(a: dict, show_json: bool = False):
    score = a["score"]
    sc    = C["green"] if score >= 80 else C["yellow"] if score >= 50 else C["red"]
    bar   = "█" * int(score / 5) + "░" * (20 - int(score / 5))

    # G2P 변환이 일어난 경우에만 발음 전사 표시
    phonetic_changed = a["phonetic"] != a["original"]

    print("\n" + "═" * 56)
    print(f"  {C['bold']}📊 발음 분석 결과{C['reset']}")
    print("═" * 56)
    print(f"  목표 문장  :  {C['blue']}{a['original']}{C['reset']}")
    if phonetic_changed:
        print(f"  발음 기준  :  {C['cyan']}{a['phonetic']}{C['reset']}  "
              f"{C['dim']}← G2P 변환{C['reset']}")
    print(f"  인식 결과  :  {a['hyp']}")
    print()
    print(f"  점  수     :  {sc}{C['bold']}{score}점{C['reset']}  {_grade(score)}")
    print(f"  오류율(CER):  {a['cer']}%")
    print(f"  [{sc}{bar}{C['reset']}]")
    print()

    # 음절 diff 시각화 (발음 전사 기준)
    print(f"  {C['bold']}[음절 단위 비교]  "
          f"{C['dim']}(발음 전사 기준){C['reset']}")
    ref_vis = "  발음 정답: "
    hyp_vis = "  모델 인식: "
    for seg in a["diff"]:
        t, r_s, h_s = seg["type"], seg["ref"], seg["hyp"]
        if t == "equal":
            ref_vis += C["green"]  + r_s         + C["reset"]
            hyp_vis += C["green"]  + h_s         + C["reset"]
        elif t == "replace":
            ref_vis += C["red"]    + f"[{r_s}]"  + C["reset"]
            hyp_vis += C["yellow"] + f"[{h_s}]"  + C["reset"]
        elif t == "delete":
            ref_vis += C["red"]    + f"[{r_s}↓]" + C["reset"]
            hyp_vis += C["dim"]    + "[·]"        + C["reset"]
        elif t == "insert":
            ref_vis += C["dim"]    + "[·]"        + C["reset"]
            hyp_vis += C["yellow"] + f"[+{h_s}]" + C["reset"]
    print(ref_vis)
    print(hyp_vis)
    print(f"\n  {C['dim']}범례: {C['green']}■정확  "
          f"{C['red']}■오류  {C['yellow']}■변형/추가  {C['dim']}■탈락{C['reset']}")

    # 오류 상세 + 자모 피드백
    errors = [s for s in a["diff"] if s["type"] != "equal"]
    print()
    if not errors:
        print(f"  {C['green']}✅ 오류 없음 — 완벽한 발음!{C['reset']}")
    else:
        print(f"  {C['bold']}[오류 {len(errors)}건]{C['reset']}")
        for seg in errors:
            t = seg["type"]
            if t == "replace":
                fb = seg.get("jamo_feedback", "")
                fb_str = f"  {C['dim']}({fb}){C['reset']}" if fb else ""
                print(f"  • {C['red']}'{seg['ref']}'{C['reset']} → "
                      f"{C['yellow']}'{seg['hyp']}'{C['reset']} 로 발음됨{fb_str}")
            elif t == "delete":
                print(f"  • {C['red']}'{seg['ref']}'{C['reset']} 음절 탈락")
            elif t == "insert":
                print(f"  • {C['yellow']}'{seg['hyp']}'{C['reset']} 음절 추가됨")

    print("═" * 56)

    if show_json:
        print(json.dumps(a, ensure_ascii=False, indent=2))

# ═══════════════════════════════════════════════════════
# 6. 연습 모드 (★ 핵심 기능: 듣기 → 따라 말하기 → 분석)
# ═══════════════════════════════════════════════════════

def practice_mode(processor, model, device, text: str, duration: int = 5,
                  save_path: str | None = None, show_json: bool = False):
    """
    단일 문장 연습 흐름:
      ① G2P 발음 전사 미리보기
      ② TTS 재생 (r 입력 시 반복)
      ③ Enter → 녹음 시작
      ④ 분석 및 결과 출력
      ⑤ 재도전 여부 선택
    """
    phonetic = to_phonetic(text)
    phonetic_changed = phonetic != text

    print(f"\n{'═'*56}")
    print(f"  {C['bold']}🎯 연습 문장{C['reset']}")
    print(f"{'═'*56}")
    print(f"  목표 문장  :  {C['blue']}{text}{C['reset']}")
    if phonetic_changed:
        print(f"  발음 기준  :  {C['cyan']}{phonetic}{C['reset']}  "
              f"{C['dim']}← G2P 변환{C['reset']}")
    print()

    attempt = 0
    while True:
        attempt += 1
        if attempt > 1:
            print(f"\n  {C['cyan']}[재도전 #{attempt}]{C['reset']}")

        # ── ② TTS 재생 루프 ──────────────────────────────
        play_tts(text)
        while True:
            cmd = input("  r=다시듣기  Enter=녹음시작 > ").strip().lower()
            if cmd == "r":
                play_tts(text)
            else:
                break  # Enter → 녹음으로

        # ── ③ 녹음 ───────────────────────────────────────
        audio = record_from_mic(duration=duration, save_path=save_path)

        # ── ④ 전처리 + 분석 ───────────────────────────────
        audio  = preprocess_audio(audio)
        print("\n  [분석 중]...", end="", flush=True)
        hyp    = transcribe_greedy(audio, processor, model, device)
        print(" 완료")
        result = analyze_pronunciation(text, hyp)
        print_result(result, show_json=show_json)

        # ── ⑤ 재도전 여부 ────────────────────────────────
        score = result["score"]
        if score >= 95:
            print(f"  {C['green']}🏆 완벽해요! 다음 문장으로 넘어가세요.{C['reset']}\n")
            break

        cmd = input("  다시 연습할까요?  Enter=재도전  q=종료 > ").strip().lower()
        if cmd == "q":
            break

    return result


# ═══════════════════════════════════════════════════════
# 7. 배치 연습 모드 (TTS 포함)
# ═══════════════════════════════════════════════════════

PRACTICE_SENTENCES = [
    "안녕하세요 반갑습니다",
    "같이 해볼까요",
    "오늘 날씨가 좋네요",
    "저는 잘 들리지 않아요",
    "천천히 말해주세요",
    "조금 더 크게 말씀해 주세요",
    "다시 한번 말씀해 주시겠어요",
]

def batch_mode(processor, model, device, duration: int = 5):
    print(f"\n{C['bold']}=== 배치 연습 모드 ==={C['reset']}")
    print(f"  총 {len(PRACTICE_SENTENCES)}문장")
    print(f"  TTS 재생 → r=다시듣기 / Enter=녹음 / s=건너뛰기 / q=종료\n")

    results = []
    for i, sentence in enumerate(PRACTICE_SENTENCES, 1):
        phonetic = to_phonetic(sentence)
        print(f"\n{C['cyan']}[{i}/{len(PRACTICE_SENTENCES)}]  {sentence}{C['reset']}", end="")
        if phonetic != sentence:
            print(f"  {C['dim']}→ {phonetic}{C['reset']}", end="")
        print()

        cmd = input("  Enter=시작  s=건너뛰기  q=종료 > ").strip().lower()
        if cmd == "q": break
        if cmd == "s": print("  ⏭ 건너뜀"); continue

        # TTS 재생
        play_tts(sentence)
        while True:
            cmd = input("  r=다시듣기  Enter=녹음시작 > ").strip().lower()
            if cmd == "r":
                play_tts(sentence)
            else:
                break

        audio  = record_from_mic(duration=duration)
        print("\n  [분석 중]...", end="", flush=True)
        hyp    = transcribe_greedy(audio, processor, model, device)
        print(" 완료")
        result = analyze_pronunciation(sentence, hyp)
        print_result(result)
        results.append(result)

    if results:
        avg = sum(r["score"] for r in results) / len(results)
        print(f"\n{'═'*56}")
        print(f"  {C['bold']}세션 요약{C['reset']}  |  "
              f"연습 {len(results)}문장  |  평균 {avg:.1f}점  {_grade(avg)}")
        print("═" * 56)

# ═══════════════════════════════════════════════════════
# 8. 메인
# ═══════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="올라잇 발음 분석기 (1단계 · CTC + Greedy Decoding)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--text",     type=str,  help="목표 문장 (정답)")
    parser.add_argument("--audio",    type=str,  help="오디오 파일 (.wav/.mp3/.m4a)")
    parser.add_argument("--mic",      action="store_true", help="마이크 실시간 녹음")
    parser.add_argument("--practice", action="store_true", help="★ 연습 모드: 듣기 → 따라 말하기 → 분석")
    parser.add_argument("--duration", type=int,  default=5,    help="녹음 시간 초 (기본 5)")
    parser.add_argument("--save",     type=str,  default=None, help="녹음 저장 경로")
    parser.add_argument("--batch",    action="store_true",     help="배치 연습 모드 (TTS 포함)")
    parser.add_argument("--json",     action="store_true",     help="JSON 출력 포함")
    parser.add_argument("--preview",  action="store_true",     help="G2P 변환 결과만 미리보기")
    parser.add_argument("--model",    type=str,  default=MODEL_OPTIONS[0])
    args = parser.parse_args()

    # --preview: 모델 로드 없이 G2P 변환만 확인
    if args.preview:
        if not args.text:
            parser.error("--preview 는 --text 와 함께 사용하세요.")
        phonetic = to_phonetic(args.text)
        print(f"\n  맞춤법 : {args.text}")
        print(f"  발음   : {C['cyan']}{phonetic}{C['reset']}")
        changed = "변환됨" if phonetic != args.text else "변환 없음"
        print(f"  ({changed})\n")
        return

    check_dependencies(need_mic=args.mic or args.practice or args.batch)

    if not args.batch and not args.practice and not args.text:
        parser.error("--text, --practice, --batch 중 하나를 지정하세요.")
    if not args.batch and not args.practice and not args.audio and not args.mic:
        parser.error("--audio 또는 --mic 중 하나를 선택하세요.")

    processor, model, device = load_model(args.model)

    # ── 연습 모드 (★ 핵심) ───────────────────────────────
    if args.practice:
        if not args.text:
            # 대화형: 문장을 직접 입력받기
            print(f"\n{C['bold']}=== 올라잇 연습 모드 ==={C['reset']}")
            print(f"  {C['dim']}문장 입력 → TTS 재생 → 따라 말하기 → 분석{C['reset']}\n")
            while True:
                text = input(f"  {C['cyan']}연습할 문장을 입력하세요 (q=종료): {C['reset']}").strip()
                if text.lower() == "q":
                    break
                if not text:
                    continue
                practice_mode(processor, model, device, text,
                              duration=args.duration, save_path=args.save,
                              show_json=args.json)
                print()
        else:
            # --text 와 함께 사용: 단일 문장 연습
            practice_mode(processor, model, device, args.text,
                          duration=args.duration, save_path=args.save,
                          show_json=args.json)
        return

    # ── 배치 모드 ─────────────────────────────────────────
    if args.batch:
        batch_mode(processor, model, device, args.duration)
        return

    # ── 단일 분석 (기존) ──────────────────────────────────
    audio = (record_from_mic(args.duration, args.save)
             if args.mic else load_audio_file(args.audio))

    audio = preprocess_audio(audio)
    print("\n  [분석 중]...", end="", flush=True)
    hyp = transcribe_greedy(audio, processor, model, device)
    print(" 완료")

    result = analyze_pronunciation(args.text, hyp)
    print_result(result, show_json=args.json)

if __name__ == "__main__":
    main()
