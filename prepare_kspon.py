"""
KsponSpeech(자유대화) 정규화 → Whisper 학습 JSONL
==================================================
실제 자유 발화 코퍼스는 외래어/구어 저빈도어가 자연스럽게 풍부하다(데시벨·프로젝트명
같은 어휘가 실제 문맥·억양으로 등장). Zeroth 낭독체의 어휘 한계를 메우는 '실제' 축.
TTS 타깃 합성(prepare_loanword_tts.py)과 섞어서 쓴다.

핵심: KsponSpeech 의 **이중전사 `(철자)/(발음)`** 에서 **(발음)** 을 택한다.
이 모델은 발음 전사 모델이라 (발음) 형태가 정확히 라벨 도메인과 일치한다.
  예) "(데시벨)/(데시벨)", "(2)/(이) 도", "(SK)/(에스케이)" → 발음형 채택
그 위에 G2P(연음/경음화 등)를 한 번 더 적용해 Zeroth 라벨과 표기 규칙을 통일한다.

입력 포맷(KsponSpeech 표준):
  - 오디오: .pcm  (16kHz, 16-bit LE, mono, 헤더 없음)
  - 전사 : 같은 basename .txt  (보통 CP949/EUC-KR 인코딩)
  - .trn 스크립트(KsponSpeech_scripts/*.trn: "<pcm경로> :: <전사>")도 지원(--trn).

실행(서버):
  # .trn 스크립트 기반 (권장 — train/eval 목록이 정해져 있음)
  python prepare_kspon.py \
      --audio_root /data/KsponSpeech \
      --trn /data/KsponSpeech_scripts/train.trn \
      --output_dir ~/mingly_workspace/Voice-Model-Test/kspon_dataset \
      --max_utts 60000

  # 디렉터리 재귀 스캔(.pcm + .txt 페어)
  python prepare_kspon.py --audio_root /data/KsponSpeech --scan \
      --output_dir ~/mingly_workspace/Voice-Model-Test/kspon_dataset
"""

import os
import re
import json
import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

from korean_g2p_nomecab import load_g2p

TARGET_SR = 16000
MIN_SEC = 0.5
MAX_SEC = 25.0      # finetune_whisper.py MAX_SEC 과 동일


# ────────────────────────────────────────────
# 전사 정규화
# ────────────────────────────────────────────
# 이중전사 (철자)/(발음) → 발음(두 번째 괄호) 채택
_DUAL = re.compile(r"\(([^()]*)\)\s*/\s*\(([^()]*)\)")
# 잡음/간투어 태그: b/ n/ l/ o/ u/  (각각 숨소리/노이즈/웃음/외래어표기/불명)
_TAG = re.compile(r"(?:^|\s)?[bnlou]\s*/")
# 남은 단독 괄호(한쪽만 전사된 경우) — 내용만 남기고 괄호 제거
_PAREN = re.compile(r"\(([^()]*)\)")


def normalize_kspon(text):
    """KsponSpeech 전사 한 줄 → 발음형 한글 텍스트. 정규화 실패 시 ''."""
    if not text:
        return ""
    t = text.strip()
    # 1) 이중전사 → 발음형
    t = _DUAL.sub(lambda m: m.group(2), t)
    # 2) 잡음/간투 태그 제거
    t = _TAG.sub(" ", t)
    # 3) 남은 단독 괄호 → 내용만
    t = _PAREN.sub(lambda m: m.group(1), t)
    # 4) 특수 마커 제거: * (오발음) + (반복) / 따옴표 등
    t = t.replace("*", " ").replace("+", " ").replace("/", " ")
    t = re.sub(r"[\"'`]", " ", t)
    # 5) 한글/숫자/공백/일부 문장부호 외 제거 (영문 잔재 등)
    t = re.sub(r"[^가-힣0-9\s\.\?\!,]", " ", t)
    # 6) 공백 정리
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ────────────────────────────────────────────
# 오디오 로드
# ────────────────────────────────────────────
def load_pcm(path, sr=TARGET_SR):
    """헤더 없는 16-bit LE mono PCM → float32 [-1,1]."""
    raw = np.fromfile(str(path), dtype=np.int16)
    return (raw.astype(np.float32) / 32768.0)


def read_transcript_txt(path):
    """.txt 전사 읽기. CP949 → UTF-8 폴백."""
    for enc in ("cp949", "euc-kr", "utf-8"):
        try:
            return Path(path).read_text(encoding=enc).strip()
        except (UnicodeDecodeError, LookupError):
            continue
    return ""


# ────────────────────────────────────────────
# 페어 수집
# ────────────────────────────────────────────
def pairs_from_trn(trn_path, audio_root):
    """KsponSpeech .trn ("<상대pcm경로> :: <전사>") → (pcm_path, raw_text) 리스트."""
    out = []
    root = Path(audio_root)
    for line in Path(trn_path).read_text(encoding="utf-8", errors="ignore").splitlines():
        if "::" not in line:
            continue
        rel, text = line.split("::", 1)
        out.append((root / rel.strip(), text.strip()))
    return out


def pairs_from_scan(audio_root):
    """디렉터리 재귀 스캔: 각 .pcm 옆 동일 basename .txt 페어."""
    out = []
    for pcm in Path(audio_root).rglob("*.pcm"):
        txt = pcm.with_suffix(".txt")
        if txt.exists():
            out.append((pcm, read_transcript_txt(txt)))
    return out


# ────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="KsponSpeech 자유대화 → Whisper JSONL")
    ap.add_argument("--audio_root", required=True, help="KsponSpeech 오디오 루트(.pcm)")
    ap.add_argument("--trn", default=None, help="KsponSpeech .trn 스크립트 경로")
    ap.add_argument("--scan", action="store_true", help=".pcm+.txt 페어 재귀 스캔")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_utts", type=int, default=0, help="최대 발화 수(0=무제한)")
    ap.add_argument("--val_ratio", type=float, default=0.02)
    ap.add_argument("--test_ratio", type=float, default=0.02)
    ap.add_argument("--no_g2p", action="store_true",
                    help="G2P 미적용(전사형 그대로 label). 기본은 G2P 적용해 Zeroth 와 통일")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.trn and not args.scan:
        raise SystemExit("❌ --trn 또는 --scan 중 하나가 필요합니다.")

    out_dir = Path(args.output_dir)
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    print("📂 페어 수집...")
    pairs = pairs_from_trn(args.trn, args.audio_root) if args.trn \
        else pairs_from_scan(args.audio_root)
    print(f"   원본 페어: {len(pairs):,}개")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    if args.max_utts > 0:
        pairs = pairs[:args.max_utts]

    g2p = None if args.no_g2p else load_g2p()

    n = len(pairs)
    n_test = int(n * args.test_ratio)
    n_val = int(n * args.val_ratio)

    writers = {sp: open(out_dir / f"{sp}.jsonl", "w", encoding="utf-8")
               for sp in ("train", "validation", "test")}
    counts = {sp: 0 for sp in writers}
    skip = {"noaudio": 0, "badtext": 0, "dur": 0}

    try:
        for i, (pcm_path, raw_text) in enumerate(tqdm(pairs, desc="kspon")):
            if not Path(pcm_path).exists():
                skip["noaudio"] += 1
                continue

            text = normalize_kspon(raw_text)
            if len(text) < 2 or not re.search(r"[가-힣]", text):
                skip["badtext"] += 1
                continue

            try:
                audio = load_pcm(pcm_path)
            except Exception:
                skip["noaudio"] += 1
                continue
            duration = len(audio) / TARGET_SR
            if duration < MIN_SEC or duration > MAX_SEC:
                skip["dur"] += 1
                continue

            sp = "test" if i < n_test else "validation" if i < n_test + n_val else "train"
            wav_name = f"{sp}_{i:07d}.wav"
            wav_path = wav_dir / wav_name
            sf.write(str(wav_path), audio, TARGET_SR)

            label = g2p(text, descriptive=True).strip() if g2p else text
            obj = {
                "wav_path": str(wav_path),
                "transcript": text,
                "label": label,
                "duration": float(duration),
                "source": "kspon",
            }
            writers[sp].write(json.dumps(obj, ensure_ascii=False) + "\n")
            counts[sp] += 1
    finally:
        for w in writers.values():
            w.close()

    print(f"\n✅ 완료 → {out_dir}")
    for sp in ("train", "validation", "test"):
        print(f"   {sp:11s}: {counts[sp]:,}개")
    print(f"   🧹 스킵 — 오디오없음:{skip['noaudio']:,} "
          f"전사불량:{skip['badtext']:,} 길이초과:{skip['dur']:,}")
    print(f"\n다음: finetune_whisper.py 에 --extra_json_dirs 로 이 디렉터리를 추가해 학습.")


if __name__ == "__main__":
    main()
