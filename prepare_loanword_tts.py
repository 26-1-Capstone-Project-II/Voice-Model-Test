"""
외래어/저빈도어 TTS 타깃 합성 코퍼스 생성
==========================================
loanword_corpus.py 가 만든 발화 원문을 한국어 TTS 로 합성해, Zeroth 가 못 본
외래어/저빈도어의 acoustic→text 매핑을 학습에 주입한다. 출력 포맷은 zeroth_dataset
과 동일(JSONL: wav_path/transcript/label/duration)해서 finetune_whisper.py 가 그대로 읽는다.

왜 TTS 인가:
  - 실패하는 어휘를 '정확히' 타깃할 수 있다(실제 코퍼스엔 데시벨·스톤이 거의 안 나옴).
  - 합성 발화의 운율/단일화자 도메인 갭은 학습 단계의 음향 증강(RIR/소음/SpecAugment)과
    여기 들어간 화자 섭동(pitch/tempo)이 상당 부분 메운다.
  - 실제 자유대화 코퍼스(prepare_kspon.py)와 '섞어서' 쓰는 게 전제다. 단독 사용 금지.

라벨:
  - 이 모델은 '발음 전사' 모델이므로 label = G2P(transcript, descriptive=True).
    (finetune_whisper.py 에서 --apply_g2p 를 쓰면 transcript 로부터 다시 생성하므로,
     여기 저장하는 label 은 그 경로를 쓰지 않을 때를 위한 것. 둘 다 동일 G2P 라 일치.)

TTS 백엔드 (--backend):
  - mms  : facebook/mms-tts-kor (transformers VITS). 오프라인·설치 간단·16kHz. 기본값.
  - melo : MeloTTS 한국어 (별도 설치). 품질↑.
  - gtts : Google TTS (인터넷 필요). 빠른 점검용.
  - xtts : Coqui XTTS-v2 다화자(참조 wav 디렉터리 필요, --speaker_wav_dir). 화자 다양성↑.

실행(서버):
  # 빠른 점검 (소량, gtts)
  python prepare_loanword_tts.py --backend gtts --per_word 2 --limit 30

  # 본 합성 (mms, 화자 섭동 on)
  python prepare_loanword_tts.py --backend mms --per_word 6 --perturb \
      --output_dir ~/mingly_workspace/Voice-Model-Test/loanword_dataset
"""

import io
import os
import json
import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

from loanword_corpus import generate_sentences
from korean_g2p_nomecab import load_g2p

TARGET_SR = 16000


# ────────────────────────────────────────────
# TTS 백엔드 — 공통 인터페이스: synth(text) -> (np.float32 mono@TARGET_SR)
# ────────────────────────────────────────────
class MMSBackend:
    """facebook/mms-tts-kor (VITS). 오프라인, 16kHz, 단일 화자."""
    def __init__(self):
        import torch
        from transformers import VitsModel, AutoTokenizer
        self.torch = torch
        self.model = VitsModel.from_pretrained("facebook/mms-tts-kor")
        self.tok = AutoTokenizer.from_pretrained("facebook/mms-tts-kor")
        self.model.eval()
        self.sr = self.model.config.sampling_rate  # 16000
        # mms-tts-kor 토크나이저는 한글에 uroman 로마자화가 필요하다. 미설치면 토큰이
        # 빈값(size 0)이 돼 *모든* 합성이 조용히 실패한다 → 시작 시점에 명확히 막는다.
        if getattr(self.tok, "is_uroman", False):
            try:
                import uroman  # noqa: F401  (transformers 가 자동 적용)
            except Exception:
                raise SystemExit(
                    "❌ mms-tts-kor 는 uroman 이 필요합니다 (한글 로마자화).\n"
                    "   해결: pip install uroman   (python>=3.10)\n"
                    "   또는: --backend gtts (인터넷) / --backend melo 사용")
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.cuda = True
        else:
            self.cuda = False

    def synth(self, text):
        inputs = self.tok(text, return_tensors="pt")
        if self.cuda:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        with self.torch.no_grad():
            wav = self.model(**inputs).waveform[0].detach().cpu().numpy()
        return _to_target_sr(wav.astype(np.float32), self.sr)


class MeloBackend:
    """MeloTTS 한국어. pip 별도 설치 필요(melo)."""
    def __init__(self):
        from melo.api import TTS
        device = "cuda" if _has_cuda() else "cpu"
        self.tts = TTS(language="KR", device=device)
        self.sid = self.tts.hps.data.spk2id["KR"]
        self.sr = self.tts.hps.data.sampling_rate

    def synth(self, text):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as f:
            self.tts.tts_to_file(text, self.sid, f.name, speed=1.0)
            wav, sr = sf.read(f.name)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return _to_target_sr(wav.astype(np.float32), sr)


class GTTSBackend:
    """Google TTS. 인터넷 필요. 빠른 배선 점검용(품질/도메인 한계 있음)."""
    def __init__(self):
        from gtts import gTTS
        self._gTTS = gTTS

    def synth(self, text):
        buf = io.BytesIO()
        self._gTTS(text=text, lang="ko").write_to_fp(buf)
        buf.seek(0)
        wav, sr = librosa.load(buf, sr=TARGET_SR, mono=True)  # librosa 가 mp3 디코딩
        return wav.astype(np.float32)


class XTTSBackend:
    """Coqui XTTS-v2 다화자(참조 wav 클로닝). --speaker_wav_dir 의 wav 들을 순환 사용."""
    def __init__(self, speaker_wav_dir):
        from TTS.api import TTS
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2",
                       gpu=_has_cuda())
        self.speakers = sorted(str(p) for p in Path(speaker_wav_dir).glob("*.wav"))
        if not self.speakers:
            raise SystemExit(f"❌ XTTS: 참조 화자 wav 없음 → {speaker_wav_dir}")
        self.sr = 24000

    def synth(self, text):
        spk = random.choice(self.speakers)
        wav = self.tts.tts(text=text, speaker_wav=spk, language="ko")
        return _to_target_sr(np.asarray(wav, dtype=np.float32), self.sr)


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _to_target_sr(wav, sr):
    if sr != TARGET_SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
    peak = float(np.max(np.abs(wav))) if len(wav) else 0.0
    if peak > 0:
        wav = 0.95 * wav / peak            # 정규화(클리핑 방지)
    return wav.astype(np.float32)


def make_backend(name, speaker_wav_dir=None):
    name = name.lower()
    if name == "mms":
        return MMSBackend()
    if name == "melo":
        return MeloBackend()
    if name == "gtts":
        return GTTSBackend()
    if name == "xtts":
        return XTTSBackend(speaker_wav_dir)
    raise SystemExit(f"❌ 알 수 없는 backend: {name}")


# ────────────────────────────────────────────
# 화자 섭동 — 단일 화자 TTS 의 음향 다양성을 싸게 늘린다(가짜 다화자).
# ────────────────────────────────────────────
def perturb_speaker(wav, rng):
    """pitch shift(±2 반음) + time stretch(±10%). 라벨은 불변(텍스트 동일)."""
    n_steps = rng.uniform(-2.0, 2.0)
    rate = rng.uniform(0.9, 1.1)
    try:
        wav = librosa.effects.pitch_shift(wav, sr=TARGET_SR, n_steps=n_steps)
        wav = librosa.effects.time_stretch(wav, rate=rate)
    except Exception:
        return wav
    return wav.astype(np.float32)


# ────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="외래어/저빈도어 TTS 타깃 합성 코퍼스")
    ap.add_argument("--output_dir", default=str(
        Path.home() / "mingly_workspace" / "Voice-Model-Test" / "loanword_dataset"))
    ap.add_argument("--backend", default="mms", choices=["mms", "melo", "gtts", "xtts"])
    ap.add_argument("--speaker_wav_dir", default=None, help="xtts 참조 화자 wav 디렉터리")
    ap.add_argument("--per_word", type=int, default=6, help="단어당 carrier 문장 수")
    ap.add_argument("--pair_ratio", type=float, default=0.15, help="외래어 2개 문장 비율")
    ap.add_argument("--perturb", action="store_true", help="화자 섭동(pitch/tempo)으로 다화자 흉내")
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--test_ratio", type=float, default=0.10,
                    help="외래어 전용 평가셋 비율(전후 OOV 개선 측정용)")
    ap.add_argument("--limit", type=int, default=0, help="문장 수 상한(0=무제한, 점검용)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    # 1. 발화 원문 생성
    sents = generate_sentences(per_word=args.per_word, pair_ratio=args.pair_ratio,
                               seed=args.seed)
    if args.limit > 0:
        sents = sents[:args.limit]
    print(f"📝 합성 대상 문장: {len(sents):,}개  (backend={args.backend}, "
          f"perturb={'on' if args.perturb else 'off'})")

    # 2. G2P 로드(발음 라벨)
    g2p = load_g2p()

    # 3. TTS 백엔드
    print(f"🔊 TTS 백엔드 로딩: {args.backend}")
    tts = make_backend(args.backend, args.speaker_wav_dir)

    # 4. split 경계 (문장 단위, 화자 누수 걱정 없음 — 합성이므로)
    rng.shuffle(sents)
    n = len(sents)
    n_test = int(n * args.test_ratio)
    n_val = int(n * args.val_ratio)
    split_of = {}
    for i, s in enumerate(sents):
        split_of[s] = "test" if i < n_test else "validation" if i < n_test + n_val else "train"

    writers = {sp: open(out_dir / f"{sp}.jsonl", "w", encoding="utf-8")
               for sp in ("train", "validation", "test")}
    counts = {sp: 0 for sp in writers}
    fail = 0

    try:
        for i, text in enumerate(tqdm(sents, desc="TTS")):
            try:
                wav = tts.synth(text)
                if args.perturb:
                    wav = perturb_speaker(wav, rng)
            except Exception as e:
                fail += 1
                if fail <= 5:
                    print(f"  ⚠️ 합성 실패: {text!r} → {e}")
                continue

            if wav is None or len(wav) < int(0.2 * TARGET_SR):
                fail += 1
                continue

            sp = split_of[text]
            wav_name = f"{sp}_{i:06d}.wav"
            wav_path = wav_dir / wav_name
            sf.write(str(wav_path), wav, TARGET_SR)

            label = g2p(text, descriptive=True).strip()
            obj = {
                "wav_path": str(wav_path),
                "transcript": text,
                "label": label,
                "duration": float(len(wav) / TARGET_SR),
                "source": "loanword_tts",      # 다중 소스 병합 시 출처 추적
            }
            writers[sp].write(json.dumps(obj, ensure_ascii=False) + "\n")
            counts[sp] += 1
    finally:
        for w in writers.values():
            w.close()

    total = sum(counts.values())
    if total == 0:
        print(f"\n❌ 합성 결과 0개 — TTS 백엔드 점검 필요(예: --backend gtts, 또는 "
              f"mms 라면 pip install uroman). 실패/스킵 {fail:,}개")
        raise SystemExit(1)

    print(f"\n✅ 완료 → {out_dir}")
    for sp in ("train", "validation", "test"):
        print(f"   {sp:11s}: {counts[sp]:,}개")
    if fail:
        print(f"   ⚠️ 합성 실패/스킵: {fail:,}개")
    print(f"\n다음: finetune_whisper.py 에 --extra_json_dirs 로 이 디렉터리를 추가해 학습.")
    print(f"      test.jsonl 은 외래어 전용 평가셋(전후 OOV 개선 측정).")


if __name__ == "__main__":
    main()
