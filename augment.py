"""
원거리/소음 환경 증강 (train split 전용)
==========================================
clean → (reverb) → (+noise@SNR) → (gain) → 클리핑 방지

목적: Zeroth-Korean clean read speech 만 본 모델에 잡음·울림·거리 도메인을 주입해
      원거리/소음 환경 개인성(robustness)을 확보한다.

코퍼스:
  - MUSAN  (OpenSLR 17): 배경 소음/음악/웅성거림  → noise_root
  - RIRS_NOISES (OpenSLR 28): 방 임펄스 응답(거리감) → rir_root

순서가 중요: 울림을 먼저 입혀 "마이크에 도달한 신호"를 만들고, 그 위에 소음을 더한다
(소음은 마이크 위치에서 더해지므로 다시 울리지 않는다).
"""

import random
import numpy as np
import librosa
from pathlib import Path
from scipy.signal import fftconvolve

TARGET_SR = 16000  # finetune_whisper.py 의 TARGET_SR 과 동일


def _load_wav_list(root, exts=(".wav", ".flac")):
    if root is None:
        return []
    root = Path(root)
    if not root.exists():
        print(f"[Augmentor] 경고: 경로 없음 {root}")
        return []
    return [str(p) for p in root.rglob("*") if p.suffix.lower() in exts]


def _load_audio(path, sr=TARGET_SR):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y.astype(np.float32)


def _match_length(noise, length):
    if len(noise) == 0:
        return np.zeros(length, dtype=np.float32)
    if len(noise) >= length:
        start = random.randint(0, len(noise) - length)
        return noise[start:start + length]
    reps = int(np.ceil(length / len(noise)))
    return np.tile(noise, reps)[:length]


class FarFieldAugmentor:
    """clean → (reverb) → (+noise@SNR) → (gain). train split 에서만 사용."""

    def __init__(
        self,
        noise_root=None,
        rir_root=None,
        p_reverb=0.5,
        p_noise=0.6,
        p_gain=0.5,
        snr_db_range=(5.0, 20.0),
        hard_snr_db_range=(0.0, 5.0),
        p_hard_snr=0.1,
        gain_db_range=(-6.0, 6.0),
        seed=None,
    ):
        self.noise_files = _load_wav_list(noise_root)
        self.rir_files = _load_wav_list(rir_root)
        self.p_reverb = p_reverb if self.rir_files else 0.0
        self.p_noise = p_noise if self.noise_files else 0.0
        self.p_gain = p_gain
        self.snr_db_range = snr_db_range
        self.hard_snr_db_range = hard_snr_db_range
        self.p_hard_snr = p_hard_snr
        self.gain_db_range = gain_db_range
        if seed is not None:
            random.seed(seed)
        print(f"[Augmentor] noise={len(self.noise_files)} rir={len(self.rir_files)} "
              f"p_reverb={self.p_reverb} p_noise={self.p_noise}")

    def _reverberate(self, audio):
        rir = _load_audio(random.choice(self.rir_files))
        if len(rir) == 0:
            return audio
        peak = int(np.argmax(np.abs(rir)))
        rir = rir / (np.max(np.abs(rir)) + 1e-8)
        wet = fftconvolve(audio, rir)[peak:peak + len(audio)]
        return wet.astype(np.float32)

    def _add_noise(self, audio):
        noise = _match_length(_load_audio(random.choice(self.noise_files)), len(audio))
        if random.random() < self.p_hard_snr:
            snr = random.uniform(*self.hard_snr_db_range)
        else:
            snr = random.uniform(*self.snr_db_range)
        clean_p = float(np.mean(audio ** 2)) + 1e-8
        noise_p = float(np.mean(noise ** 2)) + 1e-8
        scale = np.sqrt(clean_p / (10 ** (snr / 10)) / noise_p)
        return (audio + scale * noise).astype(np.float32)

    def _apply_gain(self, audio):
        gain_db = random.uniform(*self.gain_db_range)
        return (audio * (10 ** (gain_db / 20))).astype(np.float32)

    def __call__(self, audio):
        if self.p_reverb and random.random() < self.p_reverb:
            audio = self._reverberate(audio)
        if self.p_noise and random.random() < self.p_noise:
            audio = self._add_noise(audio)
        if random.random() < self.p_gain:
            audio = self._apply_gain(audio)
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > 0.99:                       # 클리핑 방지
            audio = audio * (0.99 / peak)
        return audio.astype(np.float32)
