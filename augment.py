"""
원거리/소음/경쟁화자 환경 증강 (train split 전용)
==========================================
clean → (+competing speech@SIR) → (reverb) → (+noise@SNR) → (gain) → 클리핑 방지

목적: Zeroth-Korean clean read speech 만 본 모델에 잡음·울림·거리·경쟁 화자 도메인을
      주입해 원거리/소음/동석자 환경 강인성(robustness)을 확보한다.

코퍼스:
  - MUSAN  (OpenSLR 17): 배경 소음/음악/웅성거림  → noise_root
  - RIRS_NOISES (OpenSLR 28): 방 임펄스 응답(거리감) → rir_root
  - 경쟁 화자: Zeroth held-out(test 스플릿) 발화 → speech_files
    (OpenSLR-40 train/test 는 화자 배타적 → held-out 요건 충족, 라벨 무결성 자동)

경쟁 화자 증강(On-Voice 앱 화자 게이트 보완, 마스터 플랜 §7):
  게이트를 통과한 잔여 경쟁 화자에도 강인하도록, 타깃 발화에 held-out 다른 화자 음성을
  타깃 우세 SIR(5~20dB, 일부 0~5dB 하드)로 **부분 겹침(onset/offset, full-overlap 없음)**
  으로 섞되 라벨은 타깃 발화만 유지 → 모델이 "더 우세한 화자만 전사"하도록 학습.

순서(플랜 §7: 경쟁 믹스 → RIR → MUSAN → [모델]SpecAugment):
  경쟁 화자를 먼저 섞어 "두 화자가 함께 도달한 원신호"를 만든 뒤 방 울림(RIR)을 입히고,
  그 위에 (마이크 위치에서 더해지는) 소음을 더한다.
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
    """clean → (+competing speech) → (reverb) → (+noise@SNR) → (gain) → (+비음성 꼬리).
    train split 에서만 사용."""

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
        p_tail=0.3,
        tail_sec_range=(2.0, 10.0),
        tail_snr_db_range=(5.0, 15.0),
        max_total_sec=30.0,
        # ── 경쟁 화자(간섭 화자) 증강 (플랜 §7) ──
        # speech_files: held-out(Zeroth test) 간섭 화자 wav 경로 목록. 없으면 자동 비활성.
        speech_files=None,
        p_competing=0.0,
        sir_db_range=(5.0, 20.0),          # 타깃 우세 SIR (타깃 − 간섭 dB)
        hard_sir_db_range=(0.0, 5.0),      # 근접 간섭 하드 케이스
        p_hard_sir=0.1,                    # 하드 케이스 비율
        competing_overlap_frac_range=(0.3, 0.85),  # 부분 겹침 비율 (1.0 미만 → full-overlap 없음)
        p_competing_own_rir=0.0,           # 경쟁 시, 간섭 화자를 타깃과 *다른* RIR 로 (공간 분리)
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
        # 비음성 꼬리: 음성 종료 후 윈도우 잔여 구간을 노이즈/무음으로 채운다.
        # 라벨은 음성 부분만 유지 → 모델이 "비음성 = 출력 없음(EOS)"을 학습 → 후반부 환각 억제.
        self.p_tail = p_tail
        self.tail_sec_range = tail_sec_range
        self.tail_snr_db_range = tail_snr_db_range
        self.max_total_sec = max_total_sec
        # 경쟁 화자: 간섭원 파일이 있어야만 활성 (없으면 p_competing=0 강제).
        self.speech_files = list(speech_files) if speech_files else []
        self.p_competing = p_competing if self.speech_files else 0.0
        self.sir_db_range = sir_db_range
        self.hard_sir_db_range = hard_sir_db_range
        self.p_hard_sir = p_hard_sir
        self.competing_overlap_frac_range = competing_overlap_frac_range
        # 간섭 화자 별도 RIR 은 RIR 파일이 있어야만 의미 (없으면 0)
        self.p_competing_own_rir = p_competing_own_rir if self.rir_files else 0.0
        if seed is not None:
            random.seed(seed)
        print(f"[Augmentor] noise={len(self.noise_files)} rir={len(self.rir_files)} "
              f"speech(competing)={len(self.speech_files)} "
              f"p_reverb={self.p_reverb} p_noise={self.p_noise} p_tail={self.p_tail} "
              f"p_competing={self.p_competing} p_competing_own_rir={self.p_competing_own_rir}")

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

    def _add_competing_speech(self, audio, own_rir=False):
        """held-out 다른 화자 발화를 부분 겹침(onset/offset)으로 타깃 우세 SIR 로 섞는다.

        - 라벨은 건드리지 않는다(반환은 오디오만). 타깃 전사만 남긴다는 원칙(플랜 §1·§7).
        - 부분 겹침: 간섭원이 타깃 타임라인의 일부 구간만 덮는다(full-overlap 없음).
        - SIR 은 겹치는 구간의 파워로 정의 → "그 구간에서 타깃이 얼마나 우세한가".
        - own_rir=True: 간섭 화자를 (타깃과 무관하게 새로 뽑은) 별도 RIR 로 울려 공간 분리
          화자를 모사(플랜 §7 옵션). 이때 타깃은 호출부에서 이미 RIR_a 로 울린 상태여야 한다.
        """
        L = len(audio)
        if L == 0:
            return audio
        interferer = _load_audio(random.choice(self.speech_files))
        if len(interferer) == 0:
            return audio
        if own_rir and self.rir_files:
            interferer = self._reverberate(interferer)   # 간섭 화자 자기 방(RIR_b)
            if len(interferer) == 0:
                return audio

        # 부분 겹침 구간 길이·위치 (타깃 윈도우 안에 배치 → full-overlap 방지)
        frac = random.uniform(*self.competing_overlap_frac_range)
        seg_len = max(1, min(L, int(frac * L)))
        interf_seg = _match_length(interferer, seg_len)
        start = random.randint(0, max(0, L - seg_len))

        # 타깃 우세 SIR (일부는 근접 하드 케이스)
        if random.random() < self.p_hard_sir:
            sir = random.uniform(*self.hard_sir_db_range)
        else:
            sir = random.uniform(*self.sir_db_range)

        tgt_region = audio[start:start + seg_len]
        tgt_p = float(np.mean(tgt_region ** 2)) + 1e-8
        interf_p = float(np.mean(interf_seg ** 2)) + 1e-8
        scale = np.sqrt(tgt_p / (10 ** (sir / 10)) / interf_p)

        mixed = audio.copy()
        mixed[start:start + seg_len] = tgt_region + scale * interf_seg
        return mixed.astype(np.float32)

    def _apply_gain(self, audio):
        gain_db = random.uniform(*self.gain_db_range)
        return (audio * (10 ** (gain_db / 20))).astype(np.float32)

    def _append_tail(self, audio):
        """음성 뒤에 비음성(노이즈 또는 무음) 꼬리를 붙인다. 라벨은 그대로(음성 부분만)."""
        sr = TARGET_SR
        room = int(self.max_total_sec * sr) - len(audio)
        if room <= int(0.5 * sr):             # 이미 윈도우가 거의 찼으면 생략
            return audio
        tail_sec = random.uniform(*self.tail_sec_range)
        tail_len = min(int(tail_sec * sr), room)
        if self.noise_files and random.random() < 0.8:
            noise = _match_length(_load_audio(random.choice(self.noise_files)), tail_len)
            snr = random.uniform(*self.tail_snr_db_range)
            clean_p = float(np.mean(audio ** 2)) + 1e-8
            noise_p = float(np.mean(noise ** 2)) + 1e-8
            scale = np.sqrt(clean_p / (10 ** (snr / 10)) / noise_p)
            tail = (scale * noise).astype(np.float32)
        else:
            tail = np.zeros(tail_len, dtype=np.float32)   # 일부는 순수 무음 꼬리
        return np.concatenate([audio, tail]).astype(np.float32)

    def __call__(self, audio):
        # 경쟁 화자를 먼저 섞고(두 화자가 함께 마이크로 향하는 원신호), 그 위에 방 울림·소음.
        did_reverb = False
        if self.p_competing and random.random() < self.p_competing:
            use_own_rir = (self.p_competing_own_rir > 0.0
                           and random.random() < self.p_competing_own_rir)
            if use_own_rir:
                # 공간 분리: 타깃(RIR_a)·간섭(RIR_b)을 각기 다른 방으로 울린 뒤 마이크에서 합산.
                audio = self._reverberate(audio)                        # 타깃 RIR_a
                audio = self._add_competing_speech(audio, own_rir=True)  # 간섭 RIR_b (내부 적용)
                did_reverb = True                                       # 아래 공유 reverb 스킵
            else:
                # base: 건식 혼합 → 아래 reverb 스테이지가 두 화자를 같은 방(공유 RIR)으로 울림.
                audio = self._add_competing_speech(audio, own_rir=False)
        if not did_reverb and self.p_reverb and random.random() < self.p_reverb:
            audio = self._reverberate(audio)
        if self.p_noise and random.random() < self.p_noise:
            audio = self._add_noise(audio)
        if random.random() < self.p_gain:
            audio = self._apply_gain(audio)
        # 비음성 꼬리는 (잔향·소음 적용 후) 마지막에 붙인다.
        if self.p_tail and random.random() < self.p_tail:
            audio = self._append_tail(audio)
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > 0.99:                       # 클리핑 방지
            audio = audio * (0.99 / peak)
        return audio.astype(np.float32)
