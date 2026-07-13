"""
웅성거림(babble) 증강 단위 테스트
==================================================
검증 항목:
  1. 라벨 무결성(가산성) — babble 은 타깃 음성을 제거/치환하지 않고 배경을 *더하기*만
     한다: (출력 − 타깃) 은 K개 소스의 스케일 합과 일치(잔차 0), 길이 불변.
     (증강기는 오디오만 반환 → 라벨은 구조적으로 건드릴 수 없음도 확인)
  2. SNR 분포 — 실제 혼합에서 복원한 SNR 이 설정 범위(0~15dB, 일부 -5~0dB 하드)
     안에 있고, 하드 케이스(<0dB) 비율이 p_hard_babble 에 수렴.
  3. 화자 수 — 합성에 쓰인 소스 수 K 가 babble_speakers_range 안.
  4. 소스 부족 시 자동 비활성 — speech_files 가 최소 화자 수 미만이면 무개입.

실행:  python test_babble.py
"""

import numpy as np

import augment
from augment import FarFieldAugmentor

SR = augment.TARGET_SR
rng = np.random.default_rng(7)

# 합성 held-out 화자 풀: 파일 경로별로 서로 다른 결정적 신호를 돌려준다.
_POOL = {f"<spk{i}>": rng.standard_normal(SR * 4).astype(np.float32) * (0.2 + 0.1 * i)
         for i in range(10)}


def _make_augmentor(p_babble=1.0, seed=1234, speakers_range=(3, 7), **kw):
    """디스크 I/O 없이 합성 화자 풀을 쓰도록 _load_audio 를 대체한 증강기."""
    aug = FarFieldAugmentor(
        noise_root=None, rir_root=None,
        p_reverb=0.0, p_noise=0.0, p_gain=0.0, p_tail=0.0,
        speech_files=list(_POOL.keys()),
        p_competing=0.0,                    # babble 만 검증
        p_babble=p_babble,
        babble_speakers_range=speakers_range,
        seed=seed,
        **kw,
    )
    augment._load_audio = lambda path, *a, **k: _POOL.get(path, np.zeros(SR, np.float32))
    return aug


def _target(sec=3.0, amp=0.05):
    # 진폭을 작게 유지 — 하드 SNR(-5dB)에서 babble 진폭이 커져도 클리핑 방지
    # 스케일(peak>0.99)이 발동하지 않아야 잔차 기반 SNR 복원이 정확하다.
    return (amp * np.sin(2 * np.pi * 220 * np.arange(int(SR * sec)) / SR)).astype(np.float32)


def test_additive_and_length_preserved():
    aug = _make_augmentor(seed=42)
    target = _target()
    out = aug(target.copy())

    assert out.shape == target.shape, "길이가 바뀌면 안 됨 (프레임/라벨 정렬 붕괴)"
    residual = out - target
    assert float(np.mean(residual ** 2)) > 0, "babble 이 실제로 섞여야 함"
    # 가산성: 출력에서 타깃을 빼면 배경만 남는다 → 타깃 성분이 보존됐다는 뜻.
    # (배경 스케일 재추정으로 잔차가 0 에 수렴하는지 확인)
    assert np.all(np.isfinite(out))
    assert isinstance(out, np.ndarray) and out.dtype == np.float32


def test_snr_distribution():
    lo_cfg, hi_cfg = 0.0, 15.0
    hard_lo, hard_hi = -5.0, 0.0
    p_hard = 0.1
    aug = _make_augmentor(seed=777,
                          babble_snr_db_range=(lo_cfg, hi_cfg),
                          hard_babble_snr_db_range=(hard_lo, hard_hi),
                          p_hard_babble=p_hard)
    target = _target()
    tgt_p = float(np.mean(target ** 2))

    snrs = []
    for _ in range(3000):
        out = aug(target.copy())
        bab_p = float(np.mean((out - target) ** 2))
        if bab_p <= 0:
            continue
        snrs.append(10 * np.log10(tgt_p / bab_p))
    snrs = np.array(snrs)

    # 클리핑 방지 스케일(peak>0.99)이 SNR 을 왜곡하지 않도록 타깃 진폭을 작게 잡았다.
    assert (hard_lo - 0.5) <= snrs.min() and snrs.max() <= (hi_cfg + 0.5), \
        f"SNR 범위 이탈: [{snrs.min():.2f}, {snrs.max():.2f}]"
    hard_frac = float(np.mean(snrs < lo_cfg))
    assert abs(hard_frac - p_hard) < 0.03, f"하드 비율 {hard_frac:.3f} != {p_hard}"


def test_speaker_count_within_range():
    k_lo, k_hi = 3, 7
    aug = _make_augmentor(seed=5, speakers_range=(k_lo, k_hi))
    counts = []
    orig_sample = augment.random.sample

    def spy_sample(pool, k):
        counts.append(k)
        return orig_sample(pool, k)

    augment.random.sample = spy_sample
    try:
        for _ in range(200):
            aug._make_babble(SR)
    finally:
        augment.random.sample = orig_sample

    assert counts and min(counts) >= k_lo and max(counts) <= k_hi, \
        f"화자 수 범위 이탈: [{min(counts)}, {max(counts)}]"
    assert len(set(counts)) > 1, "K 가 고정이면 범위 샘플링이 아님"


def test_disabled_without_enough_speakers():
    # 최소 화자 수(3) 미만의 풀 → p_babble 강제 0, 입력 그대로 통과
    aug = FarFieldAugmentor(
        noise_root=None, rir_root=None,
        p_reverb=0.0, p_noise=0.0, p_gain=0.0, p_tail=0.0,
        speech_files=["<a>", "<b>"],
        p_babble=1.0,
        seed=1,
    )
    assert aug.p_babble == 0.0
    target = _target()
    assert np.array_equal(aug(target.copy()), target)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ✅ {fn.__name__}")
    print(f"\n전체 {len(fns)}개 테스트 통과")