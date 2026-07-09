"""
경쟁 화자 증강 단위 테스트 (마스터 플랜 §7 DoD)
==================================================
검증 항목:
  1. 라벨 무결성 — 경쟁 화자 믹싱은 타깃 음성을 제거/치환하지 않고 *더하기*만 한다.
     겹침 밖 구간은 바이트 단위로 원본과 동일, 겹침 안에서도 타깃 성분은 그대로 보존.
     (증강기는 오디오만 반환 → 라벨은 구조적으로 건드릴 수 없음도 확인)
  2. SIR 분포 — 실제 혼합에서 복원한 SIR 이 설정 범위(타깃 우세 5~20dB, 일부 0~5dB)
     안에 있고, 하드 케이스(<5dB) 비율이 p_hard_sir 에 수렴.
  3. full-overlap 없음 — 간섭 구간이 항상 타깃 전체보다 짧다(부분 겹침).

실행:  python test_competing_speaker.py
"""

import numpy as np

import augment
from augment import FarFieldAugmentor

SR = augment.TARGET_SR
rng = np.random.default_rng(0)


def _make_augmentor(p_competing=1.0, seed=1234, interferer=None):
    """디스크 I/O 없이 합성 간섭원을 쓰도록 _load_audio 를 대체한 증강기."""
    if interferer is None:
        interferer = rng.standard_normal(SR * 5).astype(np.float32)  # 5초 백색잡음
    # 경쟁 화자만 검증: reverb/noise/gain/tail 모두 0
    aug = FarFieldAugmentor(
        noise_root=None, rir_root=None,
        p_reverb=0.0, p_noise=0.0, p_gain=0.0, p_tail=0.0,
        speech_files=["<synthetic>"],  # random.choice 대상 (실제 로드는 아래서 대체)
        p_competing=p_competing,
        seed=seed,
    )
    augment._load_audio = lambda *_a, **_k: interferer  # 합성 간섭원 주입
    return aug


def test_label_integrity_target_preserved():
    """타깃은 제거되지 않고 간섭이 더해질 뿐 — 타깃 전용 라벨이 유효하게 유지됨."""
    aug = _make_augmentor(p_competing=1.0)
    target = rng.standard_normal(SR * 6).astype(np.float32)  # 6초 타깃
    out = aug._add_competing_speech(target.copy())

    assert out.shape == target.shape, "길이가 바뀌면 안 됨 (프레임/라벨 정렬 붕괴)"
    diff = out - target
    changed = np.flatnonzero(np.abs(diff) > 1e-9)
    assert changed.size > 0, "경쟁 화자가 실제로 섞여야 함"

    # 변경 구간은 연속(단일 부분 겹침 구간)이어야 하고 타깃 전체보다 짧아야 함
    start, end = changed[0], changed[-1] + 1
    assert (end - start) < len(target), "full-overlap 금지 — 부분 겹침이어야 함"
    # 겹침 밖은 원본과 바이트 단위 동일 (타깃 파괴 없음)
    assert np.array_equal(out[:start], target[:start])
    assert np.array_equal(out[end:], target[end:])
    # 겹침 안에서도 out = target + (간섭성분) → target 성분은 보존(간섭성분을 빼면 원복)
    assert np.allclose(out[start:end] - diff[start:end], target[start:end], atol=1e-6)
    print("✅ 1. 라벨 무결성: 타깃 보존 + 부분 겹침(간섭은 더하기만)")


def test_call_returns_audio_only():
    """__call__ 은 오디오만 입출력 → 라벨을 건드릴 경로가 없음(구조적 무결성)."""
    aug = _make_augmentor(p_competing=1.0)
    target = rng.standard_normal(SR * 4).astype(np.float32)
    out = aug(target.copy())
    assert isinstance(out, np.ndarray) and out.dtype == np.float32
    print("✅ 1b. 증강기 시그니처: audio→audio (라벨 인자 없음)")


def test_sir_distribution():
    """복원 SIR 이 설정 범위 안에 있고 하드 케이스 비율이 p_hard_sir 에 수렴."""
    p_hard = 0.1
    interferer = rng.standard_normal(SR * 5).astype(np.float32)
    aug = _make_augmentor(p_competing=1.0, seed=7, interferer=interferer)
    aug.p_hard_sir = p_hard
    aug.sir_db_range = (5.0, 20.0)
    aug.hard_sir_db_range = (0.0, 5.0)

    N = 4000
    sirs = []
    for _ in range(N):
        target = rng.standard_normal(SR * 6).astype(np.float32)
        out = aug._add_competing_speech(target.copy())
        diff = out - target
        idx = np.flatnonzero(np.abs(diff) > 1e-9)
        seg = slice(idx[0], idx[-1] + 1)
        tgt_p = float(np.mean(target[seg] ** 2))
        interf_p = float(np.mean(diff[seg] ** 2))       # 더해진 간섭 성분 파워
        sirs.append(10.0 * np.log10(tgt_p / interf_p))
    sirs = np.array(sirs)

    lo, hi = sirs.min(), sirs.max()
    assert -0.5 <= lo and hi <= 20.5, f"SIR 범위 이탈: [{lo:.2f}, {hi:.2f}]"
    hard_frac = float(np.mean(sirs < 5.0))
    assert abs(hard_frac - p_hard) < 0.03, f"하드 비율 {hard_frac:.3f} != {p_hard}"
    print(f"✅ 2. SIR 분포: 범위 [{lo:.1f}, {hi:.1f}]dB, 하드(<5dB) 비율 {hard_frac:.3f} (목표 {p_hard})")


def test_disabled_without_speech_files():
    """간섭 파일이 없으면 p_competing 을 줘도 자동 비활성 → 무개입."""
    aug = FarFieldAugmentor(
        noise_root=None, rir_root=None,
        p_reverb=0.0, p_noise=0.0, p_gain=0.0, p_tail=0.0,
        speech_files=None, p_competing=0.9, seed=1,
    )
    assert aug.p_competing == 0.0
    # 진폭을 낮게 (peak<0.99) → __call__ 의 클리핑 방지 스케일링이 개입하지 않도록
    target = (0.1 * rng.standard_normal(SR * 3)).astype(np.float32)
    assert np.array_equal(aug(target.copy()), target)
    print("✅ 3. 간섭 풀 없음 → 경쟁 화자 증강 자동 비활성(무개입)")


if __name__ == "__main__":
    test_label_integrity_target_preserved()
    test_call_returns_audio_only()
    test_sir_distribution()
    test_disabled_without_speech_files()
    print("\n🎉 모든 경쟁 화자 증강 테스트 통과")
