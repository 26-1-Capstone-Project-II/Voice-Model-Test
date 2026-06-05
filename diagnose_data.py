"""
데이터 무결성 진단
==================
학습이 "무음 → 텍스트" 로 망가졌는지 확인.

검사 항목:
  1. 각 split 의 wav_path 가 실제로 존재하는 비율
  2. 샘플을 실제 로드해서 (a) 로드 성공률 (b) 무음(저에너지) 비율
  3. jsonl 의 duration 과 실제 오디오 길이 일치 여부

실행:
    python diagnose_data.py
    python diagnose_data.py --num_check 500
"""

import json
import argparse
import numpy as np
from pathlib import Path

HOME = Path.home()
SEGMENT_DIR = HOME / "mingly_workspace" / "Voice-Model-Test" / "segmented_dataset"
TARGET_SR = 16000


def check_split(name, path, num_check):
    if not path.exists():
        print(f"\n❌ {name}: {path} 없음")
        return

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line.strip()))
            except Exception:
                continue

    n = len(records)
    print(f"\n{'='*60}\n  📂 {name}: {n:,}개 레코드")

    # ── 1. 경로 존재율 (전수) ──
    exists = 0
    missing_examples = []
    for r in records:
        wp = r.get("wav_path", "")
        if wp and Path(wp).exists():
            exists += 1
        elif len(missing_examples) < 3:
            missing_examples.append(wp)
    print(f"  ✅ 존재: {exists:,}/{n:,} ({100*exists/n:.1f}%)")
    if exists < n:
        print(f"  ❌ 없음: {n-exists:,}개  예) {missing_examples}")

    # ── 2. 실제 로드 + 무음 검사 (샘플) ──
    import librosa
    import random
    random.seed(0)
    sample = random.sample(records, min(num_check, n))

    load_ok, load_fail, silent = 0, 0, 0
    dur_mismatch = 0
    for r in sample:
        wp = r.get("wav_path", "")
        try:
            audio, _ = librosa.load(wp, sr=TARGET_SR, mono=True)
            load_ok += 1
            rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0
            if rms < 1e-4:                      # 사실상 무음
                silent += 1
            real_dur = len(audio) / TARGET_SR
            exp_dur = float(r.get("duration", 0))
            if exp_dur > 0 and abs(real_dur - exp_dur) > 0.5:
                dur_mismatch += 1
        except Exception:
            load_fail += 1

    ns = len(sample)
    print(f"  🔊 샘플 {ns}개 로드: 성공 {load_ok} / 실패 {load_fail}")
    print(f"     무음(RMS<1e-4): {silent}/{load_ok}")
    print(f"     길이 불일치(>0.5s): {dur_mismatch}/{load_ok}")

    # 판정
    fail_rate = (load_fail + silent) / ns if ns else 1.0
    if fail_rate > 0.3:
        print(f"  🚨 무음/실패 비율 {100*fail_rate:.0f}% → 학습이 망가질 수준!")
    else:
        print(f"  🟢 오디오는 대체로 정상 (무음/실패 {100*fail_rate:.0f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_dir", type=str, default=str(SEGMENT_DIR))
    parser.add_argument("--num_check", type=int, default=300,
                        help="split 당 실제 로드 검사할 샘플 수")
    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    print(f"🔍 데이터 진단: {json_dir}")
    for name in ["train", "validation", "test"]:
        check_split(name, json_dir / f"{name}.jsonl", args.num_check)
