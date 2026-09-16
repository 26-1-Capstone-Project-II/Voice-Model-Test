"""
데이터셋 10% 샘플링
====================
화자 단위로 10% 샘플링해서 작은 데이터셋 생성.
원본 데이터는 유지됩니다.

실행:
    python sample_dataset.py --dataset_dir ./segmented_dataset --ratio 0.1
"""

import json
import argparse
import random
from pathlib import Path
from collections import defaultdict


def sample_by_speaker(records: list[dict], ratio: float, seed: int = 42) -> list[dict]:
    """화자 단위 샘플링 (data leakage 방지)"""
    random.seed(seed)
    speaker_map = defaultdict(list)
    for r in records:
        speaker_map[r.get("speaker_id", "unknown")].append(r)

    speakers = list(speaker_map.keys())
    n_sample = max(1, int(len(speakers) * ratio))
    sampled_speakers = random.sample(speakers, n_sample)

    result = []
    for sp in sampled_speakers:
        result.extend(speaker_map[sp])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="./segmented_dataset")
    parser.add_argument("--output_dir",  default="./sampled_dataset")
    parser.add_argument("--ratio",       type=float, default=0.1)
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    src = Path(args.dataset_dir)
    dst = Path(args.output_dir)
    dst.mkdir(exist_ok=True)

    print(f"📂 원본: {src}")
    print(f"💾 출력: {dst}")
    print(f"📊 샘플링 비율: {args.ratio * 100:.0f}%\n")

    total_before = 0
    total_after  = 0

    for split in ["train", "validation", "test"]:
        src_path = src / f"{split}.jsonl"
        if not src_path.exists():
            continue

        lines   = [l for l in src_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        records = [json.loads(l) for l in lines]
        total_before += len(records)

        sampled = sample_by_speaker(records, args.ratio, args.seed)
        total_after += len(sampled)

        dst_path = dst / f"{split}.jsonl"
        with open(dst_path, "w", encoding="utf-8") as f:
            for r in sampled:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        speakers_before = len(set(r.get("speaker_id") for r in records))
        speakers_after  = len(set(r.get("speaker_id") for r in sampled))

        print(f"  {split:12s}: {len(records):,}개 → {len(sampled):,}개  "
              f"({speakers_before}명 → {speakers_after}명 화자)")

    print(f"\n  전체: {total_before:,}개 → {total_after:,}개 ({total_after/total_before*100:.1f}%)")

    # 예상 학습 시간
    train_path = dst / "train.jsonl"
    n_train = sum(1 for _ in open(train_path, encoding="utf-8"))
    steps_per_epoch = n_train // 32
    est_hours = steps_per_epoch * 5 * 89 / 3600
    print(f"\n⏱️  예상 학습 시간 (5 epoch, 89초/스텝 기준): {est_hours:.0f}시간 ({est_hours/24:.1f}일)")
    print(f"\n✅ 완료! 파인튜닝 실행:")
    print(f"   python finetune_simple.py \\")
    print(f"     --wav_dir ./segmented_dataset/wavs \\")
    print(f"     --json_dir {dst} \\")
    print(f"     --batch_size 4 --grad_accum 8")


if __name__ == "__main__":
    main()
