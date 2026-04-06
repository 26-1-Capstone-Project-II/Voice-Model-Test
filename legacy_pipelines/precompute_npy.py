"""
WAV → numpy 사전 변환
=======================
학습 중 매번 WAV를 librosa로 디코딩하는 대신
미리 numpy 배열로 변환해서 저장합니다.

np.load()는 librosa.load()보다 10~20배 빠릅니다.

실행:
    python precompute_npy.py --dataset_dir ./segmented_dataset
"""

import json
import argparse
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

TARGET_SR = 16000
MAX_SEC   = 10.0


def convert_one(record: dict, npy_dir: Path) -> dict | None:
    """WAV 1개 → npy로 변환. 이미 있으면 스킵."""
    wav_path = record.get("wav_path", "")
    if not wav_path or not Path(wav_path).exists():
        return None

    # npy 저장 경로: segmented_dataset/npys/파일명.npy
    npy_name = Path(wav_path).stem + ".npy"
    npy_path = npy_dir / npy_name

    if not npy_path.exists():
        try:
            audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
            if sr != TARGET_SR:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio[:int(MAX_SEC * TARGET_SR)]
            np.save(str(npy_path), audio)
        except Exception as e:
            return None

    record["npy_path"] = str(npy_path)
    return record


def process_split(jsonl_path: Path, npy_dir: Path, num_workers: int = 4):
    """JSONL 파일의 모든 WAV를 npy로 변환하고 npy_path 필드 추가"""
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    lines = [l for l in lines if l.strip()]
    records = [json.loads(l) for l in lines]

    print(f"\n  [{jsonl_path.name}] {len(records):,}개 처리 중...")

    updated = []
    failed  = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(convert_one, rec, npy_dir): rec
            for rec in records
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                          desc=f"  {jsonl_path.stem}", leave=False):
            result = future.result()
            if result:
                updated.append(result)
            else:
                failed += 1

    # npy_path 추가된 JSONL 덮어쓰기
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for rec in updated:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"    ✅ 완료: {len(updated):,}개 / 실패: {failed:,}개")
    return len(updated)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="./segmented_dataset")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="병렬 변환 스레드 수")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    npy_dir     = dataset_dir / "npys"
    npy_dir.mkdir(exist_ok=True)

    print(f"📂 데이터셋: {dataset_dir}")
    print(f"💾 npy 저장: {npy_dir}")

    # 전체 WAV 수 계산
    total_wav = sum(
        1 for jp in dataset_dir.glob("*.jsonl")
        for line in jp.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    print(f"🎵 총 WAV: {total_wav:,}개")

    # 예상 시간
    est_min = total_wav / (args.num_workers * 3) / 60
    print(f"⏱️  예상 소요: 약 {est_min:.0f}분 ({args.num_workers}스레드 기준)\n")

    # split별 변환
    total = 0
    for split in ["train", "validation", "test"]:
        jsonl_path = dataset_dir / f"{split}.jsonl"
        if jsonl_path.exists():
            total += process_split(jsonl_path, npy_dir, args.num_workers)

    print(f"\n✅ 전체 완료: {total:,}개")
    print(f"\n이제 파인튜닝 실행:")
    print(f"  python finetune_simple.py \\")
    print(f"    --wav_dir {dataset_dir}/wavs \\")
    print(f"    --json_dir {dataset_dir} \\")
    print(f"    --batch_size 4 --grad_accum 8")


if __name__ == "__main__":
    main()
