"""
STEP A: 파인튜닝용 데이터 준비 (MeCab-free 버전)
==================================================
g2pk MeCab 빌드 실패 시에도 동작합니다.
korean_g2p_nomecab.py의 load_g2p()가 자동으로 폴백 처리합니다.

실행:
    python A_prepare_finetune_data.py \
        --data_root "C:\\Users\\User\\Voice-Model-Test\\구음장애 음성인식 데이터" \
        --output_dir ./finetune_dataset
"""

import json
import argparse
import random
from pathlib import Path
from collections import defaultdict, Counter

from tqdm import tqdm

# ── g2pk / 폴백 자동 선택
from korean_g2p_nomecab import load_g2p


def load_pairs(data_root: Path, g2p, max_duration_sec: float = 30.0) -> list[dict]:
    json_files = sorted(data_root.rglob("*.json"))
    wav_index  = {f.stem: f for f in data_root.rglob("*.wav")}

    print(f"\n📂 JSON: {len(json_files):,}개 / WAV 인덱스: {len(wav_index):,}개")

    pairs   = []
    skipped = Counter()

    for jp in tqdm(json_files, desc="데이터 로딩 + G2P 변환"):
        try:
            with open(jp, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            skipped["json_read_error"] += 1
            continue

        transcript = data.get("Transcript", "").strip()
        if not transcript:
            skipped["no_transcript"] += 1
            continue

        file_id  = data.get("File_id", "")
        wav_stem = Path(file_id).stem
        wav_path = wav_index.get(wav_stem)
        if wav_path is None:
            skipped["no_wav"] += 1
            continue

        play_time = data.get("playTime", 0)
        if play_time > max_duration_sec:
            skipped["too_long"] += 1
            continue

        try:
            g2p_text = g2p(transcript, descriptive=True).strip()
        except Exception:
            g2p_text = transcript

        disease = data.get("Disease_info", {})
        patient = data.get("Patient_info", {})

        pairs.append({
            "audio_path"   : str(wav_path),
            "transcript"   : transcript,
            "label"        : g2p_text,
            "duration"     : play_time,
            "speaker_id"   : wav_stem,
            "disease_type" : disease.get("Type", ""),
            "subcategory"  : disease.get("Subcategory1", ""),
            "sex"          : patient.get("Sex", ""),
            "age"          : patient.get("Age", ""),
        })

    print(f"\n✅ 매칭 성공: {len(pairs):,}개")
    for reason, count in skipped.most_common():
        print(f"  ⚠️  {reason}: {count:,}개 스킵")
    return pairs


def split_by_speaker(pairs: list[dict], seed: int = 42) -> dict:
    random.seed(seed)
    speaker_map = defaultdict(list)
    for p in pairs:
        speaker_map[p["speaker_id"]].append(p)

    speakers = list(speaker_map.keys())
    random.shuffle(speakers)
    n       = len(speakers)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)
    train_sp = set(speakers[:n_train])
    val_sp   = set(speakers[n_train:n_train + n_val])

    split = {"train": [], "validation": [], "test": []}
    for sp, sp_pairs in speaker_map.items():
        if sp in train_sp:
            split["train"].extend(sp_pairs)
        elif sp in val_sp:
            split["validation"].extend(sp_pairs)
        else:
            split["test"].extend(sp_pairs)

    print(f"\n✂️  화자 단위 분할:")
    for name, data in split.items():
        print(f"  {name:12s}: {len(data):,}개 샘플")
    return split


def save_jsonl(split: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, records in split.items():
        out_path = output_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  💾 {out_path.name}: {len(records):,}개")

    all_pairs = [p for records in split.values() for p in records]
    stats = {
        "total": len(all_pairs),
        "disease_distribution": dict(Counter(p["disease_type"] for p in all_pairs)),
        "sex_distribution":     dict(Counter(p["sex"] for p in all_pairs)),
        "duration_mean_sec":    sum(p["duration"] for p in all_pairs) / max(len(all_pairs), 1),
    }
    with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    disease_map = {"01": "뇌신경", "02": "언어청각", "03": "후두"}
    print(f"\n📊 장애 유형 분포:")
    for k, v in stats["disease_distribution"].items():
        print(f"  {disease_map.get(k, k)}: {v:,}개")
    print(f"  평균 발화 길이: {stats['duration_mean_sec']:.1f}초")


def preview_g2p_samples(pairs: list[dict], n: int = 10):
    print(f"\n🔍 G2P 변환 샘플 (상위 {n}개):")
    print(f"  {'원본':^30} → {'G2P 변환':^30}")
    print("  " + "-" * 65)
    for p in pairs[:n]:
        orig    = p["transcript"][:28]
        g2p_res = p["label"][:28]
        changed = "✅" if orig != g2p_res else "  "
        print(f"  {orig:^28} → {g2p_res:^28} {changed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",    required=True)
    parser.add_argument("--output_dir",   default="./finetune_dataset")
    parser.add_argument("--max_duration", type=float, default=30.0)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--preview_only", action="store_true")
    args = parser.parse_args()

    print("🔤 G2P 로딩...")
    g2p = load_g2p()

    pairs = load_pairs(Path(args.data_root), g2p, args.max_duration)
    if not pairs:
        print("❌ 매칭된 pair가 없습니다. WAV 압축 해제 완료 여부를 확인하세요.")
        return

    preview_g2p_samples(pairs)

    if args.preview_only:
        return

    split = split_by_speaker(pairs, args.seed)
    print(f"\n💾 저장: {args.output_dir}")
    save_jsonl(split, Path(args.output_dir))
    print("\n✅ 완료! 다음: python B_finetune_wav2vec2.py")


if __name__ == "__main__":
    main()
