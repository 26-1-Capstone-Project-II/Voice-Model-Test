"""
STEP 6: ASR 학습 데이터 준비
==============================
WAV 파일 + JSON(Transcript)을 매칭해
Whisper 파인튜닝용 HuggingFace Dataset으로 변환합니다.

설치:
    pip install datasets librosa soundfile tqdm

실행:
    python 06_prepare_asr_data.py --data_root "C:\\...\\구음장애 음성인식 데이터" --output_dir ./asr_dataset
"""

import json
import argparse
import random
from pathlib import Path
from collections import defaultdict

from datasets import Dataset, DatasetDict, Audio
from tqdm import tqdm


# ────────────────────────────────────────────
# 1. WAV ↔ JSON 매칭
# ────────────────────────────────────────────

def load_pairs(data_root: Path) -> list[dict]:
    """
    JSON의 File_id 필드와 WAV 파일명을 매칭해 pair 리스트 반환.
    반환 형식:
        [{"audio_path": str, "transcript": str, "speaker_id": str,
          "disease_type": str, "severity": str}, ...]
    """
    json_files = sorted(data_root.rglob("*.json"))
    wav_index  = {f.stem: f for f in data_root.rglob("*.wav")}

    print(f"  JSON: {len(json_files):,}개 / WAV 인덱스: {len(wav_index):,}개")

    pairs = []
    no_wav   = 0
    no_trans = 0

    for jp in tqdm(json_files, desc="매칭 중"):
        try:
            with open(jp, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        transcript = data.get("Transcript", "").strip()
        if not transcript:
            no_trans += 1
            continue

        # File_id에서 stem 추출 (e.g. "ID-01-11-N-AMG-02-02-M-56-KK.wav" → stem)
        file_id   = data.get("File_id", "")
        wav_stem  = Path(file_id).stem  # 확장자 제거
        wav_path  = wav_index.get(wav_stem)

        if wav_path is None:
            no_wav += 1
            continue

        # 메타데이터 추출
        disease   = data.get("Disease_info", {})
        patient   = data.get("Patient_info", {})

        pairs.append({
            "audio_path"    : str(wav_path),
            "transcript"    : transcript,
            "speaker_id"    : wav_stem,                          # 파일명 = 화자+세션 고유값
            "disease_type"  : disease.get("Type", ""),           # "01", "02", "03"
            "subcategory"   : disease.get("Subcategory1", ""),   # "11", "21", "31" 등
            "sex"           : patient.get("Sex", ""),
            "age"           : patient.get("Age", ""),
            "area"          : patient.get("Area", ""),
        })

    print(f"  ✅ 매칭 성공: {len(pairs):,}개")
    print(f"  ⚠️  WAV 없음: {no_wav:,}개 / 전사 없음: {no_trans:,}개")
    return pairs


# ────────────────────────────────────────────
# 2. 화자 단위 Train / Val / Test 분할
# ────────────────────────────────────────────

def split_by_speaker(
    pairs: list[dict],
    train_ratio: float = 0.8,
    val_ratio:   float = 0.1,
    seed:        int   = 42,
) -> dict[str, list]:
    """
    동일 화자가 train/val/test에 섞이지 않도록 화자 단위 분할.
    """
    random.seed(seed)

    speaker_map = defaultdict(list)
    for p in pairs:
        speaker_map[p["speaker_id"]].append(p)

    speakers = list(speaker_map.keys())
    random.shuffle(speakers)

    n       = len(speakers)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

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

    for name, data in split.items():
        print(f"  {name:12s}: {len(data):,}개 샘플")

    return split


# ────────────────────────────────────────────
# 3. HuggingFace DatasetDict 저장
# ────────────────────────────────────────────

def save_dataset(split: dict, output_dir: Path):
    """
    HuggingFace Dataset으로 변환 후 저장.
    Audio 컬럼은 경로만 저장하고 실제 로딩은 학습 시 on-the-fly로 처리.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_dict = {}
    for split_name, records in split.items():
        ds = Dataset.from_list(records)
        # Audio 컬럼: 경로를 실제 오디오로 자동 디코딩 (sampling_rate=16000으로 리샘플링)
        ds = ds.cast_column("audio_path", Audio(sampling_rate=16000))
        dataset_dict[split_name] = ds

    dd = DatasetDict(dataset_dict)
    dd.save_to_disk(str(output_dir))
    print(f"\n💾 데이터셋 저장 완료: {output_dir}")
    print(f"   불러오기: datasets.load_from_disk('{output_dir}')")
    return dd


# ────────────────────────────────────────────
# 4. 데이터 분포 리포트
# ────────────────────────────────────────────

def print_report(pairs: list[dict]):
    from collections import Counter

    disease_map = {"01": "뇌신경장애", "02": "언어청각장애", "03": "후두장애"}
    subcat_map  = {
        "11": "중풍", "12": "파킨슨", "13": "뇌성마비", "14": "루게릭", "15": "기타뇌신경",
        "21": "조음", "22": "유창성", "23": "음성언어", "24": "언어발달", "25": "복합",
        "31": "기능성", "32": "기질성", "33": "신경성",
    }

    disease_counter = Counter(p["disease_type"] for p in pairs)
    subcat_counter  = Counter(p["subcategory"]  for p in pairs)
    sex_counter     = Counter(p["sex"]          for p in pairs)

    print("\n📊 데이터 분포")
    print("\n  [장애 유형]")
    for k, v in disease_counter.most_common():
        label = disease_map.get(k, k)
        print(f"    {label:15s}: {v:,}개")

    print("\n  [세부 유형]")
    for k, v in subcat_counter.most_common():
        label = subcat_map.get(k, k)
        print(f"    {label:15s} ({k}): {v:,}개")

    print("\n  [성별]")
    for k, v in sex_counter.most_common():
        print(f"    {k}: {v:,}개")


# ────────────────────────────────────────────
# 5. 메인
# ────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root",   required=True,           help="구음장애 데이터 루트")
    parser.add_argument("--output_dir",  default="./asr_dataset", help="Dataset 저장 경로")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio",   type=float, default=0.1)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--no_save",     action="store_true",     help="저장 없이 통계만 확인")
    args = parser.parse_args()

    print("\n🔗 WAV ↔ JSON 매칭 시작...")
    pairs = load_pairs(Path(args.data_root))

    if not pairs:
        print("❌ 매칭된 pair가 없습니다. WAV 압축 해제 완료 여부를 확인하세요.")
        return

    print_report(pairs)

    print("\n✂️  화자 단위 분할...")
    split = split_by_speaker(pairs, args.train_ratio, args.val_ratio, args.seed)

    if not args.no_save:
        save_dataset(split, Path(args.output_dir))

    print("\n✅ 완료! 다음: python 07_train_whisper.py")


if __name__ == "__main__":
    main()
