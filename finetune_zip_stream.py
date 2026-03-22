"""
언어청각장애 전용 파인튜닝 — ZIP 스트리밍 버전
================================================
TS02_언어청각장애.zip을 압축 해제하지 않고 직접 읽어 학습합니다.
디스크 추가 사용량: ~0GB (ZIP 그대로 사용)

파일 위치 가정:
  라벨(JSON): .../라벨링데이터_250331_add/TL02_언어청각장애/21.조음/...
  원천(WAV) : .../원천데이터/TS02_언어청각장애.zip

실행:
    python finetune_zip_stream.py \
        --zip_path "C:\\...\\원천데이터\\TS02_언어청각장애.zip" \
        --label_dir "C:\\...\\라벨링데이터_250331_add\\TL02_언어청각장애" \
        --output_dir ./finetuned_model
"""

import io
import json
import zipfile
import argparse
import random
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict, Counter
from typing import Any

import torch
import numpy as np
import librosa
import evaluate
from torch.utils.data import Dataset
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from tqdm import tqdm

from korean_g2p_nomecab import load_g2p


BASE_MODEL = "w11wo/wav2vec2-xls-r-300m-korean"
TARGET_SR  = 16000
MAX_SEC    = 30.0   # 학습 시 오디오 앞 30초만 사용 (세션 녹음 전체 사용 시 메모리 부족)


# ────────────────────────────────────────────
# 1. ZIP 인덱스 빌드 (파일명 → ZIP 내 경로 매핑)
# ────────────────────────────────────────────

def build_zip_index(zip_path: Path) -> dict[str, str]:
    """
    ZIP 안의 WAV 파일 목록을 미리 인덱싱.
    stem(확장자 없는 이름) → zip 내부 경로
    예) "ID-02-21-N-BGW-06-01-M-56-GS2" → "21.조음/ID-02-21-...wav"
    """
    print(f"📦 ZIP 인덱스 빌드 중: {zip_path.name}")
    index = {}
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.lower().endswith(".wav"):
                stem = Path(name).stem
                index[stem] = name
    print(f"  WAV {len(index):,}개 인덱싱 완료")
    return index


# ────────────────────────────────────────────
# 2. JSON 라벨 로딩 + ZIP과 매칭
# ────────────────────────────────────────────

def load_pairs(label_dir: Path, zip_index: dict, g2p) -> list[dict]:
    """
    JSON 라벨 디렉토리에서 File_id를 읽어 zip_index와 매칭.
    라벨은 G2P 변환해서 저장.
    """
    json_files = sorted(label_dir.rglob("*.json"))
    print(f"\n📂 JSON 라벨: {len(json_files):,}개 발견")

    pairs   = []
    skipped = Counter()

    for jp in tqdm(json_files, desc="라벨 로딩 + G2P"):
        try:
            with open(jp, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            skipped["read_error"] += 1
            continue

        transcript = data.get("Transcript", "").strip()
        if not transcript:
            skipped["no_transcript"] += 1
            continue

        # ZIP 내 WAV 찾기
        file_id  = data.get("File_id", "")
        wav_stem = Path(file_id).stem
        zip_member = zip_index.get(wav_stem)
        if zip_member is None:
            skipped["no_wav_in_zip"] += 1
            continue

        # playTime은 세션 전체 길이(초) — 13~45분짜리 긴 녹음
        # __getitem__에서 MAX_SEC(30초) 단위로 자르므로 필터 불필요
        play_time = data.get("playTime", 0)

        try:
            label = g2p(transcript, descriptive=True).strip()
        except Exception:
            label = transcript

        patient = data.get("Patient_info", {})

        pairs.append({
            "zip_member"  : zip_member,   # ZIP 내부 경로
            "transcript"  : transcript,
            "label"       : label,
            "duration"    : play_time,
            "speaker_id"  : wav_stem,
            "sex"         : patient.get("Sex", ""),
            "age"         : patient.get("Age", ""),
        })

    print(f"  ✅ 매칭: {len(pairs):,}개")
    for k, v in skipped.most_common():
        print(f"  ⚠️  {k}: {v:,}개 스킵")
    return pairs


# ────────────────────────────────────────────
# 3. 화자 단위 분할
# ────────────────────────────────────────────

def split_by_speaker(pairs: list[dict], seed: int = 42) -> dict:
    random.seed(seed)
    speaker_map = defaultdict(list)
    for p in pairs:
        speaker_map[p["speaker_id"]].append(p)

    speakers = list(speaker_map.keys())
    random.shuffle(speakers)
    n = len(speakers)
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    train_sp = set(speakers[:n_train])
    val_sp   = set(speakers[n_train:n_train + n_val])

    split = {"train": [], "validation": [], "test": []}
    for sp, sp_pairs in speaker_map.items():
        key = "train" if sp in train_sp else ("validation" if sp in val_sp else "test")
        split[key].extend(sp_pairs)

    print(f"\n✂️  화자 단위 분할 ({n}명):")
    for name, data in split.items():
        print(f"  {name:12s}: {len(data):,}개")
    return split


# ────────────────────────────────────────────
# 4. ZIP 스트리밍 Dataset
# ────────────────────────────────────────────

class ZipStreamDataset(Dataset):
    """
    ZIP 파일을 압축 해제하지 않고 직접 WAV를 메모리에 읽어 학습.
    zipfile.ZipFile은 thread-safe하지 않으므로 __getitem__에서 매번 열기.
    """

    def __init__(
        self,
        records    : list[dict],
        zip_path   : Path,
        processor  : Wav2Vec2Processor,
    ):
        self.records   = records
        self.zip_path  = zip_path
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        # ZIP에서 WAV 바이트 읽기 (압축 해제 없이)
        try:
            with zipfile.ZipFile(self.zip_path, "r") as zf:
                wav_bytes = zf.read(rec["zip_member"])
            audio, _ = librosa.load(io.BytesIO(wav_bytes), sr=TARGET_SR, mono=True)
            audio = audio[:int(MAX_SEC * TARGET_SR)]
        except Exception:
            audio = np.zeros(TARGET_SR, dtype=np.float32)

        input_values = self.processor(
            audio,
            sampling_rate  = TARGET_SR,
            return_tensors = "pt",
            padding        = False,
        ).input_values[0]

        with self.processor.as_target_processor():
            labels = self.processor(rec["label"]).input_ids

        return {
            "input_values": input_values,
            "labels"      : torch.tensor(labels, dtype=torch.long),
        }


# ────────────────────────────────────────────
# 5. 데이터 콜레이터
# ────────────────────────────────────────────

@dataclass
class DataCollatorCTCWithPadding:
    processor: Any

    def __call__(self, features: list[dict]) -> dict:
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids":    f["labels"]}        for f in features]

        batch = self.processor.pad(
            input_features, padding=True, return_tensors="pt"
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features, padding=True, return_tensors="pt"
            )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


# ────────────────────────────────────────────
# 6. 평가 지표
# ────────────────────────────────────────────

def make_compute_metrics(processor):
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids  = np.argmax(pred.predictions, axis=-1)
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)

        # 샘플 3개 출력
        for i in range(min(3, len(pred_str))):
            print(f"\n  정답: {label_str[i][:50]}")
            print(f"  예측: {pred_str[i][:50]}")

        return {"cer": round(cer_metric.compute(
            predictions=pred_str, references=label_str
        ), 4)}

    return compute_metrics


# ────────────────────────────────────────────
# 7. 학습
# ────────────────────────────────────────────

def train(
    zip_path   : Path,
    label_dir  : Path,
    output_dir : Path,
    batch_size : int   = 8,
    num_epochs : int   = 30,
    lr         : float = 1e-4,
    grad_accum : int   = 4,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n💻 디바이스: {device}")
    if device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU: {torch.cuda.get_device_name(0)}  ({vram:.1f}GB)")

    # G2P 로드
    print("\n🔤 G2P 로딩...")
    g2p = load_g2p()

    # ZIP 인덱스
    zip_index = build_zip_index(zip_path)

    # 라벨 로딩 + 매칭
    pairs = load_pairs(label_dir, zip_index, g2p)
    if not pairs:
        print("❌ 매칭된 pair 없음. 경로를 확인하세요.")
        return

    split = split_by_speaker(pairs)

    # 모델 & 프로세서
    print(f"\n📥 모델 로딩: {BASE_MODEL}")
    processor = Wav2Vec2Processor.from_pretrained(BASE_MODEL)
    model     = Wav2Vec2ForCTC.from_pretrained(
        BASE_MODEL,
        ctc_loss_reduction = "mean",
        pad_token_id       = processor.tokenizer.pad_token_id,
    )
    # transformers 버전별 메서드명 차이 대응
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    else:
        model.freeze_feature_extractor()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔓 학습 파라미터: {trainable:,}")

    # 데이터셋
    train_ds = ZipStreamDataset(split["train"],      zip_path, processor)
    val_ds   = ZipStreamDataset(split["validation"], zip_path, processor)

    # ⚠️ ZIP 스트리밍은 I/O가 병목 → num_workers=0 (Windows 필수)
    training_args = TrainingArguments(
        output_dir                  = str(output_dir),
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        gradient_accumulation_steps = grad_accum,
        num_train_epochs            = num_epochs,
        learning_rate               = lr,
        weight_decay                = 0.005,
        lr_scheduler_type           = "cosine",
        fp16                        = (device == "cuda"),
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        logging_steps               = 50,
        load_best_model_at_end      = True,
        metric_for_best_model       = "cer",
        greater_is_better           = False,
        save_total_limit            = 2,
        report_to                   = "none",
        dataloader_num_workers      = 0,   # Windows ZIP 스트리밍: 반드시 0
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        processing_class = processor.feature_extractor,
        data_collator   = DataCollatorCTCWithPadding(processor),
        compute_metrics = make_compute_metrics(processor),
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=5)],
    )

    print("\n🚀 파인튜닝 시작! (ZIP 스트리밍 — 압축 해제 없음)")
    trainer.train()

    # 저장
    best_path = output_dir / "best"
    trainer.save_model(str(best_path))
    processor.save_pretrained(str(best_path))
    print(f"\n💾 저장 완료: {best_path}")
    print(f"\n✅ pronunciation_scorer.py 적용:")
    print(f"   python pronunciation_scorer.py --practice --model {best_path}")


# ────────────────────────────────────────────
# 8. 메인
# ────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip_path",
        default=r"C:\Users\User\Voice-Model-Test\구음장애 음성인식 데이터\01.데이터\1.Training\원천데이터\TS02_언어청각장애.zip",
        help="TS02_언어청각장애.zip 경로"
    )
    parser.add_argument(
        "--label_dir",
        default=r"C:\Users\User\Voice-Model-Test\구음장애 음성인식 데이터\01.데이터\1.Training\라벨링데이터_250331_add\TL02_언어청각장애",
        help="TL02 JSON 라벨 폴더"
    )
    parser.add_argument("--output_dir",  default="./finetuned_model")
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--num_epochs",  type=int,   default=30)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--grad_accum",  type=int,   default=4)
    args = parser.parse_args()

    train(
        zip_path   = Path(args.zip_path),
        label_dir  = Path(args.label_dir),
        output_dir = Path(args.output_dir),
        batch_size = args.batch_size,
        num_epochs = args.num_epochs,
        lr         = args.lr,
        grad_accum = args.grad_accum,
    )