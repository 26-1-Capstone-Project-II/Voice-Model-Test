"""
온보이스 (On-Voice) — 로컬 PC용 LoRA 파인튜닝
=============================================
finetune_simple.py의 로컬 데이터 로딩 방식 + LoRA 기법 결합.
메모리(RAM) 폭발을 방지하기 위해 데이터를 동적으로 로드하며,
GTX 1660 Super(6GB) VRAM에 맞춰 최적화되었습니다.
"""

import json
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
from peft import LoraConfig, get_peft_model, TaskType

from korean_g2p_nomecab import load_g2p


# ────────────────────────────────────────────
# 1. 환경 및 경로 설정 (로컬 맞춤형)
# ────────────────────────────────────────────
BASE_MODEL = "w11wo/wav2vec2-xls-r-300m-korean"
TARGET_SR  = 16000
MAX_SEC    = 10.0

DATA_ROOT = Path(r"C:\Users\User\Voice-Model-Test\구음장애 음성인식 데이터\01.데이터\1.Training")
WAV_DIR   = DATA_ROOT / "원천데이터"  / "TS02_언어청각장애"
JSON_DIR  = DATA_ROOT / "라벨링데이터_250331_add" / "TL02_언어청각장애"

LORA_CONFIG = dict(
    r              = 16,
    lora_alpha     = 32,
    lora_dropout   = 0.05,
    target_modules = ["q_proj", "v_proj"],
    bias           = "none",
)


# ────────────────────────────────────────────
# 2. 데이터 매칭 (finetune_simple.py 방식)
# ────────────────────────────────────────────
def load_pairs(wav_dir: Path, json_dir: Path, g2p) -> dict:
    train_jsonl = json_dir / "train.jsonl"
    val_jsonl   = json_dir / "validation.jsonl"

    if train_jsonl.exists() and val_jsonl.exists():
        print(f"  📂 JSONL 세그멘테이션 데이터 감지 → split별 직접 로드")
        split = {"train": [], "validation": [], "test": []}
        for split_name in ["train", "validation", "test"]:
            npy_path = json_dir / f"{split_name}_npy.jsonl"
            path     = npy_path if npy_path.exists() else json_dir / f"{split_name}.jsonl"
            if not path.exists(): continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line.strip())
                    if obj.get("label") and obj.get("wav_path"):
                        split[split_name].append(obj)
            print(f"  ✅ {split_name:12s}: {len(split[split_name]):,}개")
        return split

    wav_index  = {f.stem: f for f in wav_dir.rglob("*.wav")}
    json_files = sorted(json_dir.rglob("*.json"))
    pairs = []
    
    for jp in json_files:
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
            transcript = data.get("Transcript", "").strip()
            wav_stem = Path(data.get("File_id", "")).stem
            wav_path = wav_index.get(wav_stem)
            
            if not transcript or not wav_path: continue
            
            label = g2p(transcript, descriptive=True).strip()
            pairs.append({
                "wav_path"   : str(wav_path),
                "label"      : label,
                "speaker_id" : wav_stem,
            })
        except Exception:
            continue
            
    print(f"  ✅ 매칭: {len(pairs):,}개")
    return pairs

def split_by_speaker(pairs: list[dict], seed: int = 42) -> dict:
    random.seed(seed)
    spk = defaultdict(list)
    for p in pairs: spk[p["speaker_id"]].append(p)

    speakers = list(spk.keys())
    random.shuffle(speakers)
    n = len(speakers)
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    
    split = {"train": [], "validation": [], "test": []}
    for i, sp in enumerate(speakers):
        key = "train" if i < n_train else ("validation" if i < n_train + n_val else "test")
        split[key].extend(spk[sp])
    return split


# ────────────────────────────────────────────
# 3. 데이터셋 및 콜레이터 (동적 로딩 - RAM 최적화)
# ────────────────────────────────────────────
class DysarthriaDataset(Dataset):
    def __init__(self, records: list[dict], processor: Wav2Vec2Processor):
        self.records   = records
        self.processor = processor

    def __len__(self): return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        try:
            if rec.get("npy_path") and Path(rec["npy_path"]).exists():
                audio = np.load(rec["npy_path"])
            else:
                audio, _ = librosa.load(rec["wav_path"], sr=TARGET_SR, mono=True)
                audio = audio[:int(MAX_SEC * TARGET_SR)]
        except Exception:
            audio = np.zeros(int(TARGET_SR), dtype=np.float32)

        input_values = self.processor(
            audio, sampling_rate=TARGET_SR, return_tensors="pt", padding=False
        ).input_values[0]

        label_ids = self.processor.tokenizer(rec["label"]).input_ids
        return {"input_values": input_values, "labels": torch.tensor(label_ids, dtype=torch.long)}

@dataclass
class DataCollatorCTC:
    processor: Any
    def __call__(self, features: list[dict]) -> dict:
        input_features = [{"input_values": f["input_values"]} for f in features]
        batch = self.processor.pad(input_features, padding=True, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(label_features, padding=True, return_tensors="pt")
        
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

def make_compute_metrics(processor: Wav2Vec2Processor):
    cer_metric = evaluate.load("cer")
    def compute_metrics(pred):
        pred_ids  = np.argmax(pred.predictions, axis=-1)
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": round(cer, 4)}
    return compute_metrics


# ────────────────────────────────────────────
# 4. LoRA 파인튜닝 메인
# ────────────────────────────────────────────
def train(args):
    print("\n🔤 G2P 로딩...")
    g2p = load_g2p()

    print(f"\n🔗 데이터 매칭 중...")
    result = load_pairs(Path(args.wav_dir), Path(args.json_dir), g2p)
    split = result if isinstance(result, dict) else split_by_speaker(result)

    print(f"\n📥 베이스 모델 로딩: {BASE_MODEL}")
    processor = Wav2Vec2Processor.from_pretrained(BASE_MODEL)
    model     = Wav2Vec2ForCTC.from_pretrained(
        BASE_MODEL,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    # 1. Feature Extractor 동결 (메모리 절약)
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    else:
        model.freeze_feature_extractor()

    # 2. LoRA 적용
    print("\n[⚡] LoRA 어댑터 장착 중...")
    lora_config = LoraConfig(task_type=TaskType.TOKEN_CLS, **LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    
    model.print_trainable_parameters()

    # 데이터셋 구성 (OOM 방지를 위해 검증셋 500개 제한)
    val_records = split["validation"]
    if len(val_records) > 500:
        random.seed(42)
        val_records = random.sample(val_records, 500)

    train_ds = DysarthriaDataset(split["train"], processor)
    val_ds   = DysarthriaDataset(val_records, processor)

    # 파워셸 환경에서 출력 디렉토리 경로 정리
    out_dir = Path(args.output_dir)
    
    training_args = TrainingArguments(
        output_dir                  = str(out_dir),
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        num_train_epochs            = args.epochs,
        learning_rate               = args.lr,
        fp16                        = False,  # wav2vec2에서는 NaN 위험이 있어 False 유지
        gradient_checkpointing      = True,   # VRAM 대폭 절약
        eval_strategy               = "steps",
        eval_steps                  = 500,
        save_strategy               = "steps",
        save_steps                  = 500,
        logging_steps               = 50,
        load_best_model_at_end      = True,
        metric_for_best_model       = "cer",
        greater_is_better           = False,
        save_total_limit            = 2,
        report_to                   = "none",
        eval_accumulation_steps     = 10,
    )

    trainer = Trainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_ds,
        eval_dataset     = val_ds,
        processing_class = processor.feature_extractor,
        data_collator    = DataCollatorCTC(processor),
        compute_metrics  = make_compute_metrics(processor),
        callbacks        = [EarlyStoppingCallback(early_stopping_patience=5)],
    )

    print("\n🚀 LoRA 파인튜닝 시작!")
    trainer.train()

    # 어댑터 저장 (원본 모델은 저장되지 않고 작고 가벼운 가중치 파일만 저장됨)
    adapter_path = out_dir / "best_lora_adapter"
    model.save_pretrained(str(adapter_path))
    processor.save_pretrained(str(adapter_path))
    print(f"\n💾 LoRA 어댑터 저장 완료: {adapter_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir",    default=str(WAV_DIR))
    parser.add_argument("--json_dir",   default=str(JSON_DIR))
    parser.add_argument("--output_dir", default="./finetuned_model_lora")
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--grad_accum", type=int,   default=8)
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--lr",         type=float, default=1e-4)
    args = parser.parse_args()

    train(args)