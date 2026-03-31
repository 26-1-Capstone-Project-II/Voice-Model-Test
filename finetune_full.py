"""
언어청각장애 전용 파인튜닝 (100% 전체 데이터셋 학습용)
====================================================
리눅스 서버(Ubuntu) 환경에 맞춰 경로와 실행 옵션을 최적화했습니다.

실행 명령어 (tmux 안에서 복사해서 사용):
    CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python finetune_full.py
"""

import io
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
import soundfile as sf
import evaluate
from torch.utils.data import Dataset
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

from korean_g2p_nomecab import load_g2p

# 🚨 [필수 추가] 메모리 효율 및 충돌 방지
torch.backends.cudnn.enabled = False

# ────────────────────────────────────────────
# ⚙️ 경로 설정 (리눅스 서버 절대 경로로 수정)
# ────────────────────────────────────────────

BASE_MODEL = "w11wo/wav2vec2-xls-r-300m-korean"
TARGET_SR  = 16000
MAX_SEC    = 10.0

# 🚨 [수정 포인트 1] 리눅스 서버의 실제 사용자 홈 디렉토리를 사용합니다.
HOME = Path.home() # /home/slim

# 데이터 루트 경로를 리눅스 포맷으로 수정
DATA_ROOT = HOME / "mingly_workspace" / "구음장애 음성인식 데이터" / "01.데이터" / "1.Training"

# 100% 데이터셋 세부 경로 (띄어쓰기 포함된 폴더명 대응)
DEFAULT_WAV_DIR   = DATA_ROOT / "원천데이터" / "TS02_언어청각장애"
DEFAULT_JSON_DIR  = DATA_ROOT / "라벨링데이터_250331_add" / "TL02_언어청각장애"

# 결과 저장 경로 (10% 학습 시 사용했던 lora_adapter와 구분)
DEFAULT_OUTPUT_DIR = HOME / "mingly_workspace" / "Voice-Model-Test" / "best_model_full"


# ────────────────────────────────────────────
# 1. WAV ↔ JSON 매칭
# ────────────────────────────────────────────

def load_pairs(wav_dir: Path, json_dir: Path, g2p) -> dict:
    print(f"\n🔍 [데이터 탐색]")
    print(f"   📂 WAV: {wav_dir}")
    print(f"   📂 JSON: {json_dir}")

    # JSONL 파일(VAD 세그멘테이션 결과)이 있는지 확인
    train_jsonl = json_dir / "train.jsonl"
    if train_jsonl.exists():
        print(f"  📂 JSONL 데이터 감지 → 직접 로드 시작")
        split = {"train": [], "validation": [], "test": []}
        for split_name in ["train", "validation", "test"]:
            path = json_dir / f"{split_name}.jsonl"
            if not path.exists(): continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line.strip())
                    if obj.get("label") and obj.get("wav_path"):
                        split[split_name].append(obj)
            print(f"  ✅ {split_name:12s}: {len(split[split_name]):,}개")
        return split

    # 파일 직접 매칭 (JSONL이 없을 경우)
    wav_index  = {f.stem: f for f in wav_dir.rglob("*.wav")}
    json_files = sorted(json_dir.rglob("*.json"))
    print(f"  📦 파일 검색 결과: WAV {len(wav_index):,}개 / JSON {len(json_files):,}개")

    pairs = []
    for jp in json_files:
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
            transcript = data.get("Transcript", "").strip()
            if not transcript: continue
            wav_stem = Path(data.get("File_id", "")).stem
            wav_path = wav_index.get(wav_stem)
            if wav_path is None: continue
            
            # 발음 기호 변환 (G2P)
            label = g2p(transcript, descriptive=True).strip()
            pairs.append({
                "wav_path": str(wav_path),
                "label": label,
                "speaker_id": wav_stem
            })
        except: continue
    
    print(f"  ✅ 매칭 성공: {len(pairs):,}개")
    return pairs

def split_by_speaker(pairs, seed=42):
    if not pairs: return {"train": [], "validation": [], "test": []}
    random.seed(seed)
    spk = defaultdict(list)
    for p in pairs: spk[p["speaker_id"]].append(p)
    speakers = list(spk.keys())
    random.shuffle(speakers)
    n = len(speakers)
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    train_sp = set(speakers[:n_train])
    val_sp = set(speakers[n_train:n_train+n_val])
    split = {"train": [], "validation": [], "test": []}
    for sp, sp_pairs in spk.items():
        key = "train" if sp in train_sp else ("validation" if sp in val_sp else "test")
        split[key].extend(sp_pairs)
    return split

# ────────────────────────────────────────────
# 2. Dataset & Collator
# ────────────────────────────────────────────

class DysarthriaDataset(Dataset):
    def __init__(self, records, processor):
        self.records = records
        self.processor = processor
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        rec = self.records[idx]
        audio, _ = librosa.load(rec["wav_path"], sr=TARGET_SR, mono=True)
        audio = audio[:int(MAX_SEC * TARGET_SR)]
        input_values = self.processor(audio, sampling_rate=TARGET_SR, return_tensors="pt").input_values[0]
        label_ids = self.processor.tokenizer(rec["label"]).input_ids
        return {"input_values": input_values, "labels": torch.tensor(label_ids, dtype=torch.long)}

@dataclass
class DataCollatorCTC:
    processor: Any
    def __call__(self, features):
        input_features = [{"input_values": f["input_values"]} for f in features]
        batch = self.processor.pad(input_features, padding=True, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, padding=True, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

def make_compute_metrics(processor):
    cer_metric = evaluate.load("cer")
    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": round(cer, 4)}
    return compute_metrics

# ────────────────────────────────────────────
# 3. 학습 메인
# ────────────────────────────────────────────

def train(wav_dir, json_dir, output_dir, batch_size, num_epochs, lr, grad_accum):
    print(f"\n📥 모델 로딩: {BASE_MODEL}")
    processor = Wav2Vec2Processor.from_pretrained(BASE_MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(
        BASE_MODEL, ctc_loss_reduction="mean", pad_token_id=processor.tokenizer.pad_token_id
    )
    model.freeze_feature_encoder()

    g2p = load_g2p()
    result = load_pairs(wav_dir, json_dir, g2p)
    
    if not result or (isinstance(result, dict) and not result["train"]):
        print("❌ 에러: 데이터를 찾지 못했습니다. 경로 설정을 다시 확인하세요.")
        return

    split = result if isinstance(result, dict) else split_by_speaker(result)

    train_ds = DysarthriaDataset(split["train"], processor)
    val_ds = DysarthriaDataset(split["validation"][:500], processor)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,
        learning_rate=lr,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=1000,
        save_strategy="steps",
        save_steps=1000,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=processor.feature_extractor,
        data_collator=DataCollatorCTC(processor),
        compute_metrics=make_compute_metrics(processor),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    print(f"\n🚀 100% 데이터셋 학습 시작! (총 {len(split['train']):,} 샘플)")
    trainer.train()
    
    best_path = Path(output_dir) / "best"
    trainer.save_model(str(best_path))
    processor.save_pretrained(str(best_path))
    print(f"\n💾 최종 모델 저장 완료: {best_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 🚨 [수정 포인트 2] 리눅스용 기본 경로 주입
    parser.add_argument("--wav_dir",    default=str(DEFAULT_WAV_DIR))
    parser.add_argument("--json_dir",   default=str(DEFAULT_JSON_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    
    # ⭐️ 10% 학습 성공 세팅값 유지
    parser.add_argument("--batch_size", type=int,   default=1)
    parser.add_argument("--num_epochs", type=int,   default=30)
    parser.add_argument("--grad_accum", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-4)
    args = parser.parse_args()

    train(Path(args.wav_dir), Path(args.json_dir), Path(args.output_dir), 
          args.batch_size, args.num_epochs, args.lr, args.grad_accum)