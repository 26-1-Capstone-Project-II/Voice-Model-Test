"""
언어청각장애 전용 파인튜닝 (100% 전체 데이터셋 학습용)
====================================================
서버의 실제 하위 폴더 구조(21.조음, 24.발성 등)를 WAV와 JSON 모두에서 탐색하도록 수정되었습니다.

실행 명령어:
    CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python finetune_full.py --lr 5e-5
"""

import io
import json
import argparse
import random
import os
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
# ⚙️ 경로 설정 (서버 실제 목록 반영)
# ────────────────────────────────────────────

BASE_MODEL = "w11wo/wav2vec2-xls-r-300m-korean"
TARGET_SR  = 16000
MAX_SEC    = 10.0

HOME = Path.home() # /home/slim

# 데이터 루트 경로
DATA_ROOT = HOME / "mingly_workspace" / "dataset" / "013.구음장애_음성인식_데이터" / "01.데이터" / "1.Training"

# 🚨 [수정 완료] WAV와 JSON 폴더 모두 하위 카테고리(21.조음 등)의 상위 폴더까지만 지정합니다.
DEFAULT_WAV_DIR   = DATA_ROOT / "원천데이터"
DEFAULT_JSON_DIR  = DATA_ROOT / "라벨링데이터_250331_add"

# 결과 저장 경로
DEFAULT_OUTPUT_DIR = HOME / "mingly_workspace" / "Voice-Model-Test" / "best_model_full"


# ────────────────────────────────────────────
# 1. WAV ↔ JSON 매칭 로직
# ────────────────────────────────────────────

def verify_paths(wav_dir: Path, json_dir: Path):
    """경로가 존재하는지 확인하고, 없으면 상위 폴더의 실제 목록을 보여줍니다."""
    success = True
    for d, label in [(wav_dir, "WAV"), (json_dir, "JSON")]:
        if not d.exists():
            print(f"\n❌ [경로 에러] {label} 폴더가 없습니다: {d}")
            parent = d.parent
            if parent.exists():
                print(f"   💡 '{parent}' 폴더 안에는 아래 파일들이 있습니다:")
                for p in parent.iterdir():
                    print(f"      - {p.name}")
            else:
                print(f"   💡 상위 폴더 '{parent}' 도 존재하지 않습니다. 경로를 다시 확인하세요.")
            success = False
    return success

def load_pairs(wav_dir: Path, json_dir: Path, g2p) -> list:
    print(f"\n🔍 [데이터 탐색 시작]")
    print(f"   📂 WAV 경로: {wav_dir}")
    print(f"   📂 JSON 경로: {json_dir}")

    if not verify_paths(wav_dir, json_dir):
        return None

    # 1. JSONL 파일 확인 (세그멘테이션 데이터가 있을 경우)
    train_jsonl = json_dir / "train.jsonl"
    if train_jsonl.exists():
        print(f"  📂 JSONL 데이터 감지 → 직접 로드 시작")
        all_data = []
        for split_name in ["train", "validation", "test"]:
            path = json_dir / f"{split_name}.jsonl"
            if not path.exists(): continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line.strip())
                        if obj.get("label") and obj.get("wav_path"):
                            all_data.append(obj)
                    except: continue
        return all_data

    # 2. 개별 JSON 파일 매칭 (rglob) - 하위 폴더까지 검색
    print("  🔎 파일을 직접 검색 중입니다 (하위 폴더 포함)...")
    wav_index  = {f.stem: f for f in wav_dir.rglob("*.wav")}
    json_files = sorted(json_dir.rglob("*.json"))
    print(f"  📦 검색 결과: WAV {len(wav_index):,}개 / JSON {len(json_files):,}개")

    if len(wav_index) == 0 or len(json_files) == 0:
        return None

    pairs = []
    for jp in json_files:
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
            transcript = data.get("Transcript", "").strip()
            if not transcript: continue
            wav_stem = Path(data.get("File_id", "")).stem
            wav_path = wav_index.get(wav_stem)
            if wav_path is None: continue
            
            label = g2p(transcript, descriptive=True).strip()
            pairs.append({
                "wav_path": str(wav_path),
                "label": label,
                "speaker_id": wav_stem
            })
        except:
            continue
    
    # 🚨 [추가] 데이터 필터링: 빈 라벨 및 너무 짧은 라벨 제거
    before_filter = len(pairs)
    pairs = [p for p in pairs if len(p["label"].strip()) >= 2]
    if before_filter != len(pairs):
        print(f"  🧹 필터링: {before_filter - len(pairs)}개 제거 (빈/짧은 라벨)")
    
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

class DysarthriaDataset(Dataset):
    def __init__(self, records, processor):
        self.records = records
        self.processor = processor
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            audio, _ = librosa.load(rec["wav_path"], sr=TARGET_SR, mono=True)
            audio = audio[:int(MAX_SEC * TARGET_SR)]
            input_values = self.processor(
                audio, sampling_rate=TARGET_SR, return_tensors="pt", padding=False
            ).input_values[0]

            # 🚨 [핵심 수정] CTC Loss 안정성: 라벨 길이를 오디오 대비 제한 (초당 20자)
            # CTC는 라벨 길이 > 인코더 출력 길이이면 loss가 폭발합니다.
            max_label_len = int((len(audio) / TARGET_SR) * 20)
            label_text = rec["label"][:max_label_len] if len(rec["label"]) > max_label_len else rec["label"]
            label_ids = self.processor.tokenizer(label_text).input_ids

            return {"input_values": input_values, "labels": torch.tensor(label_ids, dtype=torch.long)}
        except Exception as e:
            # 🚨 [핵심 수정] 폴백을 processor 경유 1D 텐서로 생성 (2D 텐서 차원 불일치 방지)
            dummy_audio = np.zeros(int(TARGET_SR * 0.5), dtype=np.float32)  # 0.5초 무음
            input_values = self.processor(
                dummy_audio, sampling_rate=TARGET_SR, return_tensors="pt", padding=False
            ).input_values[0]
            return {"input_values": input_values, "labels": torch.tensor([0])}

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

def train(wav_dir, json_dir, output_dir, batch_size, num_epochs, lr, grad_accum):
    print(f"\n📥 모델 로드: {BASE_MODEL}")
    processor = Wav2Vec2Processor.from_pretrained(BASE_MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(
        BASE_MODEL, ctc_loss_reduction="mean", pad_token_id=processor.tokenizer.pad_token_id
    )
    model.freeze_feature_encoder()

    g2p = load_g2p()
    result = load_pairs(wav_dir, json_dir, g2p)
    
    if result is None or len(result) == 0:
        print("\n❌ [학습 중단] 데이터 로딩 실패. 위 로그의 경로를 확인하세요.")
        return

    split = split_by_speaker(result)
    train_ds = DysarthriaDataset(split["train"], processor)
    val_ds = DysarthriaDataset(split["validation"][:500], processor)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,
        learning_rate=lr,
        warmup_ratio=0.1,           # 🚨 [핵심 추가] CTC 초기 안정화: 전체 step의 10% warmup
        weight_decay=0.005,         # 🚨 [핵심 추가] 가중치 발산 방지
        fp16=False,                 # CTC Loss NaN 방지
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,   # 멀티프로세싱 충돌 방지
        eval_accumulation_steps=10, # 평가 시 OOM 방지
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

    print(f"\n🚀 100% 데이터 학습 시작! (매칭된 데이터: {len(result):,}개)")
    print(f"   Train: {len(split['train']):,}개 / Val: {len(split['validation'][:500]):,}개")
    trainer.train()
    
    best_path = Path(output_dir) / "best"
    trainer.save_model(str(best_path))
    processor.save_pretrained(str(best_path))
    print(f"\n💾 최종 저장 완료: {best_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir",    default=str(DEFAULT_WAV_DIR))
    parser.add_argument("--json_dir",   default=str(DEFAULT_JSON_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch_size", type=int,   default=1)   
    parser.add_argument("--num_epochs", type=int,   default=30)
    parser.add_argument("--grad_accum", type=int,   default=16)  
    parser.add_argument("--lr",         type=float, default=1e-4)
    args = parser.parse_args()

    train(Path(args.wav_dir), Path(args.json_dir), Path(args.output_dir), 
          args.batch_size, args.num_epochs, args.lr, args.grad_accum)