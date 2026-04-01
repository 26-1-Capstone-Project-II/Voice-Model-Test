"""
finetune_jamo.py — 자모 단위 Vocab 기반 CTC 파인튜닝
=====================================================
핵심 변경: 음절 vocab (1,207개) → 자모 vocab (57개)

왜 자모 vocab인가?
  1. OOV 완전 해소 — 어떤 발음이든 자모로 분해 가능
  2. 발음 오류 정밀 감지 — 자모 레벨에서 어디가 틀렸는지 식별
  3. CTC 학습 안정화 — vocab 축소로 classification head 부담 감소

파이프라인:
  원본 텍스트 "같이 해볼까"
      ↓ G2P (descriptive=True)
  발음 전사 "가치 해볼까"
      ↓ syllable_to_jamo()
  자모 라벨 "ㄱㅏㅊㅣ|ㅎㅐㅂㅗㄹㄲㅏ"
      ↓ Jamo Tokenizer → ID 시퀀스
  CTC 학습 라벨 [5, 25, 20, 45, 4, 24, 26, ...]

실행:
    CUDA_VISIBLE_DEVICES=0 python finetune_jamo.py --lr 3e-5
    CUDA_VISIBLE_DEVICES=0 python finetune_jamo.py --lr 3e-5 --dry_run  # 데이터 검증만
"""

import json
import argparse
import random
import os
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
import numpy as np
import librosa
import evaluate
from torch.utils.data import Dataset
from transformers import Wav2Vec2ForCTC

from korean_g2p_nomecab import load_g2p
from jamo_utils import (
    syllable_to_jamo,
    jamo_to_syllable,
    build_jamo_processor,
    build_jamo_vocab,
)

# ────────────────────────────────────────────
# ⚙️ 설정
# ────────────────────────────────────────────

BASE_MODEL = "w11wo/wav2vec2-xls-r-300m-korean"
TARGET_SR  = 16000
MAX_SEC    = 10.0

HOME = Path.home()

# 데이터 경로 (서버 기준)
DATA_ROOT = HOME / "mingly_workspace" / "dataset" / "013.구음장애_음성인식_데이터" / "01.데이터" / "1.Training"
DEFAULT_WAV_DIR  = DATA_ROOT / "원천데이터"
DEFAULT_JSON_DIR = DATA_ROOT / "라벨링데이터_250331_add"

# 출력 경로
DEFAULT_OUTPUT_DIR = HOME / "mingly_workspace" / "Voice-Model-Test" / "best_model_jamo"

# 자모 tokenizer 저장 경로
JAMO_TOKENIZER_DIR = HOME / "mingly_workspace" / "Voice-Model-Test" / "jamo_tokenizer"


# ────────────────────────────────────────────
# 1. 데이터 로드 (finetune_full.py에서 재사용)
# ────────────────────────────────────────────

def verify_paths(wav_dir: Path, json_dir: Path):
    """경로 존재 여부 확인."""
    success = True
    for d, label in [(wav_dir, "WAV"), (json_dir, "JSON")]:
        if not d.exists():
            print(f"\n❌ [경로 에러] {label} 폴더가 없습니다: {d}")
            parent = d.parent
            if parent.exists():
                print(f"   💡 '{parent}' 안에 있는 폴더들:")
                for p in parent.iterdir():
                    print(f"      - {p.name}")
            success = False
    return success


def load_pairs(wav_dir: Path, json_dir: Path, g2p) -> list:
    """WAV-JSON 매칭 후 G2P → 자모 변환된 라벨 생성."""
    print(f"\n🔍 [데이터 탐색 시작]")
    print(f"   📂 WAV 경로: {wav_dir}")
    print(f"   📂 JSON 경로: {json_dir}")

    if not verify_paths(wav_dir, json_dir):
        return None

    # JSONL 파일 확인 (세그멘테이션 데이터)
    train_jsonl = json_dir / "train.jsonl"
    if train_jsonl.exists():
        print(f"  📂 JSONL 데이터 감지 → 직접 로드")
        all_data = []
        for split_name in ["train", "validation", "test"]:
            path = json_dir / f"{split_name}.jsonl"
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line.strip())
                        text = obj.get("label", "").strip()
                        if text and obj.get("wav_path"):
                            # G2P → 자모 변환
                            phonetic = g2p(text, descriptive=True).strip()
                            jamo_label = syllable_to_jamo(phonetic)
                            if len(jamo_label) >= 2:
                                obj["label"] = jamo_label
                                obj["original_text"] = text
                                obj["phonetic"] = phonetic
                                all_data.append(obj)
                    except Exception:
                        continue
        return all_data

    # 개별 JSON 파일 매칭
    print("  🔎 파일 검색 중 (하위 폴더 포함)...")
    wav_index = {f.stem: f for f in wav_dir.rglob("*.wav")}
    json_files = sorted(json_dir.rglob("*.json"))
    print(f"  📦 검색 결과: WAV {len(wav_index):,}개 / JSON {len(json_files):,}개")

    if len(wav_index) == 0 or len(json_files) == 0:
        return None

    pairs = []
    for jp in json_files:
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
            transcript = data.get("Transcript", "").strip()
            if not transcript:
                continue
            wav_stem = Path(data.get("File_id", "")).stem
            wav_path = wav_index.get(wav_stem)
            if wav_path is None:
                continue

            # G2P → 자모 변환
            phonetic = g2p(transcript, descriptive=True).strip()
            jamo_label = syllable_to_jamo(phonetic)

            if len(jamo_label) < 2:
                continue

            pairs.append({
                "wav_path": str(wav_path),
                "label": jamo_label,
                "original_text": transcript,
                "phonetic": phonetic,
                "speaker_id": wav_stem,
            })
        except Exception:
            continue

    # 빈/짧은 라벨 필터링
    before = len(pairs)
    pairs = [p for p in pairs if len(p["label"].strip()) >= 2]
    if before != len(pairs):
        print(f"  🧹 필터링: {before - len(pairs)}개 제거")

    print(f"  ✅ 매칭 성공: {len(pairs):,}개")

    # 샘플 출력
    if pairs:
        print(f"\n  📋 라벨 변환 샘플:")
        for p in pairs[:3]:
            print(f"     원문: {p['original_text']}")
            print(f"     발음: {p['phonetic']}")
            print(f"     자모: {p['label']}")
            print()

    return pairs


def split_by_speaker(pairs, seed=42):
    """화자 기반 train/val/test 분할."""
    if not pairs:
        return {"train": [], "validation": [], "test": []}
    random.seed(seed)
    spk = defaultdict(list)
    for p in pairs:
        spk[p.get("speaker_id", "default")].append(p)
    speakers = list(spk.keys())
    random.shuffle(speakers)
    n = len(speakers)
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    train_sp = set(speakers[:n_train])
    val_sp = set(speakers[n_train:n_train + n_val])
    split = {"train": [], "validation": [], "test": []}
    for sp, sp_pairs in spk.items():
        key = "train" if sp in train_sp else ("validation" if sp in val_sp else "test")
        split[key].extend(sp_pairs)
    return split


# ────────────────────────────────────────────
# 2. Dataset 클래스
# ────────────────────────────────────────────

class JamoDataset(Dataset):
    """자모 라벨 기반 CTC 학습 데이터셋."""

    def __init__(self, records, processor):
        self.records = records
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            audio, _ = librosa.load(rec["wav_path"], sr=TARGET_SR, mono=True)
            audio = audio[:int(MAX_SEC * TARGET_SR)]

            input_values = self.processor(
                audio, sampling_rate=TARGET_SR, return_tensors="pt", padding=False
            ).input_values[0]

            # 자모 라벨 → 토큰 ID
            # CTC 제약: 라벨 길이 < 인코더 출력 길이
            # wav2vec2: 320x 다운샘플 → 초당 50프레임
            max_label_len = int((len(audio) / TARGET_SR) * 50)  # 여유롭게 50자/초
            label_text = rec["label"][:max_label_len]
            label_ids = self.processor.tokenizer(label_text).input_ids

            return {
                "input_values": input_values,
                "labels": torch.tensor(label_ids, dtype=torch.long),
            }
        except Exception as e:
            # 폴백: 0.5초 무음 + 빈 라벨
            dummy = np.zeros(int(TARGET_SR * 0.5), dtype=np.float32)
            input_values = self.processor(
                dummy, sampling_rate=TARGET_SR, return_tensors="pt", padding=False
            ).input_values[0]
            return {
                "input_values": input_values,
                "labels": torch.tensor([0]),
            }


# ────────────────────────────────────────────
# 3. Data Collator
# ────────────────────────────────────────────

@dataclass
class DataCollatorCTC:
    processor: Any

    def __call__(self, features):
        # 오디오 패딩
        input_features = [{"input_values": f["input_values"]} for f in features]
        batch = self.processor.pad(input_features, padding=True, return_tensors="pt")

        # 라벨 패딩 (-100 = CTC ignore)
        label_list = [f["labels"] for f in features]
        max_len = max(len(l) for l in label_list)
        labels_padded = []
        for lbl in label_list:
            pad_len = max_len - len(lbl)
            padded = torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)])
            labels_padded.append(padded)
        batch["labels"] = torch.stack(labels_padded)

        return batch


# ────────────────────────────────────────────
# 4. 평가 메트릭 (자모→음절 변환 후 CER)
# ────────────────────────────────────────────

def make_compute_metrics(processor):
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = np.argmax(pred.predictions, axis=-1)
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # 자모 문자열 디코딩
        pred_jamo = processor.batch_decode(pred_ids)
        label_jamo = processor.batch_decode(label_ids, group_tokens=False)

        # 자모 → 음절 변환 (사람이 읽기 쉬운 형태로)
        pred_syllable = [jamo_to_syllable(j) for j in pred_jamo]
        label_syllable = [jamo_to_syllable(j) for j in label_jamo]

        # CER 계산 (음절 기준)
        cer = cer_metric.compute(predictions=pred_syllable, references=label_syllable)

        # 샘플 출력 (첫 2개)
        for i in range(min(2, len(pred_syllable))):
            print(f"\n  📋 샘플 [{i}]")
            print(f"     정답(음절): {label_syllable[i]}")
            print(f"     예측(음절): {pred_syllable[i]}")
            print(f"     정답(자모): {label_jamo[i][:40]}...")
            print(f"     예측(자모): {pred_jamo[i][:40]}...")

        return {"cer": round(cer, 4)}

    return compute_metrics


# ────────────────────────────────────────────
# 5. 모델 구성 (CTC head 재초기화)
# ────────────────────────────────────────────

def build_model(processor):
    """
    wav2vec2 base 모델에 자모 vocab 크기 CTC head 장착.

    핵심: model.lm_head를 새 vocab_size로 교체.
    기존 lm_head (1,207개 출력)은 음절 vocab용이므로 버림.
    자모 vocab (57개 출력)으로 새로 초기화.
    """
    print(f"\n📥 모델 로드: {BASE_MODEL}")

    vocab_size = processor.tokenizer.vocab_size
    pad_token_id = processor.tokenizer.pad_token_id

    # 1. base 모델 로드 (기존 lm_head 무시)
    model = Wav2Vec2ForCTC.from_pretrained(
        BASE_MODEL,
        ctc_loss_reduction="mean",
        pad_token_id=pad_token_id,
        ignore_mismatched_sizes=True,  # lm_head 크기 차이 무시
    )

    # 2. CTC head 교체 → 자모 vocab 크기
    hidden_size = model.config.hidden_size  # 1024 (XLS-R 300M)
    model.lm_head = nn.Linear(hidden_size, vocab_size)
    model.config.vocab_size = vocab_size

    # 3. lm_head 초기화 (Xavier)
    nn.init.xavier_uniform_(model.lm_head.weight)
    nn.init.zeros_(model.lm_head.bias)

    # 4. Feature encoder 동결 (CNN 특징 추출기)
    model.freeze_feature_encoder()

    # 파라미터 통계
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✅ CTC head 교체: {hidden_size} → {vocab_size} (자모 vocab)")
    print(f"  📊 파라미터: {trainable:,} / {total:,} 학습 가능")

    return model


# ────────────────────────────────────────────
# 6. 학습 실행
# ────────────────────────────────────────────

def train(wav_dir, json_dir, output_dir, batch_size, num_epochs, lr, grad_accum, dry_run=False):
    """자모 vocab 기반 CTC 파인튜닝 메인 함수."""
    # Trainer 관련 import는 여기서 — PEFT 버전 충돌 방지 (모듈 레벨 import 제거)
    from transformers import TrainingArguments, Trainer, EarlyStoppingCallback

    # 1. 자모 Processor 생성
    print(f"\n🔧 자모 Tokenizer/Processor 생성 중...")
    processor = build_jamo_processor(str(JAMO_TOKENIZER_DIR), BASE_MODEL)

    # 2. 데이터 로드
    g2p = load_g2p()
    result = load_pairs(wav_dir, json_dir, g2p)

    if result is None or len(result) == 0:
        print("\n❌ [학습 중단] 데이터 로딩 실패.")
        return

    split = split_by_speaker(result)

    if dry_run:
        print(f"\n🧪 [DRY RUN] 데이터 검증 완료 — 학습 없이 종료")
        print(f"   Train: {len(split['train']):,}개")
        print(f"   Val:   {len(split['validation']):,}개")
        print(f"   Test:  {len(split['test']):,}개")

        # 토큰화 테스트
        print(f"\n   📋 토큰화 테스트:")
        for p in split["train"][:3]:
            ids = processor.tokenizer(p["label"]).input_ids
            print(f"      자모: {p['label'][:30]}... → IDs: {ids[:10]}...")
        return

    # 3. Dataset 생성
    train_ds = JamoDataset(split["train"], processor)
    val_ds = JamoDataset(split["validation"][:500], processor)

    # 4. 모델 생성
    model = build_model(processor)

    # 5. 학습 설정
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,
        learning_rate=lr,
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,              # 기울기 폭발 방지
        fp16=False,                     # CTC Loss NaN 방지
        gradient_checkpointing=True,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        save_total_limit=3,
        report_to="none",
        dataloader_num_workers=0,
        eval_accumulation_steps=10,
    )

    # 6. Trainer
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

    # 7. 학습 시작
    print(f"\n🚀 자모 Vocab 파인튜닝 시작!")
    print(f"   Vocab: {processor.tokenizer.vocab_size}개 (자모 단위)")
    print(f"   Train: {len(split['train']):,}개 / Val: {len(split['validation'][:500]):,}개")
    print(f"   LR: {lr} / Epochs: {num_epochs} / Batch: {batch_size}×{grad_accum}")

    trainer.train()

    # 8. 저장
    best_path = Path(output_dir) / "best"
    trainer.save_model(str(best_path))
    processor.save_pretrained(str(best_path))
    print(f"\n💾 최종 저장 완료: {best_path}")


# ────────────────────────────────────────────
# 7. CLI
# ────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="자모 단위 Vocab 기반 CTC 파인튜닝 (wav2vec2-xls-r-300m-korean)"
    )
    parser.add_argument("--wav_dir",    default=str(DEFAULT_WAV_DIR))
    parser.add_argument("--json_dir",   default=str(DEFAULT_JSON_DIR))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch_size", type=int,   default=1)
    parser.add_argument("--num_epochs", type=int,   default=30)
    parser.add_argument("--grad_accum", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=3e-5)
    parser.add_argument("--dry_run",    action="store_true",
                        help="데이터 로드 + tokenizer 검증만 (학습 없이)")
    args = parser.parse_args()

    train(
        Path(args.wav_dir), Path(args.json_dir), Path(args.output_dir),
        args.batch_size, args.num_epochs, args.lr, args.grad_accum,
        dry_run=args.dry_run,
    )
