"""
Whisper Tiny 발음 전사 파인튜닝
================================
목적: 소리나는 대로 텍스트 출력 (맞춤법 보정 없이 발음 전사)
입력: 음성 WAV
출력: 발음 전사 텍스트 (G2P 변환 결과)
예) 사용자가 "같이 먹을까?" 를 읽음 → [음성] → "가치 머글까?"

모델: openai/whisper-tiny (39M params, ~150MB)
학습: Seq2Seq + Attention (CTC 대비 유연한 정렬)
배포: CoreML/WhisperKit → iOS 온디바이스

실행:
    # dry_run (데이터 검증)
    CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python finetune_whisper.py --dry_run

    # 본 학습
    CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python finetune_whisper.py \\
        --lr 2e-5 --num_epochs 5 --batch_size 8 --grad_accum 2

    # 샘플 수 제한 (빠른 실험)
    CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python finetune_whisper.py \\
        --max_samples 10000 --lr 2e-5 --num_epochs 3
"""

import json
import argparse
import re
import os
import sys
import torch
import librosa
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Union

# ────────────────────────────────────────────
# cuDNN 비활성화 (서버 CUDA/cuDNN 버전 불일치)
# ────────────────────────────────────────────
torch.backends.cudnn.enabled = False


# ────────────────────────────────────────────
# 1. 설정
# ────────────────────────────────────────────
BASE_MODEL   = "openai/whisper-tiny"
TARGET_SR    = 16000
MAX_SEC      = 25.0    # 구음장애 환자 발화 속도 반영
MIN_SEC      = 0.5     # 노이즈 세그멘트 제외
MAX_LABEL_LEN = 256    # 최대 토큰 수 (디코더 출력)

HOME = Path.home()
SEGMENT_DIR = HOME / "mingly_workspace" / "Voice-Model-Test" / "segmented_dataset"
DEFAULT_OUTPUT_DIR = HOME / "mingly_workspace" / "Voice-Model-Test" / "best_model_whisper"


# ────────────────────────────────────────────
# 2. G2P 로딩
# ────────────────────────────────────────────
def load_g2p():
    """G2P 엔진 로드 (g2pk → 폴백 순서)."""
    print("🔤 G2P 로딩...")
    from korean_g2p_nomecab import load_g2p as _load
    return _load()


# ────────────────────────────────────────────
# 3. 데이터 로딩
# ────────────────────────────────────────────
def load_data(json_dir, max_samples=0, apply_g2p=False):
    """
    segmented_dataset의 JSONL에서 Whisper 학습 데이터 로드.

    JSONL 필드:
      - wav_path: 세그멘트 WAV 경로
      - transcript: 원문 (맞춤법)
      - label: G2P 출력 (발음 전사) ← 학습 타겟
      - duration: 세그멘트 길이 (초)

    apply_g2p=True 시 transcript에서 G2P 재적용 (label 무시).
    """
    json_dir = Path(json_dir)
    g2p = None
    if apply_g2p:
        g2p = load_g2p()

    splits = {}
    total_skipped_short = 0
    total_skipped_long = 0
    total_skipped_nolabel = 0

    for split_name in ["train", "validation", "test"]:
        path = json_dir / f"{split_name}.jsonl"
        if not path.exists():
            print(f"  ⚠️ {split_name}.jsonl 없음 — 건너뜀")
            continue

        records = []
        skipped_short, skipped_long, skipped_nolabel = 0, 0, 0

        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line.strip())
                except Exception:
                    continue

                duration = obj.get("duration", 0)
                if duration < MIN_SEC:
                    skipped_short += 1
                    continue
                if duration > MAX_SEC:
                    skipped_long += 1
                    continue

                wav_path = obj.get("wav_path", "")
                if not wav_path:
                    continue

                # 라벨 결정
                if apply_g2p and g2p:
                    text = obj.get("transcript", "").strip()
                    if not text:
                        skipped_nolabel += 1
                        continue
                    label = g2p(text, descriptive=True).strip()
                else:
                    label = obj.get("label", "").strip()

                if not label or len(label) < 2:
                    skipped_nolabel += 1
                    continue

                # 노이즈 마커 정리
                label = re.sub(r'\s+', ' ', label).strip()

                records.append({
                    "wav_path": wav_path,
                    "label": label,           # 발음 전사 (학습 타겟)
                    "transcript": obj.get("transcript", ""),  # 원문 (참고용)
                    "duration": duration,
                })

        splits[split_name] = records
        total_skipped_short += skipped_short
        total_skipped_long += skipped_long
        total_skipped_nolabel += skipped_nolabel

        print(f"  📂 {split_name}: {len(records):,}개 로드")

    # max_samples 제한
    if max_samples > 0 and "train" in splits:
        original = len(splits["train"])
        splits["train"] = splits["train"][:max_samples]
        if original > max_samples:
            print(f"  📉 train 제한: {original:,} → {max_samples:,}개")

    # 통계
    total = sum(len(v) for v in splits.values())
    print(f"\n  ✅ 총 로드: {total:,}개")
    if total_skipped_short:
        print(f"  🧹 제외 (너무 짧음, <{MIN_SEC}s): {total_skipped_short:,}개")
    if total_skipped_long:
        print(f"  🧹 제외 (너무 김, >{MAX_SEC}s): {total_skipped_long:,}개")
    if total_skipped_nolabel:
        print(f"  🧹 제외 (라벨 없음): {total_skipped_nolabel:,}개")

    return splits


# ────────────────────────────────────────────
# 4. Dataset
# ────────────────────────────────────────────
class WhisperPhoneticDataset(torch.utils.data.Dataset):
    """Whisper 입력용 Dataset: 오디오 → mel spectrogram, 라벨 → token IDs."""

    def __init__(self, records, processor):
        self.records = records
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        # 오디오 로드
        try:
            audio, _ = librosa.load(rec["wav_path"], sr=TARGET_SR, mono=True)
        except Exception:
            # 로드 실패 시 1초 무음 반환
            audio = np.zeros(TARGET_SR, dtype=np.float32)

        # 30초 이하로 자르기 (Whisper 윈도우)
        max_samples = 30 * TARGET_SR
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # Log-mel spectrogram (Whisper feature extractor)
        input_features = self.processor.feature_extractor(
            audio, sampling_rate=TARGET_SR, return_tensors="np"
        ).input_features[0]

        # 라벨 토큰화 (발음 전사 텍스트)
        labels = self.processor.tokenizer(rec["label"]).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
        }


# ────────────────────────────────────────────
# 5. Data Collator
# ────────────────────────────────────────────
@dataclass
class DataCollatorSpeechSeq2Seq:
    """
    Whisper Seq2Seq 학습용 Data Collator.
    - input_features: 패딩 후 텐서 변환
    - labels: 패딩 → -100으로 마스킹 (loss 무시)
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Input features 패딩
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Labels 패딩
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # 패딩 토큰 → -100 (loss 계산에서 제외)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # BOS 토큰 중복 제거
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ────────────────────────────────────────────
# 6. Metrics
# ────────────────────────────────────────────
def make_compute_metrics(processor):
    """CER 기반 평가 함수 생성."""
    import evaluate
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # -100 → pad_token_id (디코딩 가능하도록)
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # 디코딩
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        # CER 계산
        cer = cer_metric.compute(predictions=pred_str, references=label_str)

        # 샘플 출력
        for i in range(min(3, len(pred_str))):
            print(f"\n  📋 샘플 [{i}]")
            print(f"     정답(발음): {label_str[i][:60]}...")
            print(f"     예측(발음): {pred_str[i][:60]}...")

        return {"cer": cer}

    return compute_metrics


# ────────────────────────────────────────────
# 7. Train
# ────────────────────────────────────────────
def train(
    json_dir,
    output_dir,
    lr=2e-5,
    num_epochs=5,
    batch_size=8,
    grad_accum=2,
    max_samples=0,
    apply_g2p=False,
    dry_run=False,
):
    # Lazy imports (PEFT 버전 충돌 방지)
    from transformers import (
        WhisperProcessor,
        WhisperForConditionalGeneration,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
        EarlyStoppingCallback,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Processor 로드 ──
    print(f"\n📥 Processor 로드: {BASE_MODEL}")
    processor = WhisperProcessor.from_pretrained(
        BASE_MODEL, language="ko", task="transcribe"
    )

    # ── 2. 데이터 로드 ──
    print(f"\n🔍 [데이터 탐색 시작]")
    print(f"   📂 JSONL 경로: {json_dir}")
    splits = load_data(json_dir, max_samples, apply_g2p)

    if not splits.get("train"):
        print("❌ train 데이터가 없습니다!")
        return

    # ── Dry Run 모드 ──
    if dry_run:
        print(f"\n🧪 [DRY RUN] 데이터 검증 완료 — 학습 없이 종료")
        print(f"   Train: {len(splits['train']):,}개")
        print(f"   Val:   {len(splits.get('validation', [])):,}개")
        print(f"   Test:  {len(splits.get('test', [])):,}개")

        print(f"\n   📋 라벨 샘플:")
        for p in splits["train"][:5]:
            print(f"      원문: {p.get('transcript', '')[:40]}")
            print(f"      발음: {p['label'][:40]}")
            tokens = processor.tokenizer(p["label"]).input_ids
            print(f"      토큰({len(tokens)}): {tokens[:10]}...")
            print()
        return

    # ── 3. Dataset 생성 ──
    train_ds = WhisperPhoneticDataset(splits["train"], processor)
    val_records = splits.get("validation", splits["train"][:500])[:500]
    val_ds = WhisperPhoneticDataset(val_records, processor)

    # ── 4. 모델 로드 ──
    print(f"\n📥 모델 로드: {BASE_MODEL}")
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)

    # 강제 디코더 토큰 설정 (한국어, 전사 태스크)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="ko", task="transcribe"
    )
    model.config.suppress_tokens = []
    model.generation_config.language = "ko"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="ko", task="transcribe"
    )
    # 반복 생성 방지 ("시퍼서 시퍼서 시퍼서..." 문제 해결)
    model.generation_config.no_repeat_ngram_size = 3
    model.generation_config.repetition_penalty = 1.2

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  📊 파라미터: {trainable_params:,} / {total_params:,} 학습 가능")

    # ── 5. 학습 설정 ──
    effective_batch = batch_size * grad_accum
    steps_per_epoch = len(splits["train"]) // effective_batch
    total_steps = steps_per_epoch * num_epochs
    eval_steps = max(500, steps_per_epoch // 2)  # 에폭당 최소 2번 평가

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,
        learning_rate=lr,
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=True,                          # Whisper는 fp16 안전
        predict_with_generate=True,         # 평가 시 generate() 사용
        generation_max_length=MAX_LABEL_LEN,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",  # CER은 반복 생성 시 > 1.0 왜곡 → loss 기반
        greater_is_better=False,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=2,
    )

    # ── 6. Trainer ──
    data_collator = DataCollatorSpeechSeq2Seq(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=make_compute_metrics(processor),
        processing_class=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # ── 7. 학습 시작 ──
    print(f"\n🚀 Whisper Tiny 발음 전사 파인튜닝 시작!")
    print(f"   모델: {BASE_MODEL} ({total_params/1e6:.0f}M params)")
    print(f"   Train: {len(splits['train']):,}개 / Val: {len(val_records):,}개")
    print(f"   LR: {lr} / Epochs: {num_epochs} / Batch: {batch_size}×{grad_accum}")
    print(f"   스텝/에폭: {steps_per_epoch:,} / 총 스텝: {total_steps:,}")
    print(f"   eval_steps: {eval_steps:,}")

    trainer.train()

    # ── 8. 저장 ──
    best_path = output_dir / "best"
    trainer.save_model(str(best_path))
    processor.save_pretrained(str(best_path))
    print(f"\n💾 최종 저장 완료: {best_path}")


# ────────────────────────────────────────────
# 8. CLI
# ────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper Tiny 발음 전사 파인튜닝")
    parser.add_argument("--json_dir", type=str, default=str(SEGMENT_DIR),
                        help="JSONL 데이터 디렉토리")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="모델 저장 경로")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="train 데이터 최대 개수 (0=무제한)")
    parser.add_argument("--apply_g2p", action="store_true",
                        help="transcript에서 G2P 재적용 (label 대신)")
    parser.add_argument("--dry_run", action="store_true",
                        help="데이터 검증만 수행")
    args = parser.parse_args()

    train(
        json_dir=args.json_dir,
        output_dir=args.output_dir,
        lr=args.lr,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_samples=args.max_samples,
        apply_g2p=args.apply_g2p,
        dry_run=args.dry_run,
    )
