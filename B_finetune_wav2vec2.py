"""
STEP B: wav2vec2-xls-r-300m-korean 파인튜닝 (구음장애 음성)
=============================================================
목적: 구음장애 음성을 G2P 발음 전사에 가깝게 인식하도록 CTC 파인튜닝
모델: w11wo/wav2vec2-xls-r-300m-korean (1단계와 동일 모델)
라벨: G2P 변환된 발음 전사 (1단계 채점 기준과 일치)

설치:
    pip install transformers datasets accelerate evaluate jiwer librosa soundfile tqdm

실행 (학습):
    python B_finetune_wav2vec2.py --dataset_dir ./finetune_dataset --output_dir ./finetuned_model

실행 (추론 테스트):
    python B_finetune_wav2vec2.py --predict --model_dir ./finetuned_model/best --audio my_voice.wav --text "같이 해볼까"
"""

import json
import argparse
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional

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


BASE_MODEL = "w11wo/wav2vec2-xls-r-300m-korean"
TARGET_SR  = 16000   # wav2vec2 요구 샘플링 레이트


# ────────────────────────────────────────────
# 1. Dataset 클래스
# ────────────────────────────────────────────

class DysarthriaDataset(Dataset):
    """
    JSONL 파일을 읽어 wav2vec2 입력으로 변환.
    오디오는 학습 시 on-the-fly로 로딩 → 메모리 효율적
    """

    def __init__(
        self,
        jsonl_path : Path,
        processor  : Wav2Vec2Processor,
        max_length_sec: float = 30.0,
    ):
        self.processor      = processor
        self.max_samples    = int(max_length_sec * TARGET_SR)
        self.records        = []

        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line.strip())
                if obj.get("label") and obj.get("audio_path"):
                    self.records.append(obj)

        print(f"  [{jsonl_path.name}] {len(self.records):,}개 로드")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        # 오디오 로딩 + 16kHz 리샘플링
        try:
            audio, _ = librosa.load(rec["audio_path"], sr=TARGET_SR, mono=True)
        except Exception as e:
            # 오디오 로딩 실패 시 무음으로 대체 (학습 중단 방지)
            audio = np.zeros(TARGET_SR, dtype=np.float32)

        # 최대 길이 자르기
        audio = audio[:self.max_samples]

        # wav2vec2 feature 추출
        input_values = self.processor(
            audio,
            sampling_rate = TARGET_SR,
            return_tensors = "pt",
            padding = False,
        ).input_values[0]

        # 라벨 토크나이즈 (G2P 발음 전사)
        with self.processor.as_target_processor():
            labels = self.processor(rec["label"]).input_ids

        return {
            "input_values" : input_values,
            "labels"       : torch.tensor(labels, dtype=torch.long),
        }


# ────────────────────────────────────────────
# 2. 데이터 콜레이터 (가변 길이 패딩)
# ────────────────────────────────────────────

@dataclass
class DataCollatorCTCWithPadding:
    processor    : Any
    padding      : bool = True

    def __call__(self, features: list[dict]) -> dict:
        # input_values 패딩
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids":    f["labels"]}        for f in features]

        batch = self.processor.pad(
            input_features,
            padding     = self.padding,
            return_tensors = "pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding     = self.padding,
                return_tensors = "pt",
            )

        # -100: loss 계산에서 제외 (패딩 토큰)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


# ────────────────────────────────────────────
# 3. 평가 지표 (CER — 한국어 음절 단위)
# ────────────────────────────────────────────

def make_compute_metrics(processor: Wav2Vec2Processor):
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids    = np.argmax(pred_logits, axis=-1)
        label_ids   = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)

        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        # 샘플 출력 (진행 상황 파악용)
        for i in range(min(3, len(pred_str))):
            print(f"\n  [샘플 {i+1}]")
            print(f"  정답: {label_str[i]}")
            print(f"  예측: {pred_str[i]}")

        return {"cer": round(cer, 4), "wer": round(wer, 4)}

    return compute_metrics


# ────────────────────────────────────────────
# 4. 파인튜닝 전략
# ────────────────────────────────────────────
"""
[Feature Extractor 동결 전략]
wav2vec2는 크게 두 부분으로 구성:
  1. CNN Feature Extractor  → 음향 특징 추출 (저수준, 구음장애 범용적)
  2. Transformer Encoder    → 음성 패턴 학습 (고수준, 언어/장애 특화)

파인튜닝 시 Feature Extractor는 동결하고 Transformer만 학습:
  → 학습 속도 2-3배 향상
  → 과적합 방지
  → 구음장애 특화 패턴 학습에 집중
"""


# ────────────────────────────────────────────
# 5. 학습
# ────────────────────────────────────────────

def train(
    dataset_dir   : Path,
    output_dir    : Path,
    model_name    : str   = BASE_MODEL,
    batch_size    : int   = 8,
    num_epochs    : int   = 30,
    lr            : float = 1e-4,
    grad_accum    : int   = 4,
    freeze_feature_extractor: bool = True,
):
    print(f"\n🤖 베이스 모델: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 디바이스: {device}")
    if device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU: {torch.cuda.get_device_name(0)}  VRAM: {vram:.1f}GB")
        if vram < 8:
            print("   ⚠️  VRAM < 8GB: batch_size를 4로 낮추고 grad_accum을 8로 올리세요")

    # 프로세서 & 모델 로드
    print(f"\n📥 모델 로딩 중...")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model     = Wav2Vec2ForCTC.from_pretrained(
        model_name,
        ctc_loss_reduction = "mean",
        pad_token_id       = processor.tokenizer.pad_token_id,
    )

    # Feature Extractor 동결 (Transformer만 파인튜닝)
    if freeze_feature_extractor:
        model.freeze_feature_extractor()
        frozen_params = sum(p.numel() for p in model.feature_extractor.parameters())
        total_params  = sum(p.numel() for p in model.parameters())
        print(f"🔒 Feature Extractor 동결: {frozen_params:,} params")
        print(f"🔓 학습 파라미터: {total_params - frozen_params:,} params")

    # 데이터셋
    print(f"\n📂 데이터셋 로드: {dataset_dir}")
    train_ds = DysarthriaDataset(dataset_dir / "train.jsonl",      processor)
    val_ds   = DysarthriaDataset(dataset_dir / "validation.jsonl", processor)

    data_collator   = DataCollatorCTCWithPadding(processor=processor)
    compute_metrics = make_compute_metrics(processor)

    # 학습 인자
    training_args = TrainingArguments(
        output_dir                  = str(output_dir),
        group_by_length             = True,   # 비슷한 길이끼리 배치 → 패딩 최소화
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        gradient_accumulation_steps = grad_accum,
        eval_accumulation_steps     = 10,
        num_train_epochs            = num_epochs,
        learning_rate               = lr,
        weight_decay                = 0.005,
        warmup_ratio                = 0.1,
        lr_scheduler_type           = "cosine",
        fp16                        = (device == "cuda"),
        evaluation_strategy         = "epoch",
        save_strategy               = "epoch",
        logging_steps               = 50,
        load_best_model_at_end      = True,
        metric_for_best_model       = "cer",
        greater_is_better           = False,
        save_total_limit            = 3,
        report_to                   = "none",
        dataloader_num_workers      = 2,
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = val_ds,
        tokenizer       = processor.feature_extractor,
        data_collator   = data_collator,
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=5)],
    )

    print("\n🚀 파인튜닝 시작!")
    trainer.train()

    # 최종 테스트 평가
    print("\n📊 Test 셋 최종 평가:")
    test_ds      = DysarthriaDataset(dataset_dir / "test.jsonl", processor)
    test_results = trainer.evaluate(test_ds)
    print(f"  CER: {test_results.get('eval_cer', 'N/A'):.4f}")
    print(f"  WER: {test_results.get('eval_wer', 'N/A'):.4f}")

    # 저장
    best_path = output_dir / "best"
    trainer.save_model(str(best_path))
    processor.save_pretrained(str(best_path))
    print(f"\n💾 모델 저장 완료: {best_path}")
    print("   다음: python C_integrate.py 로 pronunciation_scorer.py와 통합")


# ────────────────────────────────────────────
# 6. 추론 — pronunciation_scorer.py와 동일한 방식으로 테스트
# ────────────────────────────────────────────

def predict(model_dir: Path, audio_path: str, target_text: str = ""):
    """
    파인튜닝 모델 단독 추론 테스트.
    pronunciation_scorer.py의 흐름과 동일하게 동작하는지 확인.
    """
    from g2pk import G2p

    print(f"\n📥 모델 로딩: {model_dir}")
    processor = Wav2Vec2Processor.from_pretrained(str(model_dir))
    model     = Wav2Vec2ForCTC.from_pretrained(str(model_dir))
    model.eval()
    g2p = G2p()

    # 오디오 로딩
    audio, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
    inputs   = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    # Greedy Decoding (1단계와 동일)
    pred_ids     = torch.argmax(logits, dim=-1)
    recognized   = processor.batch_decode(pred_ids)[0]

    print(f"\n🎤 오디오: {audio_path}")
    if target_text:
        g2p_target = g2p(target_text, descriptive=True)
        print(f"📝 목표 (맞춤법): {target_text}")
        print(f"📝 목표 (G2P):    {g2p_target}")
    print(f"🔍 인식 결과:     {recognized}")

    if target_text:
        from jiwer import cer as jiwer_cer
        g2p_target = g2p(target_text, descriptive=True)
        score = max(0, 100 - jiwer_cer(g2p_target, recognized) * 100)
        print(f"📊 발음 점수:     {score:.1f}점")

    return recognized


# ────────────────────────────────────────────
# 7. 메인
# ────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",  default="./finetune_dataset")
    parser.add_argument("--output_dir",   default="./finetuned_model")
    parser.add_argument("--model_name",   default=BASE_MODEL)
    parser.add_argument("--batch_size",   type=int,   default=8)
    parser.add_argument("--num_epochs",   type=int,   default=30)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--grad_accum",   type=int,   default=4)
    parser.add_argument("--no_freeze",    action="store_true", help="Feature Extractor 동결 해제")
    # 추론 모드
    parser.add_argument("--predict",      action="store_true")
    parser.add_argument("--model_dir",    default="./finetuned_model/best")
    parser.add_argument("--audio",        default="")
    parser.add_argument("--text",         default="")
    args = parser.parse_args()

    if args.predict:
        if not args.audio:
            print("❌ --audio 경로를 지정하세요.")
        else:
            predict(Path(args.model_dir), args.audio, args.text)
    else:
        train(
            dataset_dir              = Path(args.dataset_dir),
            output_dir               = Path(args.output_dir),
            model_name               = args.model_name,
            batch_size               = args.batch_size,
            num_epochs               = args.num_epochs,
            lr                       = args.lr,
            grad_accum               = args.grad_accum,
            freeze_feature_extractor = not args.no_freeze,
        )
