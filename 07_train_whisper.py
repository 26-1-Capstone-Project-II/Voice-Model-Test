"""
STEP 7: Whisper 파인튜닝 — 구음장애 음성 ASR (1단계)
======================================================
모델: openai/whisper-small  (RTX 8GB↑ 권장)
      openai/whisper-medium (RTX 12GB↑)
      openai/whisper-large-v3 (RTX 24GB↑)

설치:
    pip install transformers datasets accelerate evaluate jiwer
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

실행 (학습):
    python 07_train_whisper.py --dataset_dir ./asr_dataset --output_dir ./whisper_output

실행 (추론 테스트):
    python 07_train_whisper.py --predict --model_dir ./whisper_output/best_model --audio_path ./test.wav
"""

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import torch
import numpy as np
import evaluate
from datasets import load_from_disk
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
)


# ────────────────────────────────────────────
# 0. 모델 선택 가이드
# ────────────────────────────────────────────
"""
RTX VRAM별 권장 모델:

  4~6 GB  → openai/whisper-small    (244M params) — 빠름, 기본 성능
  8~10 GB → openai/whisper-medium   (769M params) — ★ 권장 (성능/속도 균형)
  12+ GB  → openai/whisper-large-v3 (1.5B params) — 최고 성능

⚠️ 한국어 구음장애 특성상 large-v3가 확연히 유리하므로
   VRAM이 충분하면 large-v3 선택 권장.
"""

DEFAULT_MODEL = "openai/whisper-medium"
LANGUAGE      = "ko"
TASK          = "transcribe"


# ────────────────────────────────────────────
# 1. 데이터 콜레이터
# ────────────────────────────────────────────

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor:  Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        # 오디오 feature 추출 (input_features)
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # 레이블 (transcript 토큰) 패딩
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # -100으로 패딩 (loss 계산 제외)
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # BOS 토큰 제거 (decoder_start_token_id와 중복 방지)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ────────────────────────────────────────────
# 2. 전처리 함수
# ────────────────────────────────────────────

def make_preprocess_fn(processor):
    def preprocess(batch):
        # 오디오 → log-mel spectrogram
        audio = batch["audio_path"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_tensors="np",
        ).input_features[0]

        # 텍스트 → 토큰 ID
        batch["labels"] = processor.tokenizer(
            batch["transcript"],
            return_tensors="np",
        ).input_ids[0].tolist()

        return batch
    return preprocess


# ────────────────────────────────────────────
# 3. 평가 지표 (CER — 한국어는 WER보다 CER이 적합)
# ────────────────────────────────────────────

def make_compute_metrics(processor):
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids   = pred.predictions
        label_ids  = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

        # 공백 정규화
        pred_str  = [" ".join(p.split()) for p in pred_str]
        label_str = [" ".join(l.split()) for l in label_str]

        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {
            "cer": round(cer, 4),
            "wer": round(wer, 4),
        }
    return compute_metrics


# ────────────────────────────────────────────
# 4. 학습
# ────────────────────────────────────────────

def train(
    dataset_dir : Path,
    output_dir  : Path,
    model_name  : str   = DEFAULT_MODEL,
    batch_size  : int   = 16,
    num_epochs  : int   = 10,
    lr          : float = 1e-5,
    grad_accum  : int   = 2,
    max_steps   : int   = -1,    # -1이면 epochs 기준
):
    print(f"\n🤖 모델: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 디바이스: {device}")
    if device == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 프로세서 & 모델
    processor = WhisperProcessor.from_pretrained(
        model_name, language=LANGUAGE, task=TASK
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    model.generation_config.language = LANGUAGE
    model.generation_config.task     = TASK
    model.generation_config.forced_decoder_ids = None

    # 데이터셋 로드 & 전처리
    print(f"\n📂 데이터셋 로드: {dataset_dir}")
    raw_ds = load_from_disk(str(dataset_dir))

    preprocess_fn = make_preprocess_fn(processor)
    print("🔄 전처리 중 (처음 실행 시 시간 소요)...")
    ds = raw_ds.map(
        preprocess_fn,
        remove_columns=raw_ds["train"].column_names,
        num_proc=4,
        desc="오디오 → 스펙트로그램 변환",
    )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor               = processor,
        decoder_start_token_id  = model.config.decoder_start_token_id,
    )
    compute_metrics = make_compute_metrics(processor)

    # 학습 인자
    training_args = Seq2SeqTrainingArguments(
        output_dir                  = str(output_dir),
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        gradient_accumulation_steps = grad_accum,
        learning_rate               = lr,
        lr_scheduler_type           = "cosine",
        warmup_ratio                = 0.05,
        num_train_epochs            = num_epochs if max_steps == -1 else 999,
        max_steps                   = max_steps,
        weight_decay                = 0.01,
        fp16                        = (device == "cuda"),
        predict_with_generate       = True,
        generation_max_length       = 448,
        evaluation_strategy         = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "cer",
        greater_is_better           = False,   # CER은 낮을수록 좋음
        logging_steps               = 25,
        report_to                   = "none",
        dataloader_num_workers      = 4,
        # 긴 오디오 처리를 위한 설정
        dataloader_pin_memory       = True,
    )

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = training_args,
        train_dataset   = ds["train"],
        eval_dataset    = ds["validation"],
        tokenizer       = processor.feature_extractor,
        data_collator   = data_collator,
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("\n🚀 학습 시작!")
    trainer.train()

    print("\n📊 최종 평가:")
    results = trainer.evaluate(ds["test"])
    print(f"  Test CER: {results.get('eval_cer', 'N/A'):.4f}")
    print(f"  Test WER: {results.get('eval_wer', 'N/A'):.4f}")

    best_path = output_dir / "best_model"
    trainer.save_model(str(best_path))
    processor.save_pretrained(str(best_path))
    print(f"\n💾 모델 저장: {best_path}")


# ────────────────────────────────────────────
# 5. 추론 (단일 WAV 파일 테스트)
# ────────────────────────────────────────────

def predict(model_dir: Path, audio_path: str, target_text: str = ""):
    import librosa

    processor = WhisperProcessor.from_pretrained(str(model_dir))
    model     = WhisperForConditionalGeneration.from_pretrained(str(model_dir))
    model.eval()

    audio, sr = librosa.load(audio_path, sr=16000)
    inputs    = processor(audio, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        predicted_ids = model.generate(inputs.input_features)

    recognized = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    print(f"\n🎤 WAV: {audio_path}")
    if target_text:
        print(f"📝 목표:  {target_text}")
    print(f"🔍 인식:  {recognized}")

    if target_text:
        # 간단한 CER 계산
        from jiwer import cer
        score = cer(target_text, recognized)
        print(f"📊 CER:   {score:.4f} ({score*100:.1f}%)")

    return recognized


# ────────────────────────────────────────────
# 6. 메인
# ────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir",  default="./asr_dataset",    help="06번 출력 폴더")
    parser.add_argument("--output_dir",   default="./whisper_output",  help="모델 저장 폴더")
    parser.add_argument("--model_name",   default=DEFAULT_MODEL)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--num_epochs",   type=int,   default=10)
    parser.add_argument("--lr",           type=float, default=1e-5)
    parser.add_argument("--grad_accum",   type=int,   default=2)
    parser.add_argument("--max_steps",    type=int,   default=-1,     help="빠른 테스트용: 100")
    # 추론 모드
    parser.add_argument("--predict",      action="store_true")
    parser.add_argument("--model_dir",    default="./whisper_output/best_model")
    parser.add_argument("--audio_path",   default="")
    parser.add_argument("--target",       default="")
    args = parser.parse_args()

    if args.predict:
        if not args.audio_path:
            print("❌ --audio_path 를 지정하세요.")
        else:
            predict(Path(args.model_dir), args.audio_path, args.target)
    else:
        train(
            dataset_dir = Path(args.dataset_dir),
            output_dir  = Path(args.output_dir),
            model_name  = args.model_name,
            batch_size  = args.batch_size,
            num_epochs  = args.num_epochs,
            lr          = args.lr,
            grad_accum  = args.grad_accum,
            max_steps   = args.max_steps,
        )
