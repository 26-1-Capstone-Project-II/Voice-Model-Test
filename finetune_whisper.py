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
BASE_MODEL   = "openai/whisper-base"   # tiny → base (≈74M, tiny의 2배)
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
    """Whisper 입력용 Dataset: 오디오 → mel spectrogram, 라벨 → token IDs.

    long-form: train 에서 일부 샘플을 여러 발화로 이어붙여 ~30초 연속 발화를 만든다.
    Zeroth 는 짧은 낭독 세그먼트뿐이라, 모델이 30초 윈도우 후반부를 학습한 적이 없어
    그곳에서 환각이 발생한다. 이어붙이기로 후반부 분포를 직접 채운다.
    """

    def __init__(self, records, processor, augmentor=None,
                 longform_prob=0.0, longform_max_sec=28.0, longform_max_chars=180,
                 noise_only_prob=0.0):
        self.records = records
        self.processor = processor
        self.augmentor = augmentor          # train split 에서만 전달 (val/test=None)
        self.longform_prob = longform_prob
        self.longform_max_sec = longform_max_sec
        self.longform_max_chars = longform_max_chars
        # 순수 비음성(노이즈/무음) → 빈 라벨 비율. "비음성 = 출력 없음(EOS)" 학습으로
        # noise-only 환각을 근본 억제. 과하면 실제 발화를 조기 EOS 할 수 있어 낮게 유지.
        self.noise_only_prob = noise_only_prob

    def __len__(self):
        return len(self.records)

    def _load(self, rec):
        try:
            audio, _ = librosa.load(rec["wav_path"], sr=TARGET_SR, mono=True)
        except Exception:
            audio = np.zeros(TARGET_SR, dtype=np.float32)   # 로드 실패 시 1초 무음
        return audio.astype(np.float32)

    def _make_noise_only(self):
        """순수 비음성 클립 생성 (노이즈/무음, 라벨 없음). 노이즈 파일 없으면 None."""
        import random
        if not (self.augmentor is not None and self.augmentor.noise_files):
            return None
        from augment import _load_audio, _match_length
        sec = random.uniform(3.0, 20.0)
        length = int(sec * TARGET_SR)
        noise = _match_length(_load_audio(random.choice(self.augmentor.noise_files)), length)
        peak = float(np.max(np.abs(noise))) + 1e-8
        noise = (random.uniform(0.05, 0.5) * noise / peak).astype(np.float32)  # 레벨 다양화
        if self.augmentor.rir_files and random.random() < 0.3:
            noise = self.augmentor._reverberate(noise)               # 잔향 섞인 비음성도
        return noise

    def _build_longform(self, idx):
        """idx 부터 연속 발화를 이어붙여 (audio, label) 생성. 발화 사이 짧은 무음 삽입."""
        import random
        target_sec = random.uniform(15.0, self.longform_max_sec)
        audios, labels, total_sec, n = [], [], 0.0, len(self.records)
        for k in range(8):                                  # 최대 8개까지 결합
            rec = self.records[(idx + k) % n]
            a = self._load(rec)
            audios.append(a)
            labels.append(rec["label"])
            total_sec += len(a) / TARGET_SR
            if k < 7:
                gap = np.zeros(int(random.uniform(0.1, 0.3) * TARGET_SR), dtype=np.float32)
                audios.append(gap)
                total_sec += len(gap) / TARGET_SR
            joined_chars = sum(len(l) for l in labels)
            if total_sec >= target_sec or joined_chars >= self.longform_max_chars:
                break
        audio = np.concatenate(audios).astype(np.float32)
        label = " ".join(labels)
        return audio, label

    def __getitem__(self, idx):
        import random
        rec = self.records[idx]

        # 순수 비음성 → 빈 라벨 (train 전용). augmentor 가 추가 열화하지 않도록 별도 분기.
        noise_only = None
        if self.noise_only_prob > 0 and random.random() < self.noise_only_prob:
            noise_only = self._make_noise_only()

        if noise_only is not None:
            audio, label = noise_only, ""
        elif self.longform_prob > 0 and random.random() < self.longform_prob:
            audio, label = self._build_longform(idx)      # long-form 결합
        else:
            audio, label = self._load(rec), rec["label"]  # 단일 발화

        # 30초 이하로 자르기 (Whisper 윈도우)
        max_samples = 30 * TARGET_SR
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # 원거리/소음 증강 (feature extractor 이전, 파형 도메인 / train 전용)
        # 비음성 클립은 이미 노이즈이므로 추가 증강을 건너뛴다.
        if self.augmentor is not None and noise_only is None:
            audio = self.augmentor(audio)

        # Log-mel spectrogram (Whisper feature extractor)
        input_features = self.processor.feature_extractor(
            audio, sampling_rate=TARGET_SR, return_tensors="np"
        ).input_features[0]

        # 라벨 토큰화 (발음 전사 텍스트)
        labels = self.processor.tokenizer(label).input_ids

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
    use_augment=True,
    p_tail=0.3,
    longform_prob=0.3,
    longform_max_sec=28.0,
    noise_only_prob=0.05,
    competing_prob=0.0,
    repetition_penalty=1.0,
    no_repeat_ngram_size=0,
    init_model=None,
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

    # 초기 가중치: init_model 이 주어지면 그 체크포인트에서 *이어서* 학습한다.
    # (성숙한 best_model_zeroth_aug/best 에 새 증강만 얹어 깨끗한 정확도 보존)
    # 없으면 생짜 BASE_MODEL 에서 처음부터 학습.
    src_model = init_model or BASE_MODEL

    # ── 1. Processor 로드 ──
    print(f"\n📥 Processor 로드: {src_model}")
    processor = WhisperProcessor.from_pretrained(
        src_model, language="ko", task="transcribe"
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
    # train 에만 원거리/소음/경쟁화자 증강 적용. val/test 는 clean 으로 일반화 측정.
    augmentor = None
    if use_augment:
        from augment import FarFieldAugmentor
        # 경쟁 화자(간섭원) 풀 = held-out test 스플릿 발화 (OpenSLR-40 train/test 화자 배타적).
        # train 발화를 쓰지 않으므로 화자 단위 배타·라벨 무결성 요건 충족 (플랜 §7).
        competing_files = []
        if competing_prob > 0:
            competing_files = [r["wav_path"] for r in splits.get("test", []) if r.get("wav_path")]
            if not competing_files:
                print("⚠️ 경쟁 화자 증강 요청됐으나 test 스플릿 발화가 없음 → 비활성화")
            else:
                print(f"🗣️ 경쟁 화자 증강: held-out(test) 간섭 풀 {len(competing_files):,}개, "
                      f"p_competing={competing_prob}")
        augmentor = FarFieldAugmentor(
            noise_root=os.environ.get("MUSAN_NOISE_DIR", "/data/musan/noise"),
            rir_root=os.environ.get("RIR_DIR", "/data/RIRS_NOISES/simulated_rirs"),
            snr_db_range=(5.0, 20.0),
            p_tail=p_tail,                  # 비음성 꼬리 → 후반부 환각 억제
            speech_files=competing_files,   # held-out 간섭 화자 (경쟁 화자 증강)
            p_competing=competing_prob,
        )
    else:
        print("⚠️ 증강 비활성화 (--no_augment) — clean 학습")
    # train 에만 증강 + long-form 결합. val 은 clean 단일 발화로 일반화 측정.
    train_ds = WhisperPhoneticDataset(
        splits["train"], processor, augmentor=augmentor,
        longform_prob=longform_prob, longform_max_sec=longform_max_sec,
        noise_only_prob=noise_only_prob,
    )
    val_records = splits.get("validation", splits["train"][:500])[:500]
    val_ds = WhisperPhoneticDataset(val_records, processor, augmentor=None)

    # ── 4. 모델 로드 ──
    print(f"\n📥 모델 로드: {src_model}" + ("  (이어서 학습)" if init_model else "  (생짜에서 학습)"))
    model = WhisperForConditionalGeneration.from_pretrained(src_model)

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
    # 디코딩 anti-repeat. 기본값을 앱(WhisperKit, neutral greedy) 과 맞춰 1.0/0 으로 두면,
    # 학습 중 eval 지표가 앱이 실제로 겪는 환각을 반영한다(데이터로 환각을 고치는지 검증).
    # 이전 crutch(1.2 / 3) 가 필요하면 CLI 로 지정.
    model.generation_config.no_repeat_ngram_size = no_repeat_ngram_size
    model.generation_config.repetition_penalty = repetition_penalty
    print(f"  🎚️  디코딩: repetition_penalty={repetition_penalty}, "
          f"no_repeat_ngram_size={no_repeat_ngram_size} "
          f"({'앱 일치(neutral)' if repetition_penalty == 1.0 and no_repeat_ngram_size == 0 else 'anti-repeat'})")

    # SpecAugment (학습 시에만 적용, eval 영향 없음) — 소음 증강과 함께 robustness 강화
    model.config.apply_spec_augment   = True
    model.config.mask_time_prob       = 0.05   # 시간축 마스크
    model.config.mask_time_length     = 10
    model.config.mask_feature_prob    = 0.05   # 주파수축 마스크
    model.config.mask_feature_length  = 10

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
        dataloader_num_workers=6,           # 증강이 CPU 부하를 더함 → 워커 증가
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
        callbacks=[EarlyStoppingCallback(early_stopping_patience=8)],
    )

    # ── 7. 학습 시작 ──
    print(f"\n🚀 Whisper 발음 전사 파인튜닝 시작!")
    print(f"   모델: {src_model} ({total_params/1e6:.0f}M params)")
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
    parser.add_argument("--num_epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="train 데이터 최대 개수 (0=무제한)")
    parser.add_argument("--apply_g2p", action="store_true",
                        help="transcript에서 G2P 재적용 (label 대신)")
    parser.add_argument("--dry_run", action="store_true",
                        help="데이터 검증만 수행")
    parser.add_argument("--no_augment", action="store_true",
                        help="원거리/소음 증강 비활성화 (clean 베이스라인 진단용)")
    parser.add_argument("--p_tail", type=float, default=0.3,
                        help="비음성 꼬리 증강 확률 (후반부 환각 억제)")
    parser.add_argument("--longform_prob", type=float, default=0.3,
                        help="여러 발화 이어붙인 long-form 샘플 비율 (>30초 후반부 커버)")
    parser.add_argument("--longform_max_sec", type=float, default=28.0,
                        help="long-form 목표 최대 길이(초)")
    parser.add_argument("--noise_only_prob", type=float, default=0.05,
                        help="순수 비음성→빈 라벨 샘플 비율 (noise-only 환각 억제). 과하면 조기 EOS 위험")
    parser.add_argument("--competing_prob", type=float, default=0.0,
                        help="경쟁 화자 증강 확률 (플랜 §7). held-out(test) 다른 화자를 타깃 우세 "
                             "SIR 5~20dB(일부 0~5dB)로 부분 겹침 혼합. 라벨은 타깃만 유지. 0=비활성")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="기본 1.0=앱(WhisperKit) 일치. 1.2 등으로 anti-repeat crutch 사용 가능")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0,
                        help="기본 0=앱 일치. 3 등으로 반복 억제 crutch 사용 가능")
    parser.add_argument("--init_model", type=str, default=None,
                        help="이 체크포인트에서 이어서 학습 (예: best_model_zeroth_aug/best). "
                             "미지정 시 openai/whisper-base 에서 처음부터")
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
        use_augment=not args.no_augment,
        p_tail=args.p_tail,
        longform_prob=args.longform_prob,
        longform_max_sec=args.longform_max_sec,
        noise_only_prob=args.noise_only_prob,
        competing_prob=args.competing_prob,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        init_model=args.init_model,
    )
