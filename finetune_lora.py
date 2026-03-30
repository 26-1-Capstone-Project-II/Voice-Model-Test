"""
밍글리 (Mingly) — 2단계 LoRA 파인튜닝 (발음평가용 G2P + 진행률 UI 버전)
======================================================
[적용된 트러블슈팅 황금 세팅]
  1. cuDNN 에러 방지 (torch.backends.cudnn.enabled = False)
  2. 메모리 최적화 (MAX_DURATION = 20.0)
  3. 역전파 단절 버그 방지 (gradient_checkpointing = False)
  4. 텍스트 특수기호([UNK]) 정규식 정제 (re.sub) + G2P(소리나는대로) 적용
  5. LoRA 맞춤형 학습률(3e-4) 및 기울기 폭발 방지(max_grad_norm=1.0)
  6. 성가신 경고 메시지 차단 & tqdm 진행률 바 추가
"""

import os
import json
import argparse
import warnings
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import numpy as np
from tqdm import tqdm  # 🚨 진행률 바를 위한 패키지 추가

# 원흉인 cuDNN 강제 비활성화
torch.backends.cudnn.enabled = False

# 🚨 성가신 Overflow 경고 등 모든 경고 메시지 차단
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# 0. 설정값
# ══════════════════════════════════════════════════════════════

BASE_MODEL_ID = "w11wo/wav2vec2-xls-r-300m-korean"
TARGET_SR     = 16_000   
MAX_DURATION  = 20.0     
MIN_DURATION  = 0.5      

LORA_CONFIG = dict(
    r              = 16,      
    lora_alpha     = 32,      
    lora_dropout   = 0.05,    
    target_modules = ["q_proj", "v_proj"],
    bias           = "none",
)

TRAIN_CONFIG = dict(
    learning_rate       = 3e-4,   
    warmup_ratio        = 0.1,   
    weight_decay        = 0.01,
    fp16                = False,  
    gradient_checkpointing = False, 
    dataloader_num_workers = 4,
)

# ══════════════════════════════════════════════════════════════
# 1. 패키지 설치 확인
# ══════════════════════════════════════════════════════════════

def check_and_install():
    required = {
        "transformers": "transformers>=4.35.0",
        "peft":         "peft>=0.9.0",
        "datasets":     "datasets>=2.14.0",
        "librosa":      "librosa",
        "jiwer":        "jiwer",
        "accelerate":   "accelerate>=0.26.0",
        "kss":          "kss", 
        "tqdm":         "tqdm",
    }
    missing = []
    for pkg, pip_name in required.items():
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print("\n[❌ 파인튜닝 필수 패키지 누락]")
        print(f"  pip install {' '.join(missing)}\n")
        raise SystemExit(1)

    print("[✅ 패키지 확인 완료]")

# ══════════════════════════════════════════════════════════════
# 2. 데이터셋 로드 & 전처리 (G2P 포함)
# ══════════════════════════════════════════════════════════════

def apply_g2p(text: str) -> str:
    try:
        from kss import Kss
        g2p = Kss("g2p")
        return g2p(text)
    except Exception:
        return text 

def discover_aihub_files(data_dir: str) -> List[Dict]:
    data_dir = Path(data_dir)
    samples = []

    json_files = list(data_dir.rglob("*.json"))
    print(f"  JSON 라벨 파일 발견: {len(json_files)}개")

    audio_index = {}
    for wav in data_dir.rglob("*.wav"):
        audio_index[wav.name] = wav
    for flac in data_dir.rglob("*.flac"):
        audio_index[flac.name] = flac
    print(f"  오디오 파일 인덱스: {len(audio_index)}개")

    print("\n  [주의] G2P 변환 작업이 진행 중입니다. (약 5~10분 소요)")
    # 🚨 tqdm으로 게이지 바 추가! 얼마나 남았는지 확인 가능
    for jf in tqdm(json_files, desc="  G2P 텍스트 변환"):
        try:
            with open(jf, encoding="utf-8") as f:
                meta = json.load(f)

            text = (
                meta.get("Transcript")
                or meta.get("transcript")
                or meta.get("transcription", {}).get("kor")
                or meta.get("발화내용")
                or meta.get("text")
            )
            if not text:
                continue

            text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            
            if not text: 
                continue

            text = apply_g2p(text)

            file_id = meta.get("File_id", "")
            audio_path = audio_index.get(file_id)

            if audio_path is None:
                audio_path = audio_index.get(jf.stem + ".wav")

            if audio_path is None:
                audio_dir = str(jf.parent).replace(
                    "라벨링데이터_250331_add", "원천데이터"
                ).replace(
                    "라벨링데이터", "원천데이터"
                )
                for ext in [".wav", ".flac", ".mp3"]:
                    candidate = Path(audio_dir) / (jf.stem + ext)
                    if candidate.exists():
                        audio_path = candidate
                        break

            if audio_path is None:
                continue

            samples.append({"audio": str(audio_path), "text": text})

        except Exception:
            continue

    print(f"\n  유효한 (오디오, 라벨) 쌍: {len(samples)}개")
    return samples

def load_audio(path: str) -> Optional[np.ndarray]:
    try:
        import soundfile as sf
        from scipy.signal import resample_poly
        from math import gcd

        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)

        if sr != TARGET_SR:
            g = gcd(TARGET_SR, sr)
            audio = resample_poly(audio, TARGET_SR // g, sr // g)

        audio = audio.astype(np.float32)
        duration = len(audio) / TARGET_SR

        if duration < MIN_DURATION:
            return None

        if duration > MAX_DURATION:
            audio = audio[:int(MAX_DURATION * TARGET_SR)]

        return audio
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════
# 3. Data Collator (CTC 학습용 배치 패딩)
# ══════════════════════════════════════════════════════════════

@dataclass
class DataCollatorCTCWithPadding:
    processor: object
    padding: Union[bool, str] = True

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:

        input_features = [
            {"input_values": f["input_values"]} for f in features
        ]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        batch = {"input_values": batch["input_values"]}

        label_list = [f["labels"] for f in features]
        max_len    = max(len(l) for l in label_list)
        labels_padded = []
        for lbl in label_list:
            pad_len = max_len - len(lbl)
            labels_padded.append(lbl + [-100] * pad_len)

        batch["labels"] = torch.tensor(labels_padded, dtype=torch.long)

        return batch

# ══════════════════════════════════════════════════════════════
# 4. 평가 지표 (CER)
# ══════════════════════════════════════════════════════════════

def build_compute_metrics(processor):
    from jiwer import cer as compute_cer

    def compute_metrics(pred):
        pred_logits = pred.predictions           
        pred_ids    = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str  = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        if len(pred_str) > 0 and len(label_str) > 0:
            print(f"\n  정답(G2P) : {label_str[0]}")
            print(f"  예측      : {pred_str[0]}\n")

        cer = compute_cer(label_str, pred_str)
        return {"cer": round(cer, 4)}

    return compute_metrics

# ══════════════════════════════════════════════════════════════
# 5. 메인 파인튜닝 함수
# ══════════════════════════════════════════════════════════════

def finetune(args):
    from transformers import (
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model

    print("\n[1/6] 데이터셋 탐색 중...")
    samples = discover_aihub_files(args.data_dir)

    if len(samples) == 0:
        print("[❌] 유효한 데이터를 찾지 못했습니다.")
        raise SystemExit(1)

    split_idx  = int(len(samples) * 0.9)
    train_data = samples[:split_idx]
    eval_data  = samples[split_idx:]
    print(f"  학습: {len(train_data)}개 | 검증: {len(eval_data)}개")

    print(f"\n[2/6] 베이스 모델 로드: {BASE_MODEL_ID}")
    processor = Wav2Vec2Processor.from_pretrained(BASE_MODEL_ID)
    model     = Wav2Vec2ForCTC.from_pretrained(
        BASE_MODEL_ID,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        ignore_mismatched_sizes=True,
    )

    model.freeze_feature_encoder()
    print("  feature encoder 동결 완료 (VRAM 절약)")

    print("\n[3/6] LoRA 어댑터 설정 중...")
    lora_config = LoraConfig(
        **LORA_CONFIG,
        inference_mode = False,
    )
    model = get_peft_model(model, lora_config)

    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    pct = 100 * trainable / total
    print(f"  학습 파라미터: {trainable:,} / {total:,} ({pct:.2f}%)")

    print("\n[4/6] 오디오 & 라벨 전처리 중...")

    class SpeechDataset(torch.utils.data.Dataset):
        def __init__(self, samples, processor, desc=""):
            self.valid = []
            skip_token = 0

            # 🚨 여기도 tqdm 적용하여 진행률 표시!
            for sample in tqdm(samples, desc=f"  [{desc}] 라벨 토크나이징"):
                try:
                    with processor.as_target_processor():
                        labels = processor(sample["text"]).input_ids
                    if len(labels) == 0:
                        skip_token += 1
                        continue
                    self.valid.append({
                        "audio_path": sample["audio"],
                        "labels":     labels,
                        "text":       sample["text"],
                    })
                except Exception:
                    skip_token += 1

            print(f"  [{desc}] 완료: {len(self.valid)}개 ")

        def __len__(self):
            return len(self.valid)

        def __getitem__(self, idx):
            item = self.valid[idx]
            audio = load_audio(item["audio_path"])
            if audio is None:
                audio = np.zeros(TARGET_SR, dtype=np.float32)

            input_values = processor(
                audio,
                sampling_rate=TARGET_SR,
            ).input_values[0]

            return {
                "input_values": input_values,
                "labels":       item["labels"],
            }

    train_dataset = SpeechDataset(train_data, processor, desc="학습")
    eval_dataset  = SpeechDataset(eval_data,  processor, desc="검증")

    print("\n[5/6] 학습 설정 구성 중...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = torch.cuda.is_available() and TRAIN_CONFIG["fp16"]

    training_args = TrainingArguments(
        output_dir                  = str(output_dir),
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,  
        learning_rate               = TRAIN_CONFIG["learning_rate"],
        warmup_ratio                = TRAIN_CONFIG["warmup_ratio"],
        weight_decay                = TRAIN_CONFIG["weight_decay"],
        fp16                        = use_fp16,
        gradient_checkpointing      = TRAIN_CONFIG["gradient_checkpointing"],
        max_grad_norm               = 1.0, 
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "cer",
        greater_is_better           = False,  
        logging_steps               = 20, 
        logging_dir                 = str(output_dir / "logs"),
        dataloader_num_workers      = 0,          
        report_to                   = "none",     
        push_to_hub                 = False,
        group_by_length             = False,      
    )

    data_collator    = DataCollatorCTCWithPadding(processor=processor)
    compute_metrics  = build_compute_metrics(processor)

    trainer = Trainer(
        model              = model,
        args               = training_args,
        train_dataset      = train_dataset,
        eval_dataset       = eval_dataset,
        data_collator      = data_collator,
        compute_metrics    = compute_metrics,
    )

    print(f"\n[6/6] LoRA 파인튜닝 시작!")
    print(f"  출력 경로  : {output_dir}")
    print(f"  에폭       : {args.epochs}")
    print(f"  배치 크기  : {args.batch_size} × {args.grad_accum} (누적)")
    print(f"  GPU 상태   : {'✅ 사용 가능' if torch.cuda.is_available() else '❌ CPU 강제 구동'}")
    print(f"  FP16       : {'✅ 활성' if use_fp16 else '❌ 비활성 (안정성 확보)'}\n")

    trainer.train()

    adapter_dir = output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))

    print(f"\n{'═'*56}")
    print(f"  ✅ 학습 완료!")
    print(f"  LoRA 어댑터 저장 위치: {adapter_dir}")
    print(f"{'═'*56}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="~/mingly_workspace/dataset")
    parser.add_argument("--output_dir", type=str, default="~/mingly_workspace/lora_adapter")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--lora_r", type=int, default=16)

    args = parser.parse_args()

    args.data_dir   = str(Path(args.data_dir).expanduser())
    args.output_dir = str(Path(args.output_dir).expanduser())

    LORA_CONFIG["r"]          = args.lora_r
    LORA_CONFIG["lora_alpha"] = args.lora_r * 2

    print("=" * 56)
    print("  밍글리 (Mingly) — LoRA 파인튜닝 (발음평가용 G2P 버전)")
    print("=" * 56)

    check_and_install()
    finetune(args)

if __name__ == "__main__":
    main()