"""
밍글리 (Mingly) — 2단계 LoRA 파인튜닝
=====================================
베이스 모델 : w11wo/wav2vec2-xls-r-300m-korean
PEFT 기법   : LoRA (Low-Rank Adaptation)
데이터셋    : AIHub 구음장애 음성인식 데이터 (dataSetSn=608)
목표        : 구음장애 음성에 특화된 발음 전사 모델 학습

기존 파일(pronunciation_scorer.py)은 건드리지 않습니다.
파인튜닝된 LoRA 어댑터만 별도 저장 → 기존 파일에서 로드해서 사용 가능.

학습 전략:
  - 전체 파라미터 파인튜닝(Full FT) 대신 LoRA로 소수의 파라미터만 업데이트
  - GTX 1660 Super 수준에서도 동작 가능하도록 gradient_checkpointing 적용
  - CTC Loss 기반 학습 (wav2vec2 구조 그대로 유지)

실행 방법:
  python finetune_lora.py --data_dir ~/mingly_workspace/dataset \
                          --output_dir ~/mingly_workspace/lora_adapter \
                          --epochs 5 \
                          --batch_size 4

LoRA 어댑터 로드 방법 (추론 시):
  from peft import PeftModel
  from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

  base_model = Wav2Vec2ForCTC.from_pretrained("w11wo/wav2vec2-xls-r-300m-korean")
  model = PeftModel.from_pretrained(base_model, "./lora_adapter")
"""

import os
import json
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# ══════════════════════════════════════════════════════════════
# 0. 설정값 (argparse로 덮어쓰기 가능)
# ══════════════════════════════════════════════════════════════

BASE_MODEL_ID = "w11wo/wav2vec2-xls-r-300m-korean"
TARGET_SR     = 16_000   # wav2vec2 요구 샘플레이트
MAX_DURATION  = 120.0    # 최대 오디오 길이 (초) — AIHub 데이터 최대 114초 대응
MIN_DURATION  = 0.5      # 최소 오디오 길이 (초) — 이 이하는 스킵

# ── LoRA 하이퍼파라미터 ──────────────────────────────────────
LORA_CONFIG = dict(
    r              = 16,      # 랭크: 작을수록 파라미터 적음 (8~32 권장)
    lora_alpha     = 32,      # 스케일링 계수 (보통 r의 2배)
    lora_dropout   = 0.05,    # 드롭아웃 (과적합 방지)
    # wav2vec2 Transformer 블록의 어텐션 Q·V 프로젝션에만 LoRA 적용
    target_modules = ["q_proj", "v_proj"],
    bias           = "none",
)

# ── 학습 하이퍼파라미터 ──────────────────────────────────────
TRAIN_CONFIG = dict(
    learning_rate       = 1e-4,
    warmup_ratio        = 0.1,   # 전체 스텝의 10%를 워밍업
    weight_decay        = 0.01,
    fp16                = True,   # GPU 메모리 절약 (CUDA 필수)
    gradient_checkpointing = True,  # VRAM 절약 (속도 ↔ 메모리 트레이드오프)
    dataloader_num_workers = 4,
)

# ══════════════════════════════════════════════════════════════
# 1. 패키지 설치 확인
# ══════════════════════════════════════════════════════════════

def check_and_install():
    """필요한 패키지가 없으면 설치 안내를 출력합니다."""
    required = {
        "transformers": "transformers>=4.35.0",
        "peft":         "peft>=0.9.0",
        "datasets":     "datasets>=2.14.0",
        "librosa":      "librosa",
        "jiwer":        "jiwer",
        "accelerate":   "accelerate>=0.26.0",
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
# 2. 데이터셋 로드 & 전처리
# ══════════════════════════════════════════════════════════════

def discover_aihub_files(data_dir: str) -> List[Dict]:
    """
    AIHub 구음장애 데이터셋(608) 실제 구조에 맞게 탐색합니다.

    실제 데이터셋 구조:
        013.구음장애_음성인식_데이터/01.데이터/1.Training/
        ├── 원천데이터/
        │   └── 21.조음/*.wav          ← 오디오 파일
        └── 라벨링데이터_250331_add/
            └── 21.조음/*.json         ← 라벨 파일

    JSON 실제 구조:
        {
            "Transcript": "발화 텍스트",   ← 텍스트
            "File_id": "ID-xxx.wav",       ← 대응 wav 파일명
            ...
        }
    """
    data_dir = Path(data_dir)
    samples = []

    # JSON 라벨 파일 수집 (재귀 탐색)
    json_files = list(data_dir.rglob("*.json"))
    print(f"  JSON 라벨 파일 발견: {len(json_files)}개")

    # 오디오 파일 전체를 파일명 기준으로 인덱싱 (빠른 탐색)
    print(f"  오디오 파일 인덱싱 중...")
    audio_index = {}
    for wav in data_dir.rglob("*.wav"):
        audio_index[wav.name] = wav
    for flac in data_dir.rglob("*.flac"):
        audio_index[flac.name] = flac
    print(f"  오디오 파일 인덱스: {len(audio_index)}개")

    for jf in json_files:
        try:
            with open(jf, encoding="utf-8") as f:
                meta = json.load(f)

            # ── 텍스트 추출 ───────────────────────────────────
            # 실제 키: "Transcript" (대문자 T)
            text = (
                meta.get("Transcript")
                or meta.get("transcript")
                or meta.get("transcription", {}).get("kor")
                or meta.get("발화내용")
                or meta.get("text")
            )
            if not text:
                continue

            # ── 오디오 파일 탐색 ──────────────────────────────
            # 1순위: JSON의 File_id 필드 사용
            file_id = meta.get("File_id", "")
            audio_path = audio_index.get(file_id)

            # 2순위: JSON 파일명과 동일한 wav 탐색
            if audio_path is None:
                audio_path = audio_index.get(jf.stem + ".wav")

            # 3순위: 라벨링데이터_250331_add → 원천데이터 경로 치환
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

            samples.append({"audio": str(audio_path), "text": text.strip()})

        except Exception:
            continue

    print(f"  유효한 (오디오, 라벨) 쌍: {len(samples)}개")
    return samples


def load_audio(path: str) -> Optional[np.ndarray]:
    """오디오 파일을 16kHz mono numpy 배열로 로드합니다.
    soundfile로 빠르게 읽고 scipy로 리샘플링합니다.
    """
    try:
        import soundfile as sf
        from scipy.signal import resample_poly
        from math import gcd

        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)  # stereo → mono

        # 리샘플링 (48kHz → 16kHz 등)
        if sr != TARGET_SR:
            g = gcd(TARGET_SR, sr)
            audio = resample_poly(audio, TARGET_SR // g, sr // g)

        audio = audio.astype(np.float32)
        duration = len(audio) / TARGET_SR

        if duration < MIN_DURATION:
            return None

        # 긴 파일은 앞부분만 사용
        if duration > MAX_DURATION:
            audio = audio[:int(MAX_DURATION * TARGET_SR)]

        return audio
    except Exception:
        return None


def apply_g2p(text: str) -> str:
    """
    맞춤법 → 표준 발음 전사.
    기존 pronunciation_scorer.py 와 동일한 방식 (kss g2p).

    파인튜닝 라벨로 발음 전사를 사용하는 이유:
      모델이 음향 신호에서 '들리는 소리 그대로'를 출력하도록 학습시키기 위함.
      예) "같이" → "가치" 로 학습해야 발음 교정에 유용.
    """
    try:
        from kss import Kss
        g2p = Kss("g2p")
        return g2p(text)
    except Exception:
        return text  # G2P 실패 시 원문 그대로

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

        # ── 오디오 패딩 ──────────────────────────────────────
        input_features = [
            {"input_values": f["input_values"]} for f in features
        ]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # 모델에 필요한 키만 유지 (input_ids 등 불필요한 키 제거)
        batch = {"input_values": batch["input_values"]}

        # ── 라벨 패딩 (-100으로 마스킹) ──────────────────────
        label_list = [f["labels"] for f in features]
        max_len    = max(len(l) for l in label_list)
        labels_padded = []
        for lbl in label_list:
            pad_len = max_len - len(lbl)
            labels_padded.append(lbl + [-100] * pad_len)

        batch["labels"] = torch.tensor(labels_padded, dtype=torch.long)

        return batch

# ══════════════════════════════════════════════════════════════
# 4. 평가 지표 (CER — Character Error Rate)
# ══════════════════════════════════════════════════════════════

def build_compute_metrics(processor):
    """
    허깅페이스 Trainer에 넘길 compute_metrics 함수를 반환합니다.
    CER(Character Error Rate) 을 주 지표로 사용합니다.
    """
    from jiwer import cer as compute_cer

    def compute_metrics(pred):
        pred_logits = pred.predictions           # [B, T, V]
        pred_ids    = np.argmax(pred_logits, axis=-1)

        # -100 → 패딩 토큰 ID 로 복원 후 디코딩
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str  = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

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
    from peft import LoraConfig, get_peft_model, TaskType

    # ── 5-1. 데이터 수집 ──────────────────────────────────────
    print("\n[1/6] 데이터셋 탐색 중...")
    samples = discover_aihub_files(args.data_dir)

    if len(samples) == 0:
        print("[❌] 유효한 데이터를 찾지 못했습니다.")
        print("     --data_dir 경로와 폴더 구조를 다시 확인해 주세요.")
        raise SystemExit(1)

    # 학습/검증 분할 (9:1)
    split_idx  = int(len(samples) * 0.9)
    train_data = samples[:split_idx]
    eval_data  = samples[split_idx:]
    print(f"  학습: {len(train_data)}개 | 검증: {len(eval_data)}개")

    # ── 5-2. 모델 & 프로세서 로드 ─────────────────────────────
    print(f"\n[2/6] 베이스 모델 로드: {BASE_MODEL_ID}")
    processor = Wav2Vec2Processor.from_pretrained(BASE_MODEL_ID)
    model     = Wav2Vec2ForCTC.from_pretrained(
        BASE_MODEL_ID,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        ignore_mismatched_sizes=True,
    )

    # CTC용 feature_extractor 레이어 동결 (LoRA와 별개로 고정)
    # → feature extractor는 raw audio → feature 변환 담당
    #   이 부분까지 학습하면 너무 많은 VRAM이 필요하므로 동결
    model.freeze_feature_encoder()
    print("  feature encoder 동결 완료 (VRAM 절약)")

    # ── 5-3. LoRA 적용 ────────────────────────────────────────
    print("\n[3/6] LoRA 어댑터 설정 중...")
    lora_config = LoraConfig(
        # task_type      = TaskType.TOKEN_CLS,   # CTC는 토큰 분류와 유사
        **LORA_CONFIG,
        inference_mode = False,
    )
    model = get_peft_model(model, lora_config)

    # 학습 가능한 파라미터 수 출력
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    pct = 100 * trainable / total
    print(f"  학습 파라미터: {trainable:,} / {total:,} ({pct:.2f}%)")
    print(f"  LoRA rank={LORA_CONFIG['r']}, alpha={LORA_CONFIG['lora_alpha']}")

    # ── 5-4. 데이터셋 전처리 ──────────────────────────────────
    print("\n[4/6] 오디오 & 라벨 전처리 중... (시간이 걸릴 수 있어요)")

    class SpeechDataset(torch.utils.data.Dataset):
        """
        PyTorch Dataset — 오디오를 매번 로드 (메모리 절약)
        HuggingFace Dataset 대신 사용하는 이유:
          pyarrow int32 한계(2GB)로 인해 114초짜리 오디오 배열을
          통째로 올리면 ArrowInvalid 오류 발생 → 파일 경로만 저장 후
          __getitem__ 에서 그때그때 로드
        """
        def __init__(self, samples, processor, desc=""):
            self.valid = []
            skip_audio = 0
            skip_token = 0

            for i, sample in enumerate(samples):
                if (i + 1) % 100 == 0:
                    print(f"  [{desc}] 검증 중: {i+1}/{len(samples)}개", flush=True)

                # 라벨만 미리 토크나이징 (텍스트는 작으므로 메모리 OK)
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

            print(f"  [{desc}] 완료: {len(self.valid)}개 "
                  f"(오디오 실패: {skip_audio}개 | 토크나이징 실패: {skip_token}개)")

        def __len__(self):
            return len(self.valid)

        def __getitem__(self, idx):
            item = self.valid[idx]
            audio = load_audio(item["audio_path"])
            if audio is None:
                # 로드 실패 시 무음 대체
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

    # ── 5-5. 학습 설정 ────────────────────────────────────────
    print("\n[5/6] 학습 설정 구성 중...")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # fp16: CUDA 있을 때만 활성화 (CPU에서는 에러 발생)
    use_fp16 = torch.cuda.is_available() and TRAIN_CONFIG["fp16"]

    training_args = TrainingArguments(
        output_dir                  = str(output_dir),
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,  # 효과적 배치 크기 = batch_size * grad_accum
        learning_rate               = TRAIN_CONFIG["learning_rate"],
        warmup_ratio                = TRAIN_CONFIG["warmup_ratio"],
        weight_decay                = TRAIN_CONFIG["weight_decay"],
        fp16                        = use_fp16,
        gradient_checkpointing      = TRAIN_CONFIG["gradient_checkpointing"],
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "cer",
        greater_is_better           = False,  # CER은 낮을수록 좋음
        logging_steps               = 50,
        logging_dir                 = str(output_dir / "logs"),
        dataloader_num_workers      = 0,          # 멀티프로세싱 비활성화 (안정성)
        report_to                   = "none",     # wandb 등 외부 로깅 비활성화
        push_to_hub                 = False,
        group_by_length             = False,      # 샘플 수 적을 때 IndexError 방지
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

    # ── 5-6. 학습 시작 ────────────────────────────────────────
    print(f"\n[6/6] LoRA 파인튜닝 시작!")
    print(f"  출력 경로  : {output_dir}")
    print(f"  에폭       : {args.epochs}")
    print(f"  배치 크기  : {args.batch_size} × {args.grad_accum} (누적)")
    print(f"  FP16       : {'✅' if use_fp16 else '❌ (CUDA 없음)'}")
    print(f"  gradient_checkpointing: ✅ (VRAM 절약)\n")

    trainer.train()

    # ── LoRA 어댑터만 저장 (베이스 모델 가중치 제외) ──────────
    adapter_dir = output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))

    print(f"\n{'═'*56}")
    print(f"  ✅ 학습 완료!")
    print(f"  LoRA 어댑터 저장 위치: {adapter_dir}")
    print(f"{'═'*56}")

    # ── 사용 방법 안내 ────────────────────────────────────────
    print("""
  [추론 시 LoRA 어댑터 로드 방법]

  from peft import PeftModel
  from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

  processor  = Wav2Vec2Processor.from_pretrained("./lora_adapter")
  base_model = Wav2Vec2ForCTC.from_pretrained("w11wo/wav2vec2-xls-r-300m-korean")
  model      = PeftModel.from_pretrained(base_model, "./lora_adapter")
  model.eval()

  # pronunciation_scorer.py 의 --model 옵션으로도 사용 가능
  # (PeftModel을 직접 load_model 함수에 전달)
    """)

# ══════════════════════════════════════════════════════════════
# 6. 진입점
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="밍글리 2단계 — LoRA 파인튜닝 (wav2vec2-xls-r-300m-korean)",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="~/mingly_workspace/dataset",
        help="AIHub 구음장애 데이터셋 루트 경로\n"
             "  (기본값: ~/mingly_workspace/dataset)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="~/mingly_workspace/lora_adapter",
        help="LoRA 어댑터 저장 경로\n"
             "  (기본값: ~/mingly_workspace/lora_adapter)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="학습 에폭 수 (기본값: 5)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="배치 크기 (기본값: 4)\n"
             "  VRAM 부족 시 2로 줄이고 --grad_accum 을 늘리세요",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=4,
        help="Gradient Accumulation 스텝 수 (기본값: 4)\n"
             "  효과적 배치 크기 = batch_size × grad_accum",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="LoRA rank (기본값: 16)\n"
             "  작을수록 파라미터 적음 (8 권장: 빠른 실험 / 32: 고품질)",
    )

    args = parser.parse_args()

    # 경로 확장 (~ → 절대 경로)
    args.data_dir   = str(Path(args.data_dir).expanduser())
    args.output_dir = str(Path(args.output_dir).expanduser())

    # LoRA rank 동적 반영
    LORA_CONFIG["r"]          = args.lora_r
    LORA_CONFIG["lora_alpha"] = args.lora_r * 2

    print("=" * 56)
    print("  밍글리 (Mingly) — LoRA 파인튜닝")
    print("  PEFT / wav2vec2-xls-r-300m-korean")
    print("=" * 56)

    check_and_install()
    finetune(args)


if __name__ == "__main__":
    main()