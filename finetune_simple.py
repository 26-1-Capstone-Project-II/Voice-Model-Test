"""
언어청각장애 전용 파인튜닝 — 직접 파일 읽기 버전
====================================================
ZIP 스트리밍 없이 해제된 WAV 파일을 직접 읽어 학습합니다.
transformers 5.x 완전 호환.

실행:
    python finetune_simple.py

경로 기본값:
    WAV : .../원천데이터/TS02_언어청각장애/
    JSON: .../라벨링데이터_250331_add/TL02_언어청각장애/
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


# ────────────────────────────────────────────
# 설정
# ────────────────────────────────────────────

BASE_MODEL = "w11wo/wav2vec2-xls-r-300m-korean"
TARGET_SR  = 16000
MAX_SEC    = 10.0   # 10초: CTC 안정성과 VRAM 균형

DATA_ROOT = Path(r"C:\Users\User\Voice-Model-Test\구음장애 음성인식 데이터\01.데이터\1.Training")
WAV_DIR   = DATA_ROOT / "원천데이터"  / "TS02_언어청각장애"
JSON_DIR  = DATA_ROOT / "라벨링데이터_250331_add" / "TL02_언어청각장애"


# ────────────────────────────────────────────
# 1. WAV ↔ JSON 매칭
# ────────────────────────────────────────────

def load_pairs(wav_dir: Path, json_dir: Path, g2p) -> dict:
    """
    VAD 세그멘테이션 결과(JSONL)를 split별로 직접 읽기.
    train.jsonl / validation.jsonl / test.jsonl 을 각각 로드.
    없으면 기존 JSON 방식으로 폴백.
    """
    train_jsonl = json_dir / "train.jsonl"
    val_jsonl   = json_dir / "validation.jsonl"

    if train_jsonl.exists() and val_jsonl.exists():
        print(f"  📂 JSONL 세그멘테이션 데이터 감지 → split별 직접 로드")

        split = {"train": [], "validation": [], "test": []}
        for split_name in ["train", "validation", "test"]:
            # npy 사전변환 버전 우선 사용 (속도 향상)
            npy_path = json_dir / f"{split_name}_npy.jsonl"
            path     = npy_path if npy_path.exists() else json_dir / f"{split_name}.jsonl"
            if npy_path.exists():
                print(f"  ⚡ {split_name}: npy 캐시 버전 사용")
            if not path.exists():
                continue
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if obj.get("label") and obj.get("wav_path"):
                        split[split_name].append(obj)
            print(f"  ✅ {split_name:12s}: {len(split[split_name]):,}개")
        return split

    # 기존 방식: JSON + WAV 매칭
    wav_index  = {f.stem: f for f in wav_dir.rglob("*.wav")}
    json_files = sorted(json_dir.rglob("*.json"))
    print(f"  WAV: {len(wav_index):,}개 / JSON: {len(json_files):,}개")

    pairs   = []
    skipped = Counter()
    for jp in json_files:
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            skipped["read_error"] += 1
            continue
        transcript = data.get("Transcript", "").strip()
        if not transcript:
            skipped["no_transcript"] += 1
            continue
        wav_stem = Path(data.get("File_id", "")).stem
        wav_path = wav_index.get(wav_stem)
        if wav_path is None:
            skipped["no_wav"] += 1
            continue
        try:
            label = g2p(transcript, descriptive=True).strip()
        except Exception:
            label = transcript
        patient = data.get("Patient_info", {})
        pairs.append({
            "wav_path"   : str(wav_path),
            "transcript" : transcript,
            "label"      : label,
            "speaker_id" : wav_stem,
            "sex"        : patient.get("Sex", ""),
            "age"        : patient.get("Age", ""),
        })
    print(f"  ✅ 매칭: {len(pairs):,}개")
    for k, v in skipped.most_common():
        print(f"  ⚠️  {k}: {v:,}개 스킵")
    return None  # 기존 방식은 None 반환 → split_by_speaker로 처리


def split_by_speaker(pairs: list[dict], seed: int = 42) -> dict:
    random.seed(seed)
    spk = defaultdict(list)
    for p in pairs:
        spk[p["speaker_id"]].append(p)

    speakers = list(spk.keys())
    random.shuffle(speakers)
    n = len(speakers)
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    train_sp = set(speakers[:n_train])
    val_sp   = set(speakers[n_train:n_train + n_val])

    split = {"train": [], "validation": [], "test": []}
    for sp, sp_pairs in spk.items():
        key = "train" if sp in train_sp else ("validation" if sp in val_sp else "test")
        split[key].extend(sp_pairs)

    print(f"\n✂️  화자 단위 분할 ({n}명):")
    for name, data in split.items():
        print(f"  {name:12s}: {len(data):,}개")
    return split


# ────────────────────────────────────────────
# 3. Dataset
# ────────────────────────────────────────────

class DysarthriaDataset(Dataset):
    def __init__(self, records: list[dict], processor: Wav2Vec2Processor, cache: bool = False):
        self.records      = records
        self.processor    = processor
        self._audio_cache = {}

        if cache and len(records) <= 1000:
            print(f"    오디오 사전 캐싱 ({len(records)}개)...", end=" ", flush=True)
            for i, rec in enumerate(records):
                self._audio_cache[i] = self._load_audio(rec)
            print("완료!")

    def _load_audio(self, rec: dict) -> np.ndarray:
        try:
            if rec.get("npy_path") and Path(rec["npy_path"]).exists():
                return np.load(rec["npy_path"])
            audio, _ = librosa.load(rec["wav_path"], sr=TARGET_SR, mono=True)
            return audio[:int(MAX_SEC * TARGET_SR)]
        except Exception:
            return np.zeros(int(TARGET_SR), dtype=np.float32)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        if idx in self._audio_cache:
            audio = self._audio_cache[idx]
        else:
            audio = self._load_audio(rec)

        input_values = self.processor(
            audio,
            sampling_rate  = TARGET_SR,
            return_tensors = "pt",
            padding        = False,
        ).input_values[0]

        # CTC 규칙: 입력 프레임 수 > 라벨 길이 필수
        # wav2vec2는 320샘플마다 1프레임 → 10초(160000샘플) = 약 500프레임
        # 라벨은 최대 200자로 제한 (500프레임 > 200자 보장)
        max_label_len = int((len(audio) / TARGET_SR) * 20)  # 초당 20자 기준
        label_text    = rec["label"][:max_label_len] if len(rec["label"]) > max_label_len else rec["label"]
        label_ids     = self.processor.tokenizer(label_text).input_ids

        return {
            "input_values": input_values,
            "labels"      : torch.tensor(label_ids, dtype=torch.long),
        }


# ────────────────────────────────────────────
# 4. 데이터 콜레이터
# ────────────────────────────────────────────

@dataclass
class DataCollatorCTC:
    processor: Any

    def __call__(self, features: list[dict]) -> dict:
        # input_values 패딩
        input_features = [{"input_values": f["input_values"]} for f in features]
        batch = self.processor.pad(
            input_features, padding=True, return_tensors="pt"
        )

        # labels 패딩 — tokenizer로 직접
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(
            label_features, padding=True, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


# ────────────────────────────────────────────
# 5. 평가 지표
# ────────────────────────────────────────────

def make_compute_metrics(processor: Wav2Vec2Processor):
    cer_metric = evaluate.load("cer")

    def compute_metrics(pred):
        pred_ids  = np.argmax(pred.predictions, axis=-1)
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)

        # 샘플 3개 출력
        print()
        for i in range(min(3, len(pred_str))):
            print(f"  정답: {label_str[i][:60]}")
            print(f"  예측: {pred_str[i][:60]}")
            print()

        cer = cer_metric.compute(predictions=pred_str, references=label_str)
        return {"cer": round(cer, 4)}

    return compute_metrics


# ────────────────────────────────────────────
# 6. 학습
# ────────────────────────────────────────────

def train(
    wav_dir    : Path,
    json_dir   : Path,
    output_dir : Path,
    batch_size : int   = 4,
    num_epochs : int   = 5,
    lr         : float = 3e-5,
    grad_accum : int   = 8,
    max_steps  : int   = -1,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"💻 디바이스: {device}")
    if device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"   GPU: {torch.cuda.get_device_name(0)} ({vram:.1f}GB)")

    # G2P
    print("\n🔤 G2P 로딩...")
    g2p = load_g2p()

    # 데이터 로딩
    print(f"\n🔗 데이터 매칭 중...")
    result = load_pairs(wav_dir, json_dir, g2p)
    if result is None:
        print("❌ 매칭 실패. WAV 압축 해제 완료 여부 확인하세요.")
        return

    # JSONL 방식은 dict, 기존 방식은 None(위에서 처리)
    if isinstance(result, dict):
        split = result
    else:
        split = split_by_speaker(result)

    # 모델 & 프로세서
    print(f"\n📥 모델 로딩: {BASE_MODEL}")
    processor = Wav2Vec2Processor.from_pretrained(BASE_MODEL)
    model     = Wav2Vec2ForCTC.from_pretrained(
        BASE_MODEL,
        ctc_loss_reduction = "mean",
        pad_token_id       = processor.tokenizer.pad_token_id,
    )

    # Feature Extractor 동결 (버전별 메서드명 대응)
    if hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()
    else:
        model.freeze_feature_extractor()


    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔓 학습 파라미터: {trainable:,}")

    # 데이터셋
    # validation 전체(11,223개) 사용 시 평가 OOM → 500개로 제한
    import random
    val_records = split["validation"]
    if len(val_records) > 500:
        random.seed(42)
        val_records = random.sample(val_records, 500)
        print(f"  validation 샘플 제한: {len(split['validation']):,} → 500개 (OOM 방지)")

    train_ds = DysarthriaDataset(split["train"], processor)
    val_ds   = DysarthriaDataset(val_records,    processor, cache=True)  # 500개 사전 캐싱

    # 학습 인자 (transformers 5.x 호환)
    training_args = TrainingArguments(
        output_dir                  = str(output_dir),
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        gradient_accumulation_steps = grad_accum,
        num_train_epochs            = num_epochs,
        max_steps                   = max_steps if max_steps > 0 else -1,
        learning_rate               = lr,
        weight_decay                = 0.005,
        fp16                        = False,  # wav2vec2+fp16 → NaN 발생, 비활성화
        eval_strategy               = "steps",
        eval_steps                  = 500,     # epoch 대신 500스텝마다 평가
        save_strategy               = "steps",
        save_steps                  = 500,
        logging_steps               = 50,
        load_best_model_at_end      = True,
        metric_for_best_model       = "cer",
        greater_is_better           = False,
        save_total_limit            = 2,
        report_to                   = "none",
        dataloader_num_workers      = 0,
        eval_accumulation_steps     = 10,      # 평가 시 10배치씩 누적 (OOM 방지)
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

    print("\n🚀 파인튜닝 시작!")
    trainer.train()

    # 테스트 평가 (500개 샘플로 제한, OOM 방지)
    print("\n📊 Test 평가 (샘플 500개):")
    import random as _random
    test_records = split["test"]
    if len(test_records) > 500:
        _random.seed(42)
        test_records = _random.sample(test_records, 500)
    test_ds = DysarthriaDataset(test_records, processor)
    results = trainer.evaluate(test_ds)
    print(f"  CER: {results.get('eval_cer', 'N/A')}")

    # 저장
    best_path = output_dir / "best"
    trainer.save_model(str(best_path))
    processor.save_pretrained(str(best_path))
    print(f"\n💾 모델 저장: {best_path}")
    print(f"\n✅ pronunciation_scorer.py 적용:")
    print(f"   python pronunciation_scorer.py --practice --model {best_path}")


# ────────────────────────────────────────────
# 7. 메인
# ────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_dir",    default=str(WAV_DIR))
    parser.add_argument("--json_dir",   default=str(JSON_DIR))
    parser.add_argument("--output_dir", default="./finetuned_model")
    parser.add_argument("--batch_size", type=int,   default=4)
    parser.add_argument("--num_epochs", type=int,   default=30)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--grad_accum", type=int,   default=8)
    parser.add_argument("--max_steps",  type=int,   default=-1, help="테스트용: -1이면 전체")
    args = parser.parse_args()

    train(
        wav_dir    = Path(args.wav_dir),
        json_dir   = Path(args.json_dir),
        output_dir = Path(args.output_dir),
        batch_size = args.batch_size,
        num_epochs = args.num_epochs,
        lr         = args.lr,
        grad_accum = args.grad_accum,
        max_steps  = args.max_steps,
    )
