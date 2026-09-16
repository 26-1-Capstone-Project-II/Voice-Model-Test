"""
라벨↔오디오 정렬 / 모델 grounding 진단
=========================================
핵심 질문: 8에폭 학습한 모델이 "자기 train 샘플"을 맞히는가?

  - train CER 도 ~0.9 → 오디오↔라벨 매핑이 학습 불가 = 내용 불일치(정렬 깨짐)
  - train CER 낮음(<0.2) but test 높음 → 데이터 정상, 일반화/레시피 문제

원본 whisper-base(파인튜닝 X)도 같은 오디오에 돌려, 그 음성에서 표준 한국어가
무엇으로 들리는지(=transcript 와 맞는지) 교차 확인.

실행:
    CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python diagnose_alignment.py \\
        --model_path best_model_noaug/best --split train --num 20
"""

import json
import argparse
import torch
import librosa
import numpy as np
from pathlib import Path

torch.backends.cudnn.enabled = False

HOME = Path.home()
SEGMENT_DIR = HOME / "mingly_workspace" / "Voice-Model-Test" / "segmented_dataset"
DEFAULT_MODEL = HOME / "mingly_workspace" / "Voice-Model-Test" / "best_model_noaug" / "best"
TARGET_SR = 16000


def load_records(json_dir, split, num, seed=0):
    path = Path(json_dir) / f"{split}.jsonl"
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                o = json.loads(line.strip())
                if o.get("wav_path") and o.get("label") and 1.0 < o.get("duration", 0) < 20.0:
                    recs.append(o)
            except Exception:
                continue
    import random
    random.seed(seed)
    random.shuffle(recs)
    return recs[:num]


def transcribe(model, processor, audio, device):
    feat = processor.feature_extractor(
        audio, sampling_rate=TARGET_SR, return_tensors="pt"
    ).input_features.to(device)
    with torch.no_grad():
        ids = model.generate(feat, max_new_tokens=256, language="ko", task="transcribe",
                             no_repeat_ngram_size=3, repetition_penalty=1.2)
    return processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--json_dir", type=str, default=str(SEGMENT_DIR))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num", type=int, default=20)
    parser.add_argument("--with_base", action="store_true",
                        help="원본 whisper-base 도 함께 비교")
    args = parser.parse_args()

    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import evaluate
    cer_metric = evaluate.load("cer")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"📥 파인튜닝 모델: {args.model_path}")
    ft_proc = WhisperProcessor.from_pretrained(args.model_path)
    ft_model = WhisperForConditionalGeneration.from_pretrained(args.model_path).to(device).eval()

    base_proc = base_model = None
    if args.with_base:
        print("📥 원본 모델: openai/whisper-base")
        base_proc = WhisperProcessor.from_pretrained("openai/whisper-base", language="ko", task="transcribe")
        base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(device).eval()

    recs = load_records(args.json_dir, args.split, args.num)
    print(f"\n🔬 {args.split} 샘플 {len(recs)}개 — 모델이 자기 학습 데이터를 맞히는가?\n{'='*60}")

    preds, refs = [], []
    for i, r in enumerate(recs):
        try:
            audio, _ = librosa.load(r["wav_path"], sr=TARGET_SR, mono=True)
        except Exception:
            continue
        if len(audio) > 30 * TARGET_SR:
            audio = audio[:30 * TARGET_SR]

        ft_out = transcribe(ft_model, ft_proc, audio, device)
        label = r["label"].strip()
        transcript = r.get("transcript", "").strip()
        preds.append(ft_out)
        refs.append(label)

        cer = cer_metric.compute(predictions=[ft_out], references=[label])
        print(f"\n[{i+1}] CER {cer:.2f}")
        print(f"   원문(transcript): {transcript[:55]}")
        print(f"   라벨(label/G2P):  {label[:55]}")
        print(f"   파인튜닝 예측:    {ft_out[:55]}")
        if base_model is not None:
            base_out = transcribe(base_model, base_proc, audio, device)
            print(f"   원본 base 예측:   {base_out[:55]}")

    overall = cer_metric.compute(predictions=preds, references=refs)
    print(f"\n{'='*60}")
    print(f"  📊 {args.split} 전체 CER: {overall:.4f}  (샘플 {len(preds)}개)")
    if overall > 0.6:
        print("  🚨 자기 학습 데이터도 못 맞힘 → 오디오↔라벨 매핑 학습 불가")
        print("     → 세그멘테이션/라벨 정렬을 의심 (위 '원문 vs 원본 base 예측' 비교)")
    elif overall < 0.2:
        print("  🟢 train 은 잘 맞힘 → 데이터 정상, test 실패는 일반화/레시피 문제")
    else:
        print("  🟡 애매 — 부분적으로 학습됨")


if __name__ == "__main__":
    main()
