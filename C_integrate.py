"""
STEP C: 파인튜닝 모델 → pronunciation_scorer.py 통합
======================================================
파인튜닝된 모델로 pronunciation_scorer.py를 그대로 사용할 수 있도록
--model 옵션에 로컬 경로를 지정하면 됩니다.

이 스크립트는 파인튜닝 전후 성능을 자동 비교합니다.

실행:
    python C_integrate.py \
        --finetuned_model ./finetuned_model/best \
        --test_jsonl ./finetune_dataset/test.jsonl \
        --n_samples 50
"""

import json
import argparse
import random
from pathlib import Path

import torch
import librosa
import numpy as np
from tqdm import tqdm


BASE_MODEL = "w11wo/wav2vec2-xls-r-300m-korean"
TARGET_SR  = 16000


def load_model(model_path: str):
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model     = Wav2Vec2ForCTC.from_pretrained(model_path)
    model.eval()
    return processor, model


def recognize(audio_path: str, processor, model) -> str:
    """단일 WAV → 인식 텍스트 (Greedy Decoding, 1단계와 동일)"""
    try:
        audio, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)
        audio    = audio[:TARGET_SR * 30]  # 30초 이하
    except Exception:
        return ""

    inputs = processor(audio, sampling_rate=TARGET_SR, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits

    pred_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(pred_ids)[0]


def compute_cer(reference: str, hypothesis: str) -> float:
    from jiwer import cer
    try:
        return cer(reference, hypothesis)
    except Exception:
        return 1.0


def compare_models(finetuned_path: str, test_jsonl: Path, n_samples: int, seed: int = 42):
    """파인튜닝 전후 CER 비교"""
    random.seed(seed)

    # 테스트 샘플 로드
    with open(test_jsonl, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    if len(records) > n_samples:
        records = random.sample(records, n_samples)

    print(f"\n🔬 비교 대상: {len(records)}개 샘플")

    # 모델 로드
    print(f"\n📥 베이스 모델 로딩: {BASE_MODEL}")
    base_proc, base_model = load_model(BASE_MODEL)

    print(f"📥 파인튜닝 모델 로딩: {finetuned_path}")
    ft_proc, ft_model = load_model(finetuned_path)

    from g2pk import G2p
    g2p = G2p()

    results = []
    print("\n🔄 추론 중...")

    for rec in tqdm(records):
        audio_path = rec["audio_path"]
        g2p_label  = rec["label"]

        base_recog = recognize(audio_path, base_proc, base_model)
        ft_recog   = recognize(audio_path, ft_proc,   ft_model)

        base_cer   = compute_cer(g2p_label, base_recog)
        ft_cer     = compute_cer(g2p_label, ft_recog)

        results.append({
            "label"    : g2p_label,
            "base"     : base_recog,
            "finetuned": ft_recog,
            "base_cer" : base_cer,
            "ft_cer"   : ft_cer,
            "improved" : ft_cer < base_cer,
        })

    # 집계
    avg_base_cer = np.mean([r["base_cer"] for r in results])
    avg_ft_cer   = np.mean([r["ft_cer"]   for r in results])
    improved_cnt = sum(1 for r in results if r["improved"])

    avg_base_score = max(0, 100 - avg_base_cer * 100)
    avg_ft_score   = max(0, 100 - avg_ft_cer   * 100)

    print("\n" + "=" * 60)
    print("  📊 파인튜닝 전후 성능 비교")
    print("=" * 60)
    print(f"  {'':30s}  {'베이스':>10}  {'파인튜닝':>10}")
    print(f"  {'평균 CER':30s}  {avg_base_cer:>10.4f}  {avg_ft_cer:>10.4f}")
    print(f"  {'평균 점수 (100-CER*100)':30s}  {avg_base_score:>10.1f}점  {avg_ft_score:>10.1f}점")
    print(f"  {'개선된 샘플':30s}  {improved_cnt:>10}/{len(results)}")
    diff = avg_ft_score - avg_base_score
    sign = "+" if diff >= 0 else ""
    print(f"\n  📈 점수 변화: {sign}{diff:.1f}점")

    # 개선 사례 / 악화 사례 출력
    print("\n  [개선된 샘플 TOP 5]")
    top_improved = sorted(results, key=lambda r: r["base_cer"] - r["ft_cer"], reverse=True)[:5]
    for r in top_improved:
        print(f"    정답:    {r['label'][:40]}")
        print(f"    베이스:  {r['base'][:40]}  (CER {r['base_cer']:.3f})")
        print(f"    파인튜닝:{r['finetuned'][:40]}  (CER {r['ft_cer']:.3f})")
        print()

    print("\n  [악화된 샘플 TOP 3]")
    top_worse = sorted(results, key=lambda r: r["ft_cer"] - r["base_cer"], reverse=True)[:3]
    for r in top_worse:
        if r["ft_cer"] > r["base_cer"]:
            print(f"    정답:    {r['label'][:40]}")
            print(f"    베이스:  {r['base'][:40]}  (CER {r['base_cer']:.3f})")
            print(f"    파인튜닝:{r['finetuned'][:40]}  (CER {r['ft_cer']:.3f})")
            print()

    print("=" * 60)
    print(f"\n✅ pronunciation_scorer.py에 파인튜닝 모델 적용 방법:")
    print(f"   python pronunciation_scorer.py --practice --model {finetuned_path}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetuned_model", default="./finetuned_model/best")
    parser.add_argument("--test_jsonl",      default="./finetune_dataset/test.jsonl")
    parser.add_argument("--n_samples",       type=int, default=50)
    parser.add_argument("--seed",            type=int, default=42)
    args = parser.parse_args()

    compare_models(
        finetuned_path = args.finetuned_model,
        test_jsonl     = Path(args.test_jsonl),
        n_samples      = args.n_samples,
        seed           = args.seed,
    )
