"""
Whisper 모델 실제 데이터 테스트
================================
segmented_dataset의 test split(13,629개)에서 실제 구음장애 음성으로 모델 성능 검증.
gTTS(합성 음성)이 아닌 실제 음성으로 모델의 발음 전사 능력 확인.

실행:
    CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python test_whisper_real.py \\
        --model_path best_model_whisper/best --num_samples 100
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
DEFAULT_MODEL = HOME / "mingly_workspace" / "Voice-Model-Test" / "best_model_whisper" / "best"
TARGET_SR = 16000


def load_test_data(json_dir, num_samples=100):
    """test.jsonl에서 샘플 로드."""
    path = Path(json_dir) / "test.jsonl"
    if not path.exists():
        print(f"❌ {path} 없음")
        return []

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                duration = obj.get("duration", 0)
                if 1.0 < duration < 20.0:  # 적절한 길이만
                    label = obj.get("label", "").strip()
                    transcript = obj.get("transcript", "").strip()
                    wav_path = obj.get("wav_path", "")
                    if label and transcript and wav_path:
                        records.append(obj)
            except:
                continue

    # 셔플 후 샘플링
    import random
    random.seed(42)
    random.shuffle(records)
    records = records[:num_samples]

    print(f"  📂 test 데이터: {len(records)}개 로드 (전체 중 샘플링)")
    return records


def test_real_data(model_path, json_dir, num_samples):
    """실제 구음장애 음성으로 모델 테스트."""
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import evaluate

    cer_metric = evaluate.load("cer")

    # 모델 로드
    print(f"📥 모델 로드: {model_path}")
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    # 데이터 로드
    records = load_test_data(json_dir, num_samples)
    if not records:
        return

    print(f"\n{'='*60}")
    print(f"  🔬 실제 구음장애 음성 테스트 ({len(records)}개)")
    print(f"{'='*60}")

    predictions = []
    references = []
    detailed_results = []

    for i, rec in enumerate(records):
        wav_path = rec["wav_path"]
        label = rec["label"]        # G2P 발음 전사 (정답)
        transcript = rec["transcript"]  # 원문

        try:
            audio, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
        except Exception as e:
            continue

        # 30초 자르기
        max_samples = 30 * TARGET_SR
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        # 전사
        input_features = processor.feature_extractor(
            audio, sampling_rate=TARGET_SR, return_tensors="pt"
        ).input_features.to(device)

        with torch.no_grad():
            pred_ids = model.generate(
                input_features,
                max_new_tokens=256,
                language="ko",
                task="transcribe",
                no_repeat_ngram_size=3,
                repetition_penalty=1.2,
            )

        pred_text = processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )[0].strip()

        predictions.append(pred_text)
        references.append(label)

        # 상세 결과 (처음 20개만 출력)
        if i < 20:
            # 간단한 매칭 확인
            match = "✅" if pred_text == label else "❌"
            # 부분 매칭 확인
            if pred_text != label and len(pred_text) > 3 and len(label) > 3:
                common = sum(1 for a, b in zip(pred_text, label) if a == b)
                match_rate = common / max(len(pred_text), len(label))
                if match_rate > 0.5:
                    match = "🟡"

            print(f"\n  [{i+1:3d}] {match}")
            print(f"      원문:   {transcript[:50]}")
            print(f"      정답:   {label[:50]}")
            print(f"      예측:   {pred_text[:50]}")

        detailed_results.append({
            "transcript": transcript,
            "label": label,
            "prediction": pred_text,
        })

    # CER 계산
    if predictions and references:
        # 빈 문자열 필터
        valid = [(p, r) for p, r in zip(predictions, references) if p and r]
        if valid:
            preds, refs = zip(*valid)
            overall_cer = cer_metric.compute(predictions=list(preds), references=list(refs))

            # 개별 CER 분포
            individual_cers = []
            for p, r in zip(preds, refs):
                try:
                    c = cer_metric.compute(predictions=[p], references=[r])
                    individual_cers.append(c)
                except:
                    individual_cers.append(1.0)

            # 통계
            good = sum(1 for c in individual_cers if c < 0.3)
            decent = sum(1 for c in individual_cers if 0.3 <= c < 0.6)
            poor = sum(1 for c in individual_cers if c >= 0.6)

            print(f"\n{'='*60}")
            print(f"  📊 실제 데이터 테스트 결과")
            print(f"{'='*60}")
            print(f"  전체 CER:   {overall_cer:.4f}")
            print(f"  평균 CER:   {np.mean(individual_cers):.4f}")
            print(f"  중간값 CER: {np.median(individual_cers):.4f}")
            print(f"")
            print(f"  📊 CER 분포:")
            print(f"    ✅ 우수 (CER < 0.3): {good}/{len(individual_cers)} ({100*good/len(individual_cers):.0f}%)")
            print(f"    🟡 보통 (0.3-0.6):   {decent}/{len(individual_cers)} ({100*decent/len(individual_cers):.0f}%)")
            print(f"    ❌ 미흡 (CER > 0.6): {poor}/{len(individual_cers)} ({100*poor/len(individual_cers):.0f}%)")

            # 최고 / 최저 5개
            sorted_results = sorted(zip(individual_cers, detailed_results), key=lambda x: x[0])

            print(f"\n  🏆 최고 5개:")
            for cer_val, res in sorted_results[:5]:
                print(f"    CER {cer_val:.3f} | 정답: {res['label'][:30]} | 예측: {res['prediction'][:30]}")

            print(f"\n  ⚠️ 최저 5개:")
            for cer_val, res in sorted_results[-5:]:
                print(f"    CER {cer_val:.3f} | 정답: {res['label'][:30]} | 예측: {res['prediction'][:30]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper 실제 데이터 테스트")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--json_dir", type=str, default=str(SEGMENT_DIR))
    parser.add_argument("--num_samples", type=int, default=100,
                        help="테스트 샘플 수")
    args = parser.parse_args()

    test_real_data(args.model_path, args.json_dir, args.num_samples)
