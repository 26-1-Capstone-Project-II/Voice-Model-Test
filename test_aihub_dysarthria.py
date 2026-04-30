"""
AIHub 구음장애 데이터셋 — 파인튜닝 모델 효과 검증
====================================================
목적
----
청각장애인이 본 앱을 실제로 사용했을 때의 효과를 **간접적으로** 증명한다.
청각장애인 발화 데이터의 직접 확보가 어려우므로, 음향적 특성이 유사한
**구음장애(Dysarthria) 음성**(AIHub)을 대리 지표(Proxy)로 활용하여
파인튜닝 모델이 "맞춤법 보정 없이 들리는 그대로 출력"하는지를 검증한다.

핵심 비교
---------
A) 원본 Whisper-tiny 출력  vs  화자 의도 텍스트(transcript)
   → 자동 교정으로 인해 의도와 유사하게 출력될수록 CER이 낮음
B) 파인튜닝 모델 출력       vs  g2pk(transcript) (기대 발음)
   → 화자가 정확히 발음한 경우에만 CER이 낮음
   → 화자가 틀리게 발음하면 "들린 그대로" Raw 출력 → 오류 감지 가능

(A) - (B) 갭이 작거나, (A) << (B)이면 파인튜닝이 의도대로 학습됐다는 증거.
또한 자모 레벨 정렬(align_jamo)로 어떤 음소가 어떻게 어긋났는지 추출.

실행 (Linux 서버)
-----------------
    CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python test_aihub_dysarthria.py \\
        --model_path best_model_whisper/best \\
        --baseline_model openai/whisper-tiny \\
        --json_dir segmented_dataset \\
        --num_samples 200 \\
        --output_dir results/aihub_eval

사전 준비
---------
1) AIHub 구음장애 데이터 다운로드:
   https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=608
2) vad_segment.py로 세그멘테이션 → segmented_dataset/test.jsonl 생성
"""

import os
import json
import argparse
import random
from pathlib import Path
from collections import Counter, defaultdict

import torch
import librosa
import numpy as np
from tqdm import tqdm

torch.backends.cudnn.enabled = False

TARGET_SR = 16000
HOME = Path.home()


# ────────────────────────────────────────────
# 1. 데이터 로딩
# ────────────────────────────────────────────
def load_test_data(json_dir, num_samples=100, min_dur=1.0, max_dur=20.0, seed=42):
    """segmented_dataset/test.jsonl에서 평가 샘플 로드.

    JSONL 필드:
      - wav_path:   세그멘트 WAV 경로
      - transcript: 화자 의도 텍스트 (맞춤법) ← 베이스라인 비교 기준
      - label:      g2pk(transcript) (기대 발음) ← 파인튜닝 비교 기준
      - duration:   세그멘트 길이 (초)
      - speaker_id, disease_type, sex, age (메타데이터)
    """
    path = Path(json_dir) / "test.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 없음. vad_segment.py로 AIHub 데이터 세그멘테이션을 먼저 실행하세요."
        )

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            duration = obj.get("duration", 0)
            if not (min_dur < duration < max_dur):
                continue
            transcript = (obj.get("transcript") or "").strip()
            label = (obj.get("label") or "").strip()
            wav_path = obj.get("wav_path", "")
            if transcript and label and wav_path and Path(wav_path).exists():
                records.append(obj)

    random.seed(seed)
    random.shuffle(records)
    if num_samples > 0:
        records = records[:num_samples]

    print(f"  📂 AIHub test 데이터: {len(records)}개 로드")
    return records


# ────────────────────────────────────────────
# 2. 모델 로더
# ────────────────────────────────────────────
def load_whisper(model_path, device):
    """Whisper 모델 + processor 로드."""
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()
    return processor, model


def transcribe(audio, processor, model, device):
    """단일 오디오 → 전사 텍스트."""
    max_samples = 30 * TARGET_SR
    if len(audio) > max_samples:
        audio = audio[:max_samples]

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
    return processor.tokenizer.batch_decode(
        pred_ids, skip_special_tokens=True
    )[0].strip()


# ────────────────────────────────────────────
# 3. 평가
# ────────────────────────────────────────────
def evaluate_pair(records, baseline_pack, finetuned_pack, device):
    """베이스라인 vs 파인튜닝 동시 평가.

    반환:
      list of dict:
        - transcript:        화자 의도 텍스트
        - expected_pron:     g2pk 기대 발음 (label)
        - baseline_output:   원본 Whisper 출력
        - finetuned_output:  파인튜닝 출력
        - cer_baseline_vs_intent:  베이스라인이 의도 텍스트로 얼마나 교정했는가
        - cer_finetuned_vs_pron:   파인튜닝이 기대 발음과 얼마나 일치하는가
        - is_corrected_by_baseline: 베이스라인 출력이 transcript와 거의 같음 (교정 발생)
        - is_raw_by_finetuned:      파인튜닝 출력이 transcript와 다름 (교정 거부)
        - jamo_errors:       자모 레벨 오류 (파인튜닝 vs 기대 발음)
    """
    import evaluate
    from pronunciation_evaluator import text_to_jamo, align_jamo, extract_errors

    cer_metric = evaluate.load("cer")
    base_proc, base_model = baseline_pack
    ft_proc, ft_model = finetuned_pack

    results = []
    for rec in tqdm(records, desc="추론 (baseline+finetuned)"):
        wav_path = rec["wav_path"]
        transcript = rec["transcript"]
        expected_pron = rec["label"]

        try:
            audio, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
        except Exception:
            continue
        if len(audio) < TARGET_SR * 0.3:
            continue

        baseline_out = transcribe(audio, base_proc, base_model, device)
        finetuned_out = transcribe(audio, ft_proc, ft_model, device)

        try:
            cer_a = cer_metric.compute(
                predictions=[baseline_out or " "], references=[transcript]
            )
        except Exception:
            cer_a = 1.0
        try:
            cer_b = cer_metric.compute(
                predictions=[finetuned_out or " "], references=[expected_pron]
            )
        except Exception:
            cer_b = 1.0

        # "교정 발생" 판단: 베이스라인 출력이 의도 텍스트와 매우 가까움 (CER < 0.15)
        is_corrected_by_baseline = cer_a < 0.15
        # "교정 거부" 판단: 파인튜닝 출력이 의도 텍스트와 다름
        is_raw_by_finetuned = (finetuned_out.strip() != transcript.strip())

        # 자모 레벨 오류 추출 (파인튜닝 출력 vs 기대 발음)
        exp_jamo = text_to_jamo(expected_pron)
        act_jamo = text_to_jamo(finetuned_out)
        alignment = align_jamo(exp_jamo, act_jamo)
        jamo_errors = extract_errors(alignment)

        results.append({
            "wav_path": wav_path,
            "speaker_id": rec.get("speaker_id", ""),
            "disease_type": rec.get("disease_type", ""),
            "transcript": transcript,
            "expected_pron": expected_pron,
            "baseline_output": baseline_out,
            "finetuned_output": finetuned_out,
            "cer_baseline_vs_intent": round(float(cer_a), 4),
            "cer_finetuned_vs_pron": round(float(cer_b), 4),
            "is_corrected_by_baseline": bool(is_corrected_by_baseline),
            "is_raw_by_finetuned": bool(is_raw_by_finetuned),
            "jamo_error_count": len(jamo_errors),
            "jamo_errors": [
                {"expected": e["expected"], "actual": e["actual"], "status": e["status"]}
                for e in jamo_errors
            ],
        })

    return results


# ────────────────────────────────────────────
# 4. 통계 집계
# ────────────────────────────────────────────
def summarize(results):
    n = len(results)
    if n == 0:
        return {}

    cer_a = [r["cer_baseline_vs_intent"] for r in results]
    cer_b = [r["cer_finetuned_vs_pron"] for r in results]

    n_corrected = sum(1 for r in results if r["is_corrected_by_baseline"])
    n_raw = sum(1 for r in results if r["is_raw_by_finetuned"])

    # 자모 오류 패턴 빈도 (substitution만)
    sub_counter = Counter()
    for r in results:
        for e in r["jamo_errors"]:
            if e["status"] == "substitution" and e["expected"] and e["actual"]:
                sub_counter[(e["expected"], e["actual"])] += 1

    summary = {
        "n_samples": n,
        "cer_baseline_vs_intent": {
            "mean": round(float(np.mean(cer_a)), 4),
            "median": round(float(np.median(cer_a)), 4),
            "std": round(float(np.std(cer_a)), 4),
        },
        "cer_finetuned_vs_pron": {
            "mean": round(float(np.mean(cer_b)), 4),
            "median": round(float(np.median(cer_b)), 4),
            "std": round(float(np.std(cer_b)), 4),
        },
        "auto_correction_rate_baseline": round(n_corrected / n, 4),
        "non_correction_rate_finetuned": round(n_raw / n, 4),
        "top_jamo_substitutions": [
            {"expected": exp, "actual": act, "count": cnt}
            for (exp, act), cnt in sub_counter.most_common(15)
        ],
    }
    return summary


def print_summary(summary, results, sample_n=20):
    print(f"\n{'='*70}")
    print(f"  📊 AIHub 구음장애 검증 결과 요약")
    print(f"{'='*70}")
    print(f"  샘플 수: {summary['n_samples']}")
    print()
    print(f"  [A] 원본 Whisper-tiny  vs  화자 의도 텍스트")
    print(f"      평균 CER: {summary['cer_baseline_vs_intent']['mean']:.4f}  "
          f"중간값: {summary['cer_baseline_vs_intent']['median']:.4f}")
    print(f"      → 베이스라인 자동 교정율: "
          f"{summary['auto_correction_rate_baseline']:.1%} "
          f"(CER<0.15로 의도와 거의 일치 = 교정 발생)")
    print()
    print(f"  [B] 파인튜닝 모델       vs  g2pk 기대 발음")
    print(f"      평균 CER: {summary['cer_finetuned_vs_pron']['mean']:.4f}  "
          f"중간값: {summary['cer_finetuned_vs_pron']['median']:.4f}")
    print(f"      → 파인튜닝 교정 거부율 (Raw 출력율): "
          f"{summary['non_correction_rate_finetuned']:.1%}")
    print()
    print(f"  💡 해석:")
    print(f"     - 베이스라인 자동 교정율이 높을수록 → 기존 ASR은 발음 오류를 숨김")
    print(f"     - 파인튜닝 교정 거부율이 높을수록 → 우리 모델은 들린 그대로 출력")
    print(f"     - 두 비율의 차이가 클수록 → 파인튜닝의 효과가 큼 (앱에서 발음 오류 감지 가능)")

    if summary["top_jamo_substitutions"]:
        print(f"\n  🔍 자주 감지된 자모 치환 오류 TOP 10:")
        for item in summary["top_jamo_substitutions"][:10]:
            print(f"      {item['expected']} → {item['actual']}  "
                  f"({item['count']}회)")

    print(f"\n  📋 샘플 비교 (앞 {sample_n}개):")
    print(f"  {'-'*70}")
    for i, r in enumerate(results[:sample_n]):
        flag_a = "✅교정됨" if r["is_corrected_by_baseline"] else "❌교정안됨"
        flag_b = "✅Raw출력" if r["is_raw_by_finetuned"] else "❌의도와동일"
        print(f"\n  [{i+1:3d}]")
        print(f"    화자 의도(transcript): {r['transcript']}")
        print(f"    기대 발음(g2pk):       {r['expected_pron']}")
        print(f"    원본 Whisper [{flag_a}]:   {r['baseline_output']}  "
              f"(CER={r['cer_baseline_vs_intent']:.3f})")
        print(f"    파인튜닝   [{flag_b}]:   {r['finetuned_output']}  "
              f"(CER={r['cer_finetuned_vs_pron']:.3f})")
        if r["jamo_errors"]:
            top_errs = r["jamo_errors"][:5]
            errs_str = ", ".join(
                f"{e['expected'] or '∅'}→{e['actual'] or '∅'}" for e in top_errs
            )
            print(f"    자모 오류 (파인튜닝 vs 기대): {errs_str}")


# ────────────────────────────────────────────
# 5. 결과 저장
# ────────────────────────────────────────────
def save_results(results, summary, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 전체 결과 JSON
    with open(out / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results},
                  f, ensure_ascii=False, indent=2)

    # 요약 마크다운 리포트
    md = []
    md.append("# AIHub 구음장애 데이터셋 — 파인튜닝 모델 효과 검증\n")
    md.append(f"- 평가 샘플 수: **{summary['n_samples']}**\n")
    md.append("\n## 1. 핵심 지표\n")
    md.append("| 항목 | 값 |\n|------|----|")
    md.append(f"| (A) 원본 Whisper CER (vs 의도 텍스트) — 평균 | "
              f"{summary['cer_baseline_vs_intent']['mean']:.4f} |")
    md.append(f"| (B) 파인튜닝 CER (vs 기대 발음) — 평균 | "
              f"{summary['cer_finetuned_vs_pron']['mean']:.4f} |")
    md.append(f"| **베이스라인 자동 교정율** (CER<0.15) | "
              f"**{summary['auto_correction_rate_baseline']:.1%}** |")
    md.append(f"| **파인튜닝 교정 거부율 (Raw 출력)** | "
              f"**{summary['non_correction_rate_finetuned']:.1%}** |")

    md.append("\n## 2. 자주 감지된 자모 치환 오류 (TOP 15)\n")
    md.append("| 순위 | 기대 자모 | 실제 자모 | 횟수 |\n|------|----------|----------|-----|")
    for i, item in enumerate(summary["top_jamo_substitutions"], 1):
        md.append(f"| {i} | {item['expected']} | {item['actual']} | "
                  f"{item['count']} |")

    md.append("\n## 3. 해석\n")
    md.append(
        "- 베이스라인(원본 Whisper)은 **자동 교정**으로 인해 화자가 틀리게 "
        "발음해도 의도 텍스트로 출력 → **앱에서 발음 오류 감지 불가**\n"
        "- 파인튜닝 모델은 **들린 그대로 Raw 출력** → 화자의 실제 발음 음소를 "
        "포착 → **자모 레벨 오류 피드백 제공 가능**\n"
        "- 청각장애인 사용 시에도 동일한 메커니즘으로 **발음 교정 피드백**을 "
        "제공할 수 있음을 간접 증명"
    )

    with open(out / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\n  💾 결과 저장:")
    print(f"     {out / 'eval_results.json'}")
    print(f"     {out / 'summary.md'}")


# ────────────────────────────────────────────
# 6. 메인
# ────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="AIHub 구음장애 데이터로 파인튜닝 모델의 발음 전사 효과 검증"
    )
    parser.add_argument("--model_path", type=str,
                        default=str(HOME / "mingly_workspace" / "Voice-Model-Test"
                                    / "best_model_whisper" / "best"),
                        help="파인튜닝된 Whisper 모델 경로")
    parser.add_argument("--baseline_model", type=str, default="openai/whisper-tiny",
                        help="베이스라인 (원본) Whisper 모델 경로/이름")
    parser.add_argument("--json_dir", type=str,
                        default=str(HOME / "mingly_workspace" / "Voice-Model-Test"
                                    / "segmented_dataset"),
                        help="VAD 처리된 AIHub segmented_dataset 디렉토리")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="평가 샘플 수 (0=전체)")
    parser.add_argument("--output_dir", type=str, default="results/aihub_eval",
                        help="결과 저장 경로")
    parser.add_argument("--print_samples", type=int, default=20,
                        help="콘솔에 출력할 샘플 비교 수")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")

    # 1) 데이터
    records = load_test_data(args.json_dir, num_samples=args.num_samples)
    if not records:
        print("❌ 평가 가능한 샘플이 없습니다.")
        return

    # 2) 모델 로드
    print(f"\n📥 베이스라인 로드: {args.baseline_model}")
    base_pack = load_whisper(args.baseline_model, device)
    print(f"📥 파인튜닝 로드: {args.model_path}")
    ft_pack = load_whisper(args.model_path, device)

    # 3) 평가
    print(f"\n{'='*70}")
    print(f"  🔬 AIHub 구음장애 검증 시작 ({len(records)}개)")
    print(f"{'='*70}")
    results = evaluate_pair(records, base_pack, ft_pack, device)

    # 4) 집계 및 출력
    summary = summarize(results)
    print_summary(summary, results, sample_n=args.print_samples)

    # 5) 저장
    save_results(results, summary, args.output_dir)

    print(f"\n{'='*70}")
    print(f"  ✅ AIHub 구음장애 검증 완료")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()