"""
AIHub 구음장애 — Baseline-Referenced 검증
==========================================
정렬 문제 우회: transcript 라벨 대신 **베이스라인 Whisper의 출력**을
"표준 표기" 기준점으로 삼아 파인튜닝의 phonetic-transcription 능력을 검증.

핵심 논리
---------
같은 오디오에 두 모델을 돌리면:
  - 베이스라인 Whisper:  X (표준 한국어 표기, 자동 교정됨)
  - 파인튜닝 Whisper:    Y (들리는 그대로 = phonetic)

파인튜닝이 의도대로 학습됐다면 Y ≈ g2pk(X). 즉, 두 모델 출력이 자동으로
"표기 ↔ 발음" 관계를 형성해야 한다. 이는 라벨 정확도와 무관하게 측정 가능.

측정 지표
---------
1. Phonetic Match CER     = CER(Y, g2pk(X))           ← 핵심
2. Standard Divergence    = (Y != X) 비율             ← Raw 출력율
3. Phonetic Hit Rate      = X→Y 차이 중 한국어 음운변동 패턴 적중률
4. 데모 사례               = X / g2pk(X) / Y 3-way 비교

실행 (Linux 서버)
-----------------
    CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python test_aihub_baseline_ref.py \\
        --model_path best_model_whisper/best \\
        --baseline_model openai/whisper-tiny \\
        --json_dir segmented_dataset \\
        --num_samples 200 \\
        --min_dur 1.0 --max_dur 10.0 \\
        --output_dir results/aihub_baseline_ref
"""

import os
import json
import argparse
import random
from pathlib import Path
from collections import Counter

import torch
import librosa
import numpy as np
from tqdm import tqdm

torch.backends.cudnn.enabled = False

TARGET_SR = 16000
HOME = Path.home()


# ────────────────────────────────────────────
# 1. 데이터 로딩 (transcript 라벨 미사용)
# ────────────────────────────────────────────
def load_audio_samples(json_dir, num_samples=200, min_dur=1.0, max_dur=10.0,
                      seed=42):
    """segmented_dataset/test.jsonl에서 wav_path만 사용 (라벨 무시).

    짧은 세그먼트만 추출하여 단일 발화일 가능성을 높임.
    """
    path = Path(json_dir) / "test.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음")

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
            wav_path = obj.get("wav_path", "")
            if wav_path and Path(wav_path).exists():
                records.append({
                    "wav_path": wav_path,
                    "duration": duration,
                    "speaker_id": obj.get("speaker_id", ""),
                    # 참고용 (평가에는 미사용)
                    "ref_transcript": obj.get("transcript", ""),
                })

    random.seed(seed)
    random.shuffle(records)
    if num_samples > 0:
        records = records[:num_samples]

    print(f"  📂 평가 샘플: {len(records)}개  "
          f"(duration {min_dur}~{max_dur}s 필터)")
    return records


# ────────────────────────────────────────────
# 2. 모델
# ────────────────────────────────────────────
def load_whisper(model_path, device):
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.to(device).eval()
    return processor, model


def transcribe(audio, processor, model, device):
    max_samples = 30 * TARGET_SR
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    feats = processor.feature_extractor(
        audio, sampling_rate=TARGET_SR, return_tensors="pt"
    ).input_features.to(device)
    with torch.no_grad():
        ids = model.generate(
            feats, max_new_tokens=256, language="ko", task="transcribe",
            no_repeat_ngram_size=3, repetition_penalty=1.2,
        )
    return processor.tokenizer.batch_decode(ids, skip_special_tokens=True)[0].strip()


# ────────────────────────────────────────────
# 3. 한국어 음운 변동 감지
# ────────────────────────────────────────────
def detect_phonetic_transforms(standard, phonetic):
    """X(표준) vs Y(발음)의 음절-단위 차이를 한국어 음운변동 패턴으로 분류.

    카테고리:
      - tensification (경음화)   : ㄱ→ㄲ, ㄷ→ㄸ, ㅂ→ㅃ, ㅅ→ㅆ, ㅈ→ㅉ
      - nasalization (비음화)    : 종성 ㄱ/ㄷ/ㅂ + 초성 ㄴ/ㅁ → ㅇ/ㄴ/ㅁ + ㄴ/ㅁ
      - palatalization (구개음화) : ㄷ/ㅌ + ㅣ → ㅈ/ㅊ + ㅣ
      - linking (연음화)         : 종성 자음이 다음 음절 초성으로 이동
      - other                    : 그 외 차이
    """
    from pronunciation_evaluator import text_to_jamo, align_jamo

    exp_jamo = text_to_jamo(standard)
    act_jamo = text_to_jamo(phonetic)
    alignment = align_jamo(exp_jamo, act_jamo)

    # 치환 쌍만 추출 (insertion/deletion은 별도)
    subs = [(a["expected"], a["actual"]) for a in alignment
            if a["status"] == "substitution"
            and a["expected"] and a["actual"]]

    cats = Counter()
    tensification_pairs = {"ㄱ": "ㄲ", "ㄷ": "ㄸ", "ㅂ": "ㅃ",
                           "ㅅ": "ㅆ", "ㅈ": "ㅉ"}
    palatalization_pairs = {"ㄷ": "ㅈ", "ㅌ": "ㅊ"}
    nasalization_pairs = {"ㄱ": "ㅇ", "ㄷ": "ㄴ", "ㅂ": "ㅁ"}

    for exp, act in subs:
        if exp in tensification_pairs and act == tensification_pairs[exp]:
            cats["tensification"] += 1
        elif exp in palatalization_pairs and act == palatalization_pairs[exp]:
            cats["palatalization"] += 1
        elif exp in nasalization_pairs and act == nasalization_pairs[exp]:
            cats["nasalization"] += 1
        else:
            cats["other"] += 1

    # insertion/deletion은 종성 이동(연음화) 후보로 카운트
    n_insert = sum(1 for a in alignment if a["status"] == "insertion")
    n_delete = sum(1 for a in alignment if a["status"] == "deletion")
    if n_insert and n_delete:
        cats["linking_candidate"] = min(n_insert, n_delete)

    return cats, subs


# ────────────────────────────────────────────
# 4. 평가
# ────────────────────────────────────────────
def evaluate(records, baseline_pack, finetuned_pack, g2p, device):
    """샘플별로 두 모델 추론 → baseline-referenced 지표 계산."""
    import evaluate as hf_eval
    cer_metric = hf_eval.load("cer")

    base_proc, base_model = baseline_pack
    ft_proc, ft_model = finetuned_pack

    results = []
    for rec in tqdm(records, desc="추론 (baseline+finetuned)"):
        try:
            audio, _ = librosa.load(rec["wav_path"], sr=TARGET_SR, mono=True)
        except Exception:
            continue
        if len(audio) < TARGET_SR * 0.3:
            continue

        X = transcribe(audio, base_proc, base_model, device)
        Y = transcribe(audio, ft_proc, ft_model, device)

        # 베이스라인 출력이 비어있거나 비한국어인 경우 스킵
        if not X or not any('가' <= ch <= '힣' for ch in X):
            continue

        # g2pk(베이스라인) = 기대 발음
        try:
            expected_phonetic = g2p(X, descriptive=True).strip()
        except Exception:
            continue

        # 핵심 지표 1: 파인튜닝 vs g2pk(베이스라인)
        try:
            cer_main = cer_metric.compute(
                predictions=[Y or " "], references=[expected_phonetic]
            )
        except Exception:
            cer_main = 1.0

        # 보조 지표: 파인튜닝 vs 베이스라인 (Raw 출력 정도)
        try:
            cer_vs_baseline = cer_metric.compute(
                predictions=[Y or " "], references=[X]
            )
        except Exception:
            cer_vs_baseline = 1.0

        # 음운 변동 감지 (X→Y 변환)
        # 의미 있는 변환 기회: g2pk(X) != X 인 경우
        transform_opportunity = (expected_phonetic != X)
        cats, subs = detect_phonetic_transforms(X, Y)

        # Y가 g2pk(X)와 일치 → 음운변동 적용 성공
        applied_phonetic = (Y.strip() == expected_phonetic.strip())

        # Y가 X와 다름 → Raw 출력 (자동 교정 거부)
        is_raw_output = (Y.strip() != X.strip())

        results.append({
            "wav_path": rec["wav_path"],
            "speaker_id": rec["speaker_id"],
            "duration": rec["duration"],
            "baseline_X": X,
            "expected_phonetic": expected_phonetic,
            "finetuned_Y": Y,
            "ref_transcript_for_log": rec["ref_transcript"],
            "cer_finetuned_vs_phonetic": round(float(cer_main), 4),
            "cer_finetuned_vs_baseline": round(float(cer_vs_baseline), 4),
            "transform_opportunity": transform_opportunity,
            "applied_phonetic": applied_phonetic,
            "is_raw_output": is_raw_output,
            "transform_cats": dict(cats),
            "subs": subs,
        })

    return results


# ────────────────────────────────────────────
# 5. 통계
# ────────────────────────────────────────────
def summarize(results):
    n = len(results)
    if n == 0:
        return {}

    cer_main = [r["cer_finetuned_vs_phonetic"] for r in results]
    cer_base = [r["cer_finetuned_vs_baseline"] for r in results]

    n_applied = sum(1 for r in results if r["applied_phonetic"])
    n_raw = sum(1 for r in results if r["is_raw_output"])
    n_opportunity = sum(1 for r in results if r["transform_opportunity"])

    # 음운 변동 카테고리 집계
    cat_total = Counter()
    for r in results:
        for k, v in r["transform_cats"].items():
            cat_total[k] += v

    # 가장 흔한 음절-치환 패턴
    sub_pairs = Counter()
    for r in results:
        for exp, act in r["subs"]:
            sub_pairs[(exp, act)] += 1

    return {
        "n_samples": n,
        "cer_finetuned_vs_phonetic": {
            "mean": round(float(np.mean(cer_main)), 4),
            "median": round(float(np.median(cer_main)), 4),
            "std": round(float(np.std(cer_main)), 4),
        },
        "cer_finetuned_vs_baseline": {
            "mean": round(float(np.mean(cer_base)), 4),
            "median": round(float(np.median(cer_base)), 4),
        },
        "phonetic_match_rate": round(n_applied / n, 4),
        "raw_output_rate": round(n_raw / n, 4),
        "transform_opportunity_rate": round(n_opportunity / n, 4),
        "phonetic_transform_categories": dict(cat_total),
        "top_substitutions": [
            {"expected": e, "actual": a, "count": c}
            for (e, a), c in sub_pairs.most_common(15)
        ],
    }


def print_summary(summary, results, sample_n=15):
    print(f"\n{'='*70}")
    print(f"  📊 Baseline-Referenced 검증 결과")
    print(f"{'='*70}")
    print(f"  샘플 수: {summary['n_samples']}")
    print()
    print(f"  ★ 핵심 지표 — 파인튜닝(Y) vs g2pk(베이스라인 X)")
    cm = summary["cer_finetuned_vs_phonetic"]
    print(f"    평균 CER: {cm['mean']:.4f}  중간값: {cm['median']:.4f}  σ={cm['std']:.4f}")
    print(f"    → 낮을수록 \"파인튜닝이 g2pk-변환된 출력을 안정적으로 생성\"")
    print()
    print(f"  📐 보조 지표")
    cb = summary["cer_finetuned_vs_baseline"]
    print(f"    파인튜닝 vs 베이스라인 CER: 평균 {cb['mean']:.4f}  중간값 {cb['median']:.4f}")
    print(f"    → 0이 아닐수록 파인튜닝이 베이스라인과 다른 (Raw) 출력을 만듦")
    print()
    print(f"  📊 정성 지표")
    print(f"    Phonetic Match Rate (Y == g2pk(X)): "
          f"{summary['phonetic_match_rate']:.1%}")
    print(f"    Raw Output Rate (Y != X):           "
          f"{summary['raw_output_rate']:.1%}")
    print(f"    음운 변동 기회 비율 (g2pk(X) != X): "
          f"{summary['transform_opportunity_rate']:.1%}")
    print()
    print(f"  🔬 한국어 음운 변동 감지")
    cats = summary.get("phonetic_transform_categories", {})
    for k, v in sorted(cats.items(), key=lambda x: -x[1]):
        label = {
            "tensification": "경음화 (ㄱ→ㄲ 등)",
            "palatalization": "구개음화 (ㄷ/ㅌ + ㅣ → ㅈ/ㅊ)",
            "nasalization": "비음화 (ㄱ→ㅇ, ㄷ→ㄴ, ㅂ→ㅁ)",
            "linking_candidate": "연음화 후보",
            "other": "기타 변동",
        }.get(k, k)
        print(f"    {label}: {v}회")

    if summary["top_substitutions"]:
        print(f"\n  🔍 빈도 높은 자모 치환 TOP 10:")
        for it in summary["top_substitutions"][:10]:
            print(f"    {it['expected']} → {it['actual']}  ({it['count']}회)")

    print(f"\n  📋 시연 사례 (앞 {sample_n}개):")
    print(f"  {'-'*70}")
    for i, r in enumerate(results[:sample_n]):
        ok_phon = "✅" if r["applied_phonetic"] else " "
        ok_raw = "✅" if r["is_raw_output"] else " "
        print(f"\n  [{i+1:3d}] dur={r['duration']:.1f}s")
        print(f"    베이스라인 X        : {r['baseline_X']}")
        print(f"    g2pk(X) 기대 발음   : {r['expected_phonetic']}")
        print(f"    파인튜닝 Y          : {r['finetuned_Y']}")
        print(f"    Y==g2pk(X)? [{ok_phon}]   Y!=X (Raw)? [{ok_raw}]   "
              f"CER(Y, g2pk(X))={r['cer_finetuned_vs_phonetic']:.3f}")


# ────────────────────────────────────────────
# 6. 결과 저장
# ────────────────────────────────────────────
def save_results(results, summary, output_dir):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results},
                  f, ensure_ascii=False, indent=2)

    md = []
    md.append("# AIHub 구음장애 — Baseline-Referenced 검증\n")
    md.append("> 정렬 문제 우회: 라벨 대신 베이스라인 Whisper 출력을 표준 표기로 사용\n")
    md.append(f"\n- 평가 샘플 수: **{summary['n_samples']}**\n")

    md.append("\n## 1. 핵심 지표\n")
    md.append("| 항목 | 값 | 해석 |")
    md.append("|------|----|------|")
    cm = summary["cer_finetuned_vs_phonetic"]
    md.append(f"| **CER(파인튜닝 Y, g2pk(베이스라인 X))** — 평균 | "
              f"**{cm['mean']:.4f}** | 낮을수록 phonetic-transcription 학습 성공 |")
    md.append(f"| CER 중간값 | {cm['median']:.4f} | |")
    cb = summary["cer_finetuned_vs_baseline"]
    md.append(f"| CER(파인튜닝 Y, 베이스라인 X) — 평균 | "
              f"{cb['mean']:.4f} | 0이 아닐수록 Raw 출력 (교정 거부) |")
    md.append(f"| Phonetic Match Rate (Y == g2pk(X)) | "
              f"**{summary['phonetic_match_rate']:.1%}** | |")
    md.append(f"| Raw Output Rate (Y != X) | "
              f"**{summary['raw_output_rate']:.1%}** | |")

    md.append("\n## 2. 한국어 음운 변동 감지\n")
    md.append("| 카테고리 | 횟수 |\n|---------|-----|")
    for k, v in sorted(summary.get("phonetic_transform_categories", {}).items(),
                       key=lambda x: -x[1]):
        md.append(f"| {k} | {v} |")

    md.append("\n## 3. 빈도 높은 자모 치환 TOP 15\n")
    md.append("| 순위 | 표준 X | 발음 Y | 횟수 |\n|------|-------|-------|-----|")
    for i, it in enumerate(summary["top_substitutions"], 1):
        md.append(f"| {i} | {it['expected']} | {it['actual']} | "
                  f"{it['count']} |")

    md.append("\n## 4. 해석\n")
    md.append(
        "- 베이스라인(원본 Whisper)은 자동 교정으로 표준 한국어 표기를 출력\n"
        "- 파인튜닝 모델은 동일 오디오에 대해 **g2pk-변환된 발음 표기**를 자동 생성\n"
        "- 두 모델 출력이 \"표기 ↔ 발음\" 관계를 형성하면, 앱에서 사용자의 "
        "발음 오류를 자모 단위로 감지 가능\n"
        "- 청각장애인 사용 시에도 동일 메커니즘으로 발음 피드백을 제공할 수 "
        "있음을 간접 증명"
    )

    with open(out / "summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\n  💾 결과 저장:")
    print(f"     {out / 'eval_results.json'}")
    print(f"     {out / 'summary.md'}")


# ────────────────────────────────────────────
# 7. 메인
# ────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="AIHub 구음장애 — Baseline-Referenced 검증"
    )
    parser.add_argument("--model_path", type=str,
                        default=str(HOME / "mingly_workspace" / "Voice-Model-Test"
                                    / "best_model_whisper" / "best"))
    parser.add_argument("--baseline_model", type=str, default="openai/whisper-tiny")
    parser.add_argument("--json_dir", type=str,
                        default=str(HOME / "mingly_workspace" / "Voice-Model-Test"
                                    / "segmented_dataset"))
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--min_dur", type=float, default=1.0,
                        help="최소 세그먼트 길이 (초)")
    parser.add_argument("--max_dur", type=float, default=10.0,
                        help="최대 세그먼트 길이 (초). 단일 발화 가능성 ↑")
    parser.add_argument("--output_dir", type=str, default="results/aihub_baseline_ref")
    parser.add_argument("--print_samples", type=int, default=15)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Device: {device}")

    # 1) 데이터 (라벨 미사용)
    records = load_audio_samples(
        args.json_dir, num_samples=args.num_samples,
        min_dur=args.min_dur, max_dur=args.max_dur,
    )
    if not records:
        print("❌ 평가 가능한 샘플이 없습니다.")
        return

    # 2) 모델
    print(f"\n📥 베이스라인 로드: {args.baseline_model}")
    base_pack = load_whisper(args.baseline_model, device)
    print(f"📥 파인튜닝 로드: {args.model_path}")
    ft_pack = load_whisper(args.model_path, device)

    # 3) G2P
    print(f"\n🔤 G2P 로드...")
    from korean_g2p_nomecab import load_g2p
    g2p = load_g2p()

    # 4) 평가
    print(f"\n{'='*70}")
    print(f"  🔬 Baseline-Referenced 검증 시작 ({len(records)}개)")
    print(f"{'='*70}")
    results = evaluate(records, base_pack, ft_pack, g2p, device)

    # 5) 집계 및 출력
    summary = summarize(results)
    print_summary(summary, results, sample_n=args.print_samples)

    # 6) 저장
    save_results(results, summary, args.output_dir)

    print(f"\n{'='*70}")
    print(f"  ✅ 검증 완료")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()