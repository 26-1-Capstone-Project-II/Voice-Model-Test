"""
화자별 발음 약점 프로파일 분석 (기존 eval_results.json 재가공)
================================================================
test_aihub_baseline_ref.py 가 이미 저장한 eval_results.json 을 읽어서
화자별로 어떤 음운변동에 자주 어긋나는지 자모 단위로 출력한다.

재추론 없이 빠르게 결과만 다시 분석할 때 사용.

실행:
    python analyze_speaker_profiles.py \\
        --result_json results/aihub_value_proposition/eval_results.json
"""

import json
import argparse
from pathlib import Path
from collections import Counter, defaultdict


CATEGORY_KO = {
    "tensification": "경음화",
    "palatalization": "구개음화",
    "nasalization": "비음화",
    "linking": "연음화",
}


def compute_speaker_profiles(results, min_utt=3, top_n=5):
    by_speaker = defaultdict(list)
    for r in results:
        if r.get("speaker_id"):
            by_speaker[r["speaker_id"]].append(r)

    profiles = []
    for sp, items in by_speaker.items():
        if len(items) < min_utt:
            continue
        n_utt = len(items)

        pair_total = Counter()
        pair_utts = Counter()
        for r in items:
            seen = set()
            for s in r.get("subs", []):
                pair = tuple(s) if isinstance(s, list) else s
                pair_total[pair] += 1
                if pair not in seen:
                    pair_utts[pair] += 1
                    seen.add(pair)

        cat_total = Counter()
        cat_utts = Counter()
        for r in items:
            seen_cats = set()
            for k, v in r.get("transform_cats", {}).items():
                cat_total[k] += v
                if v > 0 and k not in seen_cats:
                    cat_utts[k] += 1
                    seen_cats.add(k)

        rule_only = {k: cat_utts.get(k, 0)
                     for k in ["tensification", "palatalization",
                               "nasalization", "linking"]}
        weakness = (max(rule_only.items(), key=lambda x: x[1])[0]
                    if any(rule_only.values()) else None)

        profiles.append({
            "speaker_id": sp,
            "n_utterances": n_utt,
            "weakness": CATEGORY_KO.get(weakness, "없음") if weakness else "없음",
            "category_distribution": {
                CATEGORY_KO[k]: {
                    "total_count": cat_total.get(k, 0),
                    "utterance_count": cat_utts.get(k, 0),
                    "utterance_rate": round(cat_utts.get(k, 0) / n_utt, 4),
                }
                for k in ["tensification", "palatalization",
                          "nasalization", "linking"]
            },
            "top_patterns": [
                {
                    "pattern": f"{e}→{a}",
                    "total": pair_total[(e, a)],
                    "utts": pair_utts[(e, a)],
                    "rate": round(pair_utts[(e, a)] / n_utt, 4),
                }
                for (e, a), _ in pair_total.most_common(top_n)
            ],
        })

    profiles.sort(key=lambda p: -p["n_utterances"])
    return profiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_json", required=True,
                        help="eval_results.json 경로")
    parser.add_argument("--top_n", type=int, default=5)
    parser.add_argument("--min_utt", type=int, default=3)
    args = parser.parse_args()

    data = json.loads(Path(args.result_json).read_text(encoding="utf-8"))
    results = data.get("results", [])
    if not results:
        print("❌ results 비어있음")
        return

    profiles = compute_speaker_profiles(results, args.min_utt, args.top_n)
    if not profiles:
        print(f"❌ ≥{args.min_utt} 발화 화자가 없습니다")
        return

    print(f"\n{'='*72}")
    print(f"  👤 화자별 발음 약점 프로파일  "
          f"(tag={data.get('tag', 'unknown')}, n={len(profiles)}명)")
    print(f"{'='*72}")
    print(f"  ※ 각 화자가 어떤 음운변동에 자주 어긋나는지를 자모 단위로 분석.")

    for p in profiles:
        print(f"\n  ━ {p['speaker_id']}  ({p['n_utterances']} 발화)")
        print(f"    🎯 주요 약점 카테고리: {p['weakness']}")
        print(f"    📊 음운변동 카테고리 분포:")
        for cat, info in p["category_distribution"].items():
            if info["utterance_count"] > 0:
                print(f"        {cat}: {info['utterance_count']}/"
                      f"{p['n_utterances']} 발화 "
                      f"({info['utterance_rate']:.0%})  "
                      f"총 {info['total_count']}회")
        print(f"    🔤 자주 등장 자모 치환 TOP {args.top_n}:")
        for i, pat in enumerate(p["top_patterns"], 1):
            print(f"        {i}. {pat['pattern']}  "
                  f"{pat['utts']}/{p['n_utterances']} 발화 "
                  f"({pat['rate']:.0%})  "
                  f"총 {pat['total']}회")


if __name__ == "__main__":
    main()