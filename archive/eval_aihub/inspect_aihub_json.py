"""
AIHub 구음장애 라벨 JSON 구조 점검
====================================
세그멘테이션 재구현을 위해 AIHub JSON에 어떤 필드가 들어있는지,
특히 문장별 timestamp가 있는지를 확인한다.

실행:
    python inspect_aihub_json.py --json_dir "/path/to/TL02_언어청각장애"
"""

import json
import argparse
from pathlib import Path


def find_json_files(json_dir, max_files=3):
    """디렉토리에서 JSON 파일 몇 개 찾기."""
    p = Path(json_dir)
    if not p.exists():
        print(f"❌ 경로 없음: {p}")
        return []
    files = sorted(p.rglob("*.json"))
    print(f"📂 발견된 JSON: {len(files):,}개")
    return files[:max_files]


def dump_keys(obj, prefix="", max_depth=3, depth=0):
    """JSON 구조의 모든 키와 타입을 재귀적으로 덤프."""
    if depth >= max_depth:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            t = type(v).__name__
            if isinstance(v, (dict, list)) and v:
                size = len(v)
                print(f"  {prefix}{k}  [{t}, len={size}]")
                if isinstance(v, list) and v:
                    print(f"  {prefix}  └─ (첫 항목 미리보기)")
                    dump_keys(v[0], prefix + "    ", max_depth, depth + 1)
                else:
                    dump_keys(v, prefix + "  ", max_depth, depth + 1)
            else:
                preview = repr(v)[:80]
                print(f"  {prefix}{k}  [{t}]  = {preview}")


def find_timestamp_fields(obj, path=""):
    """timestamp 비슷한 필드를 찾아서 보고."""
    candidates = []
    keywords = ["time", "start", "end", "duration", "begin", "offset",
                "Time", "Start", "End", "Duration", "Begin", "Offset",
                "sentence", "Sentence", "segment", "Segment", "utterance"]

    def _walk(x, p):
        if isinstance(x, dict):
            for k, v in x.items():
                np_ = f"{p}.{k}" if p else k
                if any(kw in str(k) for kw in keywords):
                    sample = repr(v)[:120]
                    candidates.append((np_, type(v).__name__, sample))
                _walk(v, np_)
        elif isinstance(x, list) and x:
            _walk(x[0], f"{p}[0]")

    _walk(obj, path)
    return candidates


def main():
    parser = argparse.ArgumentParser(description="AIHub JSON 구조 점검")
    parser.add_argument("--json_dir", required=True,
                        help="AIHub 라벨 JSON 디렉토리 (예: TL02_언어청각장애)")
    parser.add_argument("--max_files", type=int, default=3,
                        help="검사할 파일 수 (기본 3)")
    args = parser.parse_args()

    files = find_json_files(args.json_dir, args.max_files)
    if not files:
        return

    for i, fp in enumerate(files, 1):
        print(f"\n{'='*70}")
        print(f"  📄 [{i}] {fp.name}")
        print(f"{'='*70}")
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  ⚠️ 읽기 실패: {e}")
            continue

        print(f"\n🔑 최상위 키 목록:")
        if isinstance(data, dict):
            for k in data.keys():
                print(f"  - {k}")

        print(f"\n📋 전체 구조 (depth=3):")
        dump_keys(data, max_depth=3)

        print(f"\n🕐 timestamp/sentence 관련 필드 후보:")
        candidates = find_timestamp_fields(data)
        if not candidates:
            print(f"  (없음)")
        else:
            for path, t, sample in candidates[:30]:
                print(f"  • {path}  [{t}]  = {sample}")

        # Transcript 필드 길이 확인
        if "Transcript" in data:
            tr = data["Transcript"]
            print(f"\n📝 Transcript 길이: {len(tr)} chars")
            print(f"  시작: {tr[:120]!r}")
            print(f"  끝:   {tr[-120:]!r}")


if __name__ == "__main__":
    main()