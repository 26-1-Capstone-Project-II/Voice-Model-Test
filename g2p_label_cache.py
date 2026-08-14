"""
G2P 라벨 캐시 + 병렬 계산
==========================
문제: `--apply_g2p` 학습은 매 런마다 코퍼스 전체(Zeroth 21,763 + 외래어 1,446 +
KsponSpeech 56,832 ≈ 8만 문장)에 g2pk 를 다시 돌린다. 어절 단위 래퍼
(korean_g2p_nomecab._EojeolWiseG2p)가 어절마다 g2pk 를 호출하므로 실제 호출은 80만 회를
넘고, 서버에서 3시간 가까이 CPU 한 코어만 붙들며 그동안 GPU 는 논다.

대응 두 가지:
  1) 캐시 — 문장→라벨 을 JSON 으로 저장해 두 번째 런부터 재계산을 생략한다.
     캐시 파일명에 **G2P 규약 지문**(korean_g2p_nomecab.py 내용 + 엔진 종류의 해시)을
     넣어, G2P 규칙을 고치면 자동으로 다른 파일이 된다. 규약이 바뀐 뒤에도 옛 라벨이
     조용히 재사용되는 사고(어절 단위 수정 전 라벨과 섞이는 문제)를 구조적으로 막는다.
  2) 병렬 — 캐시에 없는 문장만 워커 프로세스로 나눠 계산한다(g2pk 는 CPU 전용이라
     GPU 로는 못 옮긴다). 워커 수는 G2P_WORKERS 로 조절.

사용:
    from g2p_label_cache import build_label_map
    label_map = build_label_map(texts, cache_dir="zeroth_dataset")
    label = label_map.get(text, "")
"""

import os
import json
import hashlib
from pathlib import Path

_G2P = None                      # 워커 프로세스별 G2P 인스턴스
_SIGNATURE = None                # 규약 지문 캐시(프로세스 내 1회 계산)


def _engine_name():
    """실제로 쓰이는 G2P 엔진 이름 (g2pk 유무에 따라 라벨이 달라지므로 지문에 포함)."""
    try:
        import g2pk  # noqa: F401
        return "g2pk"
    except Exception:
        return "fallback"


def g2p_signature():
    """G2P 규약 지문 — 규칙 파일 내용 + 엔진 종류의 sha256(앞 16자)."""
    global _SIGNATURE
    if _SIGNATURE is None:
        h = hashlib.sha256()
        rules = Path(__file__).with_name("korean_g2p_nomecab.py")
        if rules.exists():
            h.update(rules.read_bytes())
        h.update(_engine_name().encode())
        _SIGNATURE = h.hexdigest()[:16]
    return _SIGNATURE


def _init_worker(quiet=True):
    """워커 프로세스마다 G2P 를 한 번만 로드(포크 후 초기화라 MeCab 도 안전).

    quiet: 워커 수만큼 반복되는 G2P 로딩 배너를 삼킨다(로그 가독성).
    """
    global _G2P
    import io, contextlib
    from korean_g2p_nomecab import load_g2p
    if quiet:
        with contextlib.redirect_stdout(io.StringIO()):
            _G2P = load_g2p()
    else:
        _G2P = load_g2p()


def _apply(text):
    return text, _G2P(text, descriptive=True).strip()


def _resolve_workers(workers, n_texts):
    if workers is None:
        workers = int(os.environ.get("G2P_WORKERS", "0")) or (os.cpu_count() or 2)
    # 문장이 적으면 프로세스 생성 비용이 더 크다
    return max(1, min(workers, 32, (n_texts // 200) or 1))


def build_label_map(texts, cache_dir=None, workers=None, desc=""):
    """문장 목록 → {문장: 발음 라벨} 사전.

    Args:
      texts:     원문 리스트(중복 허용 — 내부에서 중복 제거해 한 번만 계산)
      cache_dir: 캐시 JSON 을 둘 디렉터리(보통 해당 코퍼스의 json_dir). None 이면 캐시 미사용.
      workers:   병렬 프로세스 수. None 이면 G2P_WORKERS 또는 CPU 코어 수.
      desc:      로그 라벨
    """
    uniq = list(dict.fromkeys(t for t in texts if t))
    if not uniq:
        return {}

    sig = g2p_signature()
    cache_path = Path(cache_dir) / f".g2p_labels_{sig}.json" if cache_dir else None

    cached = {}
    if cache_path and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  ⚠️ G2P 캐시 읽기 실패({cache_path.name}): {e} — 새로 계산")
            cached = {}

    todo = [t for t in uniq if t not in cached]
    hit = len(uniq) - len(todo)
    tag = f"{desc} " if desc else ""
    print(f"  🔤 {tag}G2P 라벨: 총 {len(uniq):,}문장 / 캐시 적중 {hit:,} / 계산 {len(todo):,} "
          f"(규약 {sig})")

    if todo:
        n_workers = _resolve_workers(workers, len(todo))
        results = None
        if n_workers > 1:
            try:
                import multiprocessing as mp
                ctx = mp.get_context("fork") if hasattr(mp, "get_context") else mp
                print(f"     ↳ 병렬 {n_workers} 프로세스로 계산")
                with ctx.Pool(n_workers, initializer=_init_worker) as pool:
                    results = pool.map(_apply, todo, chunksize=256)
            except Exception as e:
                print(f"     ⚠️ 병렬 계산 실패({e}) — 단일 프로세스로 대체")
                results = None
        if results is None:
            _init_worker(quiet=False)
            results = [_apply(t) for t in todo]
        cached.update(dict(results))

        if cache_path:
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
                tmp.replace(cache_path)          # 원자적 교체(중단 시 반쪽 캐시 방지)
                print(f"     💾 캐시 저장: {cache_path.name} ({len(cached):,}문장)")
            except Exception as e:
                print(f"     ⚠️ 캐시 저장 실패: {e} (학습은 계속)")

    return {t: cached[t] for t in uniq if t in cached}

# ────────────────────────────────────────────
# 캐시 미리 만들기 (학습 전에 따로 돌려두면 학습 시작이 즉시 이뤄진다)
#   python g2p_label_cache.py --json_dirs zeroth_dataset,loanword_dataset,kspon_dataset
# ────────────────────────────────────────────
def _prebuild(json_dirs, splits=("train", "validation", "test"), workers=None):
    for d in json_dirs:
        d = Path(d)
        for sp in splits:
            path = d / f"{sp}.jsonl"
            if not path.exists():
                continue
            texts = []
            with open(path, encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line.strip())
                    except Exception:
                        continue
                    t = (obj.get("transcript") or "").strip()
                    if t:
                        texts.append(t)
            build_label_map(texts, cache_dir=d, workers=workers, desc=f"{d.name}/{sp}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="G2P 라벨 캐시 미리 생성")
    ap.add_argument("--json_dirs", required=True,
                    help="콤마 구분 데이터 디렉터리 (예: zeroth_dataset,kspon_dataset)")
    ap.add_argument("--workers", type=int, default=None,
                    help="병렬 프로세스 수 (기본: G2P_WORKERS 또는 CPU 코어 수)")
    a = ap.parse_args()
    _prebuild([x.strip() for x in a.json_dirs.split(",") if x.strip()], workers=a.workers)
    print("\n✅ 캐시 생성 완료 — 다음 학습부터 이 구간을 건너뜁니다.")
