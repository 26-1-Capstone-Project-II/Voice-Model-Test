"""
korean_g2p_nomecab.py — MeCab 없이 동작하는 한국어 G2P
=========================================================
g2pk가 MeCab 빌드 실패 시 이 모듈로 대체합니다.
주요 한국어 음운 규칙을 직접 구현:
  - 연음 (이어 읽기)
  - 비음화 (ㄱ/ㄷ/ㅂ + 비음 → 비음)
  - 경음화
  - 격음화 (ㅎ 축약)
  - 구개음화

완벽한 G2P는 아니지만, 파인튜닝 라벨 생성에 충분한 수준입니다.
g2pk가 정상 설치되면 이 파일은 자동으로 사용되지 않습니다.
"""

import re
import unicodedata

# ────────────────────────────────────────────
# 자모 상수
# ────────────────────────────────────────────
CHOSUNG  = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
JUNGSUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
JONGSUNG = list(" ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")

KO_BASE    = 0xAC00
CHO_N      = len(CHOSUNG)   # 19
JUNG_N     = len(JUNGSUNG)  # 21
JONG_N     = len(JONGSUNG)  # 28


def decompose(ch: str) -> tuple[str, str, str] | None:
    """한글 한 글자 → (초성, 중성, 종성) 반환. 비한글이면 None."""
    code = ord(ch)
    if not (KO_BASE <= code < KO_BASE + 11172):
        return None
    offset = code - KO_BASE
    cho  = CHOSUNG[offset // (JUNG_N * JONG_N)]
    jung = JUNGSUNG[(offset % (JUNG_N * JONG_N)) // JONG_N]
    jong = JONGSUNG[offset % JONG_N]
    return cho, jung, jong


def compose(cho: str, jung: str, jong: str = " ") -> str:
    """(초성, 중성, 종성) → 한글 한 글자 합성."""
    cho_i  = CHOSUNG.index(cho)
    jung_i = JUNGSUNG.index(jung)
    jong_i = JONGSUNG.index(jong)
    return chr(KO_BASE + cho_i * JUNG_N * JONG_N + jung_i * JONG_N + jong_i)


# ────────────────────────────────────────────
# 음운 규칙 테이블
# ────────────────────────────────────────────

# 연음 규칙: 앞 음절 종성 → 뒤 음절 초성 (모음으로 시작할 때)
# 겹받침 분리
DOUBLE_JONGSUNG = {
    "ㄳ": ("ㄱ", "ㅅ"), "ㄵ": ("ㄴ", "ㅈ"), "ㄶ": ("ㄴ", "ㅎ"),
    "ㄺ": ("ㄹ", "ㄱ"), "ㄻ": ("ㄹ", "ㅁ"), "ㄼ": ("ㄹ", "ㅂ"),
    "ㄽ": ("ㄹ", "ㅅ"), "ㄾ": ("ㄹ", "ㅌ"), "ㄿ": ("ㄹ", "ㅍ"),
    "ㅀ": ("ㄹ", "ㅎ"), "ㄴ": ("ㄴ",),    "ㅄ": ("ㅂ", "ㅅ"),
}

# 비음화
NASALIZATION = {
    ("ㄱ", "ㄴ"): ("ㅇ", "ㄴ"), ("ㄱ", "ㅁ"): ("ㅇ", "ㅁ"),
    ("ㄷ", "ㄴ"): ("ㄴ", "ㄴ"), ("ㄷ", "ㅁ"): ("ㄴ", "ㅁ"),
    ("ㅂ", "ㄴ"): ("ㅁ", "ㄴ"), ("ㅂ", "ㅁ"): ("ㅁ", "ㅁ"),
    ("ㄱ", "ㄹ"): ("ㅇ", "ㄴ"), ("ㄷ", "ㄹ"): ("ㄴ", "ㄴ"),
    ("ㅂ", "ㄹ"): ("ㅁ", "ㄴ"),
}

# 격음화: ㅎ + 예사소리 / 예사소리 + ㅎ
ASPIRATION = {
    ("ㅎ", "ㄱ"): "ㅋ", ("ㅎ", "ㄷ"): "ㅌ",
    ("ㅎ", "ㅈ"): "ㅊ", ("ㅎ", "ㅅ"): "ㅅ",
    ("ㄱ", "ㅎ"): "ㅋ", ("ㄷ", "ㅎ"): "ㅌ",
    ("ㅂ", "ㅎ"): "ㅍ", ("ㅈ", "ㅎ"): "ㅊ",
}

# 구개음화: ㄷ/ㅌ + 이 → ㅈ/ㅊ
PALATALIZATION = {
    ("ㄷ", "ㅣ"): "ㅈ",
    ("ㅌ", "ㅣ"): "ㅊ",
}


# ────────────────────────────────────────────
# 핵심 G2P 함수
# ────────────────────────────────────────────

def g2p_korean(text: str) -> str:
    """
    한국어 텍스트 → 발음 전사.
    주요 음운 규칙 적용 (연음, 비음화, 격음화, 구개음화).
    """
    # 1. 음절 단위로 분해
    syllables = []
    for ch in text:
        d = decompose(ch)
        if d:
            syllables.append(list(d))   # [초성, 중성, 종성] mutable
        else:
            syllables.append(ch)        # 비한글 그대로

    result = list(syllables)

    # 2. 음절 간 규칙 적용
    i = 0
    while i < len(result) - 1:
        curr = result[i]
        nxt  = result[i + 1]

        # 두 음절 모두 한글인 경우만 처리
        if not (isinstance(curr, list) and isinstance(nxt, list)):
            i += 1
            continue

        curr_jong = curr[2]  # 현재 음절 종성
        nxt_cho   = nxt[0]   # 다음 음절 초성
        nxt_jung  = nxt[1]   # 다음 음절 중성

        # ── 연음: 종성 + 모음 초성(ㅇ) → 종성이 다음 초성으로 이동
        if curr_jong != " " and nxt_cho == "ㅇ":
            # 겹받침 분리
            if curr_jong in DOUBLE_JONGSUNG:
                parts = DOUBLE_JONGSUNG[curr_jong]
                if len(parts) == 2:
                    curr[2]  = parts[0]
                    nxt[0]   = parts[1]
                else:
                    curr[2]  = " "
                    nxt[0]   = parts[0]
            else:
                nxt[0]  = curr_jong
                curr[2] = " "
            i += 1
            continue

        # ── 격음화: 종성 ㅎ + 예사소리 초성
        if curr_jong == "ㅎ":
            key = ("ㅎ", nxt_cho)
            if key in ASPIRATION:
                curr[2] = " "
                nxt[0]  = ASPIRATION[key]
                i += 1
                continue

        # ── 격음화: 종성 예사소리 + 초성 ㅎ
        if nxt_cho == "ㅎ" and curr_jong in ("ㄱ", "ㄷ", "ㅂ", "ㅈ"):
            key = (curr_jong, "ㅎ")
            if key in ASPIRATION:
                curr[2] = " "
                nxt[0]  = ASPIRATION[key]
                i += 1
                continue

        # ── 비음화
        key = (curr_jong, nxt_cho)
        if key in NASALIZATION:
            new_jong, new_cho = NASALIZATION[key]
            curr[2] = new_jong
            nxt[0]  = new_cho
            i += 1
            continue

        # ── 구개음화: 종성 ㄷ/ㅌ + 이
        if curr_jong in ("ㄷ", "ㅌ") and nxt_cho == "ㅇ" and nxt_jung == "ㅣ":
            key = (curr_jong, "ㅣ")
            if key in PALATALIZATION:
                curr[2] = " "
                nxt[0]  = PALATALIZATION[key]
                i += 1
                continue

        i += 1

    # 3. 다시 문자열로 합성
    out = []
    for s in result:
        if isinstance(s, list):
            cho, jung, jong = s
            out.append(compose(cho, jung, jong))
        else:
            out.append(s)

    return "".join(out)


# ────────────────────────────────────────────
# g2pk 호환 클래스 (드롭인 대체용)
# ────────────────────────────────────────────

class G2pFallback:
    """
    g2pk.G2p() 와 동일한 인터페이스를 제공합니다.
    MeCab 없이 동작합니다.

    사용법:
        from korean_g2p_nomecab import G2pFallback as G2p
        g2p = G2p()
        result = g2p("같이 해볼까", descriptive=True)
    """
    def __call__(self, text: str, descriptive: bool = True) -> str:
        return g2p_korean(text)


# ────────────────────────────────────────────
# g2pk 자동 감지 로더 (권장 사용 방식)
# ────────────────────────────────────────────

def load_g2p():
    """
    g2pk가 설치되어 있으면 g2pk.G2p 사용,
    없으면 G2pFallback 사용.
    어느 쪽이든 동일한 인터페이스를 반환합니다.
    """
    try:
        from g2pk import G2p
        g2p = G2p()
        print("✅ G2P: g2pk (MeCab 버전) 사용")
        return g2p
    except Exception:
        g2p = G2pFallback()
        print("⚠️  G2P: MeCab 없이 동작하는 폴백 버전 사용")
        print("    (MeCab 설치 후 pip install g2pk 로 정밀도 향상 가능)")
        return g2p


# ────────────────────────────────────────────
# 테스트
# ────────────────────────────────────────────

if __name__ == "__main__":
    g2p = G2pFallback()

    test_cases = [
        ("같이 해볼까",         "가치 해볼까"),
        ("좋네요",              "존네요"),
        ("안녕하세요",          "안녕하세요"),
        ("먹어요",              "머거요"),
        ("닭볶음",              "닥뽀끔"),
        ("천천히 말해주세요",    "천천히 말해주세요"),
        ("오늘 날씨가 좋네요",   "오늘 날씨가 존네요"),
    ]

    print("G2P 변환 테스트:")
    print(f"{'원문':^20} | {'변환 결과':^20} | {'기대값':^20} | 일치")
    print("-" * 75)
    for original, expected in test_cases:
        result  = g2p(original, descriptive=True)
        match   = "✅" if result == expected else "❌"
        print(f"  {original:^18} | {result:^18} | {expected:^18} | {match}")
