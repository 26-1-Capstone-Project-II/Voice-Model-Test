"""
jamo_utils.py — 자모 단위 Vocab + 음절↔자모 변환 유틸리티
==========================================================
기존 음절 단위 vocab (1,207개)을 자모 단위 (~57개)로 교체하여:
  - OOV(미등록 음절) 문제 완전 해소
  - 발음 전사 라벨을 100% 인코딩 가능
  - 자모 레벨 오류 감지 (청각장애인 발음 분석 핵심)

vocab 구성:
  [특수토큰 4개] + [구분자 1개] + [초성 19] + [중성 21] + [겹받침 11] = 56개
  ※ 종성 단자음(ㄱㄴㄷ 등)은 초성과 동일 문자 → 별도 토큰 불필요

사용법:
    from jamo_utils import syllable_to_jamo, jamo_to_syllable, build_jamo_processor

    # 음절 → 자모 (학습 라벨 생성)
    syllable_to_jamo("가치 해볼까")  # → "ㄱㅏㅊㅣ|ㅎㅐㅂㅗㄹㄲㅏ"

    # 자모 → 음절 (추론 결과 복원)
    jamo_to_syllable("ㄱㅏㅊㅣ ㅎㅐㅂㅗㄹㄲㅏ")  # → "가치 해볼까"

    # Wav2Vec2 호환 Processor 생성 (자모 vocab)
    processor = build_jamo_processor("./jamo_tokenizer")
"""

import json
import os
from pathlib import Path


# ════════════════════════════════════════════════════════════
# 1. 자모 상수 정의
# ════════════════════════════════════════════════════════════

CHOSUNG = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")       # 19개
JUNGSUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")  # 21개
JONGSUNG = list(" ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")  # 28개 (첫번째 공백 = 종성 없음)

# 겹받침: 초성에 없는 종성 전용 문자
COMPOUND_JONGSUNG = list("ㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ")  # 11개

# 자모 분류 집합
CONSONANTS = set(CHOSUNG) | set(COMPOUND_JONGSUNG)
VOWELS = set(JUNGSUNG)

KO_BASE = 0xAC00
JUNG_N = len(JUNGSUNG)   # 21
JONG_N = len(JONGSUNG)   # 28


# ════════════════════════════════════════════════════════════
# 2. 음절 분해 / 합성
# ════════════════════════════════════════════════════════════

def decompose(ch: str):
    """한글 한 음절 → (초성, 중성, 종성) 반환. 종성 없으면 종성=None."""
    code = ord(ch) - KO_BASE
    if code < 0 or code > 11171:
        return None
    cho = CHOSUNG[code // (JUNG_N * JONG_N)]
    jung = JUNGSUNG[(code % (JUNG_N * JONG_N)) // JONG_N]
    jong_idx = code % JONG_N
    jong = JONGSUNG[jong_idx] if jong_idx > 0 else None
    return cho, jung, jong


def compose(cho: str, jung: str, jong: str = None) -> str:
    """(초성, 중성, 종성) → 한글 한 음절 합성."""
    cho_i = CHOSUNG.index(cho)
    jung_i = JUNGSUNG.index(jung)
    jong_i = JONGSUNG.index(jong) if jong and jong in JONGSUNG else 0
    return chr(KO_BASE + cho_i * JUNG_N * JONG_N + jung_i * JONG_N + jong_i)


# ════════════════════════════════════════════════════════════
# 3. 음절 ↔ 자모 변환
# ════════════════════════════════════════════════════════════

def syllable_to_jamo(text: str) -> str:
    """
    한글 음절 문자열 → 자모 시퀀스 (학습 라벨용).

    규칙:
      - 한글 음절 → 초성 + 중성 + (종성) 분해
      - 공백 → | (Wav2Vec2 표준 word_delimiter_token)
      - 비한글 문자 → 제거 (숫자, 영문, 특수문자)

    예시:
      "가치 해볼까" → "ㄱㅏㅊㅣ|ㅎㅐㅂㅗㄹㄲㅏ"
      "좋네요"     → "ㅈㅗㅎㄴㅔㅇㅛ"
      "먹어요"     → "ㅁㅓㄱㅇㅓㅇㅛ"
    """
    result = []
    for ch in text:
        if ch == " ":
            result.append("|")
        elif "\uAC00" <= ch <= "\uD7A3":  # 한글 음절 범위
            d = decompose(ch)
            if d:
                cho, jung, jong = d
                result.append(cho)
                result.append(jung)
                if jong:
                    result.append(jong)
        # 비한글 문자는 스킵
    return "".join(result)


def jamo_to_syllable(jamo_text: str) -> str:
    """
    자모 시퀀스 → 한글 음절 문자열 (추론 결과 복원용).

    그리디 재조립 알고리즘:
      1. 자음이 오면 → 초성 후보로 설정
      2. 모음이 오면 → 중성 확정
      3. 다음 문자 확인:
         - 모음이면 → 현재 자음은 다음 음절 초성 (현재 음절 종성 없이 확정)
         - 자음/끝이면 → 현재 자음을 종성으로 추가

    예시:
      "ㄱㅏㅊㅣ ㅎㅐㅂㅗㄹㄲㅏ" → "가치 해볼까"
    """
    # | 를 공백으로 변환 후 단어 단위 처리
    words = jamo_text.replace("|", " ").split(" ")
    result_words = []

    for word in words:
        if not word:
            continue
        chars = list(word)
        syllables = []
        i = 0

        while i < len(chars):
            ch = chars[i]

            # 자음 → 초성 시작
            if ch in CONSONANTS and ch in CHOSUNG:
                cho = ch
                i += 1

                # 모음 → 중성
                if i < len(chars) and chars[i] in VOWELS:
                    jung = chars[i]
                    i += 1

                    # 다음이 자음이면 종성 후보
                    if i < len(chars) and chars[i] in CONSONANTS:
                        # 다다음이 모음이면 → 이 자음은 다음 음절 초성
                        if i + 1 < len(chars) and chars[i + 1] in VOWELS:
                            syllables.append(compose(cho, jung))
                        else:
                            # 종성으로 확정
                            jong = chars[i]
                            i += 1
                            # 겹받침인지 확인
                            if jong in COMPOUND_JONGSUNG:
                                syllables.append(compose(cho, jung, jong))
                            elif jong in CHOSUNG:
                                # 단자음 종성
                                jong_char = jong if jong in JONGSUNG else None
                                if jong_char:
                                    syllables.append(compose(cho, jung, jong_char))
                                else:
                                    syllables.append(compose(cho, jung))
                                    i -= 1  # 재처리
                            else:
                                syllables.append(compose(cho, jung))
                                i -= 1
                    else:
                        # 종성 없이 확정
                        syllables.append(compose(cho, jung))
                else:
                    # 초성만 단독 (비정상) → 그대로 출력
                    syllables.append(cho)

            elif ch in VOWELS:
                # 초성 없이 모음 단독 → ㅇ 초성 추가
                jung = ch
                i += 1
                syllables.append(compose("ㅇ", jung))

            elif ch in COMPOUND_JONGSUNG:
                # 겹받침이 단독으로 온 경우 → 그대로 출력
                syllables.append(ch)
                i += 1
            else:
                # 알 수 없는 문자 → 그대로
                syllables.append(ch)
                i += 1

        result_words.append("".join(syllables))

    return " ".join(result_words)


# ════════════════════════════════════════════════════════════
# 4. 자모 Vocab 생성
# ════════════════════════════════════════════════════════════

def build_jamo_vocab() -> dict:
    """
    자모 단위 vocab dict 생성 (Wav2Vec2CTCTokenizer 호환).

    구조:
      <pad>: 0   — CTC blank + padding
      <s>:   1   — BOS
      </s>:  2   — EOS
      <unk>: 3   — Unknown
      |:     4   — Word delimiter (공백)
      ㄱ:    5   — 초성/종성 자음
      ...
      ㅣ:    45  — 마지막 모음
      ㄳ:    46  — 겹받침 시작
      ...
      ㅄ:    56  — 마지막 겹받침

    총 57개 토큰.
    """
    vocab = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "<unk>": 3,
        "|": 4,
    }

    idx = 5

    # 초성 자음 (종성 단자음과 공유)
    for ch in CHOSUNG:
        if ch not in vocab:
            vocab[ch] = idx
            idx += 1

    # 중성 모음
    for ch in JUNGSUNG:
        if ch not in vocab:
            vocab[ch] = idx
            idx += 1

    # 겹받침 (초성에 없는 종성 전용)
    for ch in COMPOUND_JONGSUNG:
        if ch not in vocab:
            vocab[ch] = idx
            idx += 1

    return vocab


def save_jamo_vocab(save_dir: str) -> str:
    """vocab.json 파일로 저장."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    vocab = build_jamo_vocab()
    vocab_path = save_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"✅ 자모 vocab 저장 완료: {vocab_path} ({len(vocab)}개 토큰)")
    return str(vocab_path)


# ════════════════════════════════════════════════════════════
# 5. Wav2Vec2 Processor 생성 (자모 vocab 기반)
# ════════════════════════════════════════════════════════════

def build_jamo_processor(save_dir: str, base_model: str = "w11wo/wav2vec2-xls-r-300m-korean"):
    """
    자모 vocab 기반 Wav2Vec2Processor 생성 및 저장.

    1. base_model에서 FeatureExtractor 가져오기 (오디오 전처리는 동일)
    2. 새 자모 vocab으로 CTCTokenizer 생성
    3. Processor = FeatureExtractor + Tokenizer

    Returns:
        Wav2Vec2Processor 인스턴스
    """
    from transformers import (
        Wav2Vec2CTCTokenizer,
        Wav2Vec2FeatureExtractor,
        Wav2Vec2Processor,
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Vocab 저장
    vocab_path = save_jamo_vocab(str(save_dir))

    # 2. Tokenizer 생성 (자모 vocab 사용)
    tokenizer = Wav2Vec2CTCTokenizer(
        vocab_path,
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
        bos_token="<s>",
        eos_token="</s>",
    )

    # 3. FeatureExtractor (base 모델의 오디오 전처리 설정 재사용)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(base_model)

    # 4. Processor 조립
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )

    # 5. 저장
    processor.save_pretrained(str(save_dir))
    print(f"✅ 자모 Processor 저장 완료: {save_dir}")
    print(f"   vocab_size = {tokenizer.vocab_size}")

    return processor


# ════════════════════════════════════════════════════════════
# 6. 검증 테스트
# ════════════════════════════════════════════════════════════

def _test():
    """자모 분해/재조립 + vocab 검증."""
    print("=" * 56)
    print("  자모 유틸리티 검증 테스트")
    print("=" * 56)

    # 1. 분해 → 재조립 왕복 테스트
    test_cases = [
        "가치 해볼까",
        "좋네요",
        "먹어요",
        "안녕하세요",
        "같이 해볼까",
        "닭볶음탕",
        "앉아서",
        "읽었다",
        "천천히 말해주세요",
        "저는 잘 들리지 않아요",
    ]

    print("\n[1] 분해 → 재조립 왕복 테스트")
    print(f"{'원문':^16} | {'자모':^24} | {'복원':^16} | 일치")
    print("-" * 75)

    all_pass = True
    for text in test_cases:
        jamo = syllable_to_jamo(text)
        restored = jamo_to_syllable(jamo)
        match = "✅" if restored == text else "❌"
        if restored != text:
            all_pass = False
        print(f"  {text:^14} | {jamo:^22} | {restored:^14} | {match}")

    # 2. Vocab 검증
    print(f"\n[2] Vocab 검증")
    vocab = build_jamo_vocab()
    print(f"  총 토큰 수: {len(vocab)}개")
    print(f"  특수 토큰: <pad>={vocab['<pad>']}, <unk>={vocab['<unk>']}, |={vocab['|']}")

    # 모든 자모가 vocab에 있는지 확인
    missing = []
    for text in test_cases:
        jamo = syllable_to_jamo(text)
        for ch in jamo:
            if ch not in vocab:
                missing.append(ch)

    if missing:
        print(f"  ❌ Vocab 누락 문자: {set(missing)}")
        all_pass = False
    else:
        print(f"  ✅ 모든 자모가 vocab에 포함됨")

    # 3. 토큰화 검증
    print(f"\n[3] 토큰화 예시")
    for text in test_cases[:3]:
        jamo = syllable_to_jamo(text)
        ids = [vocab.get(ch, vocab["<unk>"]) for ch in jamo]
        print(f"  {text:^14} → {jamo} → {ids}")

    print(f"\n{'=' * 56}")
    result = "✅ 모든 테스트 통과!" if all_pass else "⚠️  일부 테스트 실패"
    print(f"  {result}")
    print(f"{'=' * 56}")


if __name__ == "__main__":
    _test()
