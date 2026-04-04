"""
발음 평가 모듈
=============
Whisper 모델 + G2P + 자모 비교로 발음 오류를 자모 단위로 감지.

앱 흐름:
  1. 사용자가 목표 문장 입력: "같이 먹을까?"
  2. 사용자가 문장 읽기 (녹음)
  3. Whisper → 발음 전사 출력: "가티 머글까"
  4. G2P(목표문장) → 기대 발음: "가치 머글까"
  5. 자모 레벨 비교 → 오류 감지: ㅊ→ㅌ

사용 예시:
    from pronunciation_evaluator import PronunciationEvaluator

    evaluator = PronunciationEvaluator("best_model_whisper/best")
    result = evaluator.evaluate(
        audio_path="recording.wav",
        target_text="같이 먹을까?"
    )
    print(result)
"""

import re
import json
import torch
import librosa
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

torch.backends.cudnn.enabled = False

TARGET_SR = 16000


# ────────────────────────────────────────────
# 1. 자모 유틸리티
# ────────────────────────────────────────────
CHOSUNG  = list("ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ")
JUNGSUNG = list("ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ")
JONGSUNG = list(" ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ")

KO_BASE = 0xAC00
JUNG_N = len(JUNGSUNG)   # 21
JONG_N = len(JONGSUNG)   # 28


def decompose_syllable(ch):
    """한글 음절 → (초성, 중성, 종성) 분해. 비한글이면 None."""
    code = ord(ch)
    if not (KO_BASE <= code < KO_BASE + 11172):
        return None
    offset = code - KO_BASE
    cho  = CHOSUNG[offset // (JUNG_N * JONG_N)]
    jung = JUNGSUNG[(offset % (JUNG_N * JONG_N)) // JONG_N]
    jong = JONGSUNG[offset % JONG_N]
    return (cho, jung, jong.strip() if jong.strip() else None)


def text_to_jamo(text):
    """텍스트 → 자모 리스트 (공백 포함)."""
    jamo_list = []
    for ch in text:
        if ch == ' ':
            jamo_list.append(' ')
        else:
            parts = decompose_syllable(ch)
            if parts:
                cho, jung, jong = parts
                jamo_list.extend([cho, jung])
                if jong:
                    jamo_list.append(jong)
            else:
                jamo_list.append(ch)  # 비한글 (숫자, 구두점 등)
    return jamo_list


def jamo_to_syllable_groups(jamo_list):
    """자모 리스트 → 음절 단위 그룹 (디스플레이용)."""
    groups = []
    current = []
    for j in jamo_list:
        if j == ' ':
            if current:
                groups.append(current)
                current = []
            groups.append([' '])
        else:
            current.append(j)
    if current:
        groups.append(current)
    return groups


# ────────────────────────────────────────────
# 2. 자모 레벨 비교 (Levenshtein Alignment)
# ────────────────────────────────────────────
def align_jamo(expected, actual):
    """
    두 자모 시퀀스를 Levenshtein 정렬하여 오류 위치 반환.

    Returns:
        list of dict: 각 위치의 비교 결과
            - position: 인덱스
            - expected: 기대 자모
            - actual: 실제 자모
            - status: "correct" | "substitution" | "insertion" | "deletion"
    """
    n, m = len(expected), len(actual)

    # DP 테이블
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if expected[i-1] == actual[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],    # deletion
                    dp[i][j-1],    # insertion
                    dp[i-1][j-1],  # substitution
                )

    # Backtrack
    alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and expected[i-1] == actual[j-1]:
            alignment.append({
                "expected": expected[i-1],
                "actual": actual[j-1],
                "status": "correct",
            })
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            alignment.append({
                "expected": expected[i-1],
                "actual": actual[j-1],
                "status": "substitution",
            })
            i -= 1
            j -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            alignment.append({
                "expected": None,
                "actual": actual[j-1],
                "status": "insertion",
            })
            j -= 1
        else:
            alignment.append({
                "expected": expected[i-1],
                "actual": None,
                "status": "deletion",
            })
            i -= 1

    alignment.reverse()

    # 위치 번호 추가
    for idx, a in enumerate(alignment):
        a["position"] = idx

    return alignment


def compute_pronunciation_score(alignment):
    """정렬 결과로 발음 점수(0~1) 계산."""
    if not alignment:
        return 0.0
    correct = sum(1 for a in alignment if a["status"] == "correct")
    return correct / len(alignment)


def extract_errors(alignment):
    """정렬 결과에서 오류만 추출."""
    return [a for a in alignment if a["status"] != "correct"]


# ────────────────────────────────────────────
# 3. 발음 평가기
# ────────────────────────────────────────────
class PronunciationEvaluator:
    """
    Whisper 기반 발음 평가기.

    사용:
        evaluator = PronunciationEvaluator("best_model_whisper/best")
        result = evaluator.evaluate("recording.wav", "같이 먹을까?")
    """

    def __init__(self, model_path, device=None):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        from korean_g2p_nomecab import load_g2p

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"📥 모델 로드: {model_path}")
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        print("🔤 G2P 로드...")
        self.g2p = load_g2p()

    def transcribe(self, audio_path):
        """오디오 → 발음 전사 텍스트."""
        audio, _ = librosa.load(audio_path, sr=TARGET_SR, mono=True)

        # 30초 이하로 자르기
        max_samples = 30 * TARGET_SR
        if len(audio) > max_samples:
            audio = audio[:max_samples]

        input_features = self.processor.feature_extractor(
            audio, sampling_rate=TARGET_SR, return_tensors="pt"
        ).input_features.to(self.device)

        with torch.no_grad():
            predicted_ids = self.model.generate(
                input_features,
                max_new_tokens=256,
                language="ko",
                task="transcribe",
            )

        transcription = self.processor.tokenizer.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0].strip()

        return transcription

    def evaluate(self, audio_path, target_text):
        """
        발음 평가 실행.

        Args:
            audio_path: 사용자 녹음 WAV 경로
            target_text: 목표 문장 (맞춤법)

        Returns:
            dict:
                target_text: 목표 문장
                expected_pronunciation: G2P 변환 결과 (기대 발음)
                actual_pronunciation: Whisper 출력 (실제 발음)
                score: 발음 점수 (0~1)
                cer: 문자 오류율
                errors: 오류 목록 [{position, expected, actual, status}]
                alignment: 전체 정렬 결과
        """
        # 1. 기대 발음 생성
        expected_pron = self.g2p(target_text, descriptive=True).strip()

        # 2. 실제 발음 전사
        actual_pron = self.transcribe(audio_path)

        # 3. 자모 분해
        expected_jamo = text_to_jamo(expected_pron)
        actual_jamo = text_to_jamo(actual_pron)

        # 4. 자모 레벨 정렬
        alignment = align_jamo(expected_jamo, actual_jamo)

        # 5. 점수 계산
        score = compute_pronunciation_score(alignment)
        errors = extract_errors(alignment)

        # 6. CER (문자 레벨)
        cer = 1.0 - score

        return {
            "target_text": target_text,
            "expected_pronunciation": expected_pron,
            "actual_pronunciation": actual_pron,
            "score": round(score, 4),
            "cer": round(cer, 4),
            "errors": errors,
            "error_count": len(errors),
            "total_jamo": len(alignment),
            "alignment": alignment,
        }

    def evaluate_batch(self, items):
        """
        배치 평가.

        Args:
            items: list of (audio_path, target_text)

        Returns:
            list of evaluation results
        """
        results = []
        for audio_path, target_text in items:
            try:
                result = self.evaluate(audio_path, target_text)
                results.append(result)
            except Exception as e:
                results.append({
                    "target_text": target_text,
                    "error": str(e),
                    "score": 0.0,
                })
        return results


# ────────────────────────────────────────────
# 4. CLI (독립 실행 테스트)
# ────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="발음 평가 테스트")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Whisper 모델 경로")
    parser.add_argument("--audio", type=str, required=True,
                        help="테스트 오디오 WAV 경로")
    parser.add_argument("--target", type=str, required=True,
                        help="목표 문장")
    args = parser.parse_args()

    evaluator = PronunciationEvaluator(args.model_path)
    result = evaluator.evaluate(args.audio, args.target)

    print(f"\n{'='*50}")
    print(f"  📋 발음 평가 결과")
    print(f"{'='*50}")
    print(f"  목표 문장:  {result['target_text']}")
    print(f"  기대 발음:  {result['expected_pronunciation']}")
    print(f"  실제 발음:  {result['actual_pronunciation']}")
    print(f"  발음 점수:  {result['score']:.1%}")
    print(f"  오류 수:    {result['error_count']}/{result['total_jamo']}")

    if result['errors']:
        print(f"\n  🔍 오류 상세:")
        for err in result['errors'][:10]:
            exp = err['expected'] or '∅'
            act = err['actual'] or '∅'
            print(f"     [{err['position']:2d}] {exp} → {act} ({err['status']})")
