"""
Whisper 발음 전사 테스트
========================
gTTS로 테스트 오디오를 생성한 후, Whisper 모델의 발음 전사 정확도를 검증.

실행:
    CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python test_whisper_phonetic.py \\
        --model_path best_model_whisper/best

기대 결과:
    - 정상 발화(gTTS): CER < 0.2
    - 발음 전사 정확도 확인
    - 자모 레벨 오류 감지 검증
"""

import os
import sys
import torch
import tempfile
import argparse
from pathlib import Path

torch.backends.cudnn.enabled = False

HOME = Path.home()
DEFAULT_MODEL = HOME / "mingly_workspace" / "Voice-Model-Test" / "best_model_whisper" / "best"


def generate_test_audio(text, output_path):
    """gTTS로 테스트 오디오 생성."""
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang='ko')
        tts.save(output_path)
        return True
    except Exception as e:
        print(f"  ⚠️ gTTS 실패: {e}")
        return False


def test_transcription(evaluator, test_cases):
    """발음 전사 정확도 테스트."""
    print(f"\n{'='*60}")
    print(f"  🔬 발음 전사 테스트 ({len(test_cases)}개 문장)")
    print(f"{'='*60}")

    results = []
    tmp_dir = tempfile.mkdtemp()

    for i, (text, expected_pron) in enumerate(test_cases):
        wav_path = os.path.join(tmp_dir, f"test_{i}.wav")

        # mp3 → wav 변환
        mp3_path = os.path.join(tmp_dir, f"test_{i}.mp3")
        if not generate_test_audio(text, mp3_path):
            continue

        # mp3 → wav
        import librosa
        import soundfile as sf
        audio, sr = librosa.load(mp3_path, sr=16000, mono=True)
        sf.write(wav_path, audio, 16000)

        # 전사
        actual_pron = evaluator.transcribe(wav_path)

        # 결과
        match = "✅" if actual_pron.strip() == expected_pron.strip() else "❌"
        print(f"\n  [{i+1}] {match}")
        print(f"      원문:     {text}")
        print(f"      기대발음: {expected_pron}")
        print(f"      실제출력: {actual_pron}")

        results.append({
            "text": text,
            "expected": expected_pron,
            "actual": actual_pron,
            "match": actual_pron.strip() == expected_pron.strip(),
        })

    # 정확도
    correct = sum(1 for r in results if r["match"])
    print(f"\n  📊 전사 정확도: {correct}/{len(results)}")

    return results


def test_pronunciation_scoring(evaluator, test_cases):
    """발음 평가 점수 테스트."""
    print(f"\n{'='*60}")
    print(f"  🎯 발음 평가 테스트 ({len(test_cases)}개)")
    print(f"{'='*60}")

    tmp_dir = tempfile.mkdtemp()
    results = []

    for i, (text, _) in enumerate(test_cases):
        wav_path = os.path.join(tmp_dir, f"score_{i}.wav")

        mp3_path = os.path.join(tmp_dir, f"score_{i}.mp3")
        if not generate_test_audio(text, mp3_path):
            continue

        import librosa
        import soundfile as sf
        audio, sr = librosa.load(mp3_path, sr=16000, mono=True)
        sf.write(wav_path, audio, 16000)

        # 평가
        result = evaluator.evaluate(wav_path, text)

        print(f"\n  [{i+1}] 점수: {result['score']:.1%}  CER: {result['cer']:.3f}")
        print(f"      목표:   {result['target_text']}")
        print(f"      기대:   {result['expected_pronunciation']}")
        print(f"      실제:   {result['actual_pronunciation']}")
        if result['errors']:
            errors_str = ", ".join(
                f"{e['expected'] or '∅'}→{e['actual'] or '∅'}"
                for e in result['errors'][:5]
            )
            print(f"      오류:   {errors_str}")

        results.append(result)

    # 평균 점수
    if results:
        avg_score = sum(r['score'] for r in results) / len(results)
        avg_cer = sum(r['cer'] for r in results) / len(results)
        print(f"\n  📊 평균 점수: {avg_score:.1%}  평균 CER: {avg_cer:.3f}")

    return results


def test_baseline_comparison(evaluator, test_cases):
    """파인튜닝 모델 vs 베이스라인(원본 Whisper) 비교."""
    print(f"\n{'='*60}")
    print(f"  📊 베이스라인 비교 (원본 Whisper tiny vs 파인튜닝)")
    print(f"{'='*60}")

    try:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        # 원본 Whisper 로드
        print("  📥 원본 모델 로드: openai/whisper-tiny")
        base_processor = WhisperProcessor.from_pretrained(
            "openai/whisper-tiny", language="ko", task="transcribe"
        )
        base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        base_model.to(device)
        base_model.eval()

    except Exception as e:
        print(f"  ⚠️ 베이스라인 로드 실패: {e}")
        return

    tmp_dir = tempfile.mkdtemp()
    import librosa

    for i, (text, expected_pron) in enumerate(test_cases[:3]):  # 3개만
        mp3_path = os.path.join(tmp_dir, f"baseline_{i}.mp3")
        wav_path = os.path.join(tmp_dir, f"baseline_{i}.wav")

        if not generate_test_audio(text, mp3_path):
            continue

        audio, _ = librosa.load(mp3_path, sr=16000, mono=True)
        import soundfile as sf
        sf.write(wav_path, audio, 16000)

        # 원본 Whisper 전사
        input_features = base_processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)

        with torch.no_grad():
            pred_ids = base_model.generate(
                input_features, max_new_tokens=256,
                language="ko", task="transcribe",
            )
        base_output = base_processor.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )[0].strip()

        # 파인튜닝 모델 전사
        finetuned_output = evaluator.transcribe(wav_path)

        print(f"\n  [{i+1}] 원문: {text}")
        print(f"      기대 발음: {expected_pron}")
        print(f"      원본 출력: {base_output}")
        print(f"      파인튜닝:  {finetuned_output}")

        # 원본은 맞춤법 교정, 파인튜닝은 발음 전사
        if base_output.strip() == text.strip():
            print(f"      → 원본: 맞춤법 교정 ✅ (예상대로)")
        if finetuned_output.strip() == expected_pron.strip():
            print(f"      → 파인튜닝: 발음 전사 ✅")


def main():
    parser = argparse.ArgumentParser(description="Whisper 발음 전사 테스트")
    parser.add_argument("--model_path", type=str, default=str(DEFAULT_MODEL),
                        help="파인튜닝된 Whisper 모델 경로")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="베이스라인 비교 건너뛰기")
    args = parser.parse_args()

    # 테스트 케이스: (원문, 기대 발음)
    test_cases = [
        ("같이 먹을까",         "가치 머글까"),
        ("좋네요",             "존네요"),
        ("안녕하세요",         "안녕하세요"),
        ("먹어요",             "머거요"),
        ("닭볶음",             "닥뽀끔"),
        ("천천히 말해주세요",   "천천히 말해주세요"),
        ("오늘 날씨가 좋네요",  "오늘 날씨가 존네요"),
        ("맛있는 음식",        "마신는 음식"),
        ("국물이 끓고 있어요",  "궁무리 끌코 이써요"),
        ("학교에 갑니다",      "학꾜에 감니다"),
    ]

    # 모델 로드
    from pronunciation_evaluator import PronunciationEvaluator
    evaluator = PronunciationEvaluator(args.model_path)

    # 테스트 실행
    test_transcription(evaluator, test_cases)
    test_pronunciation_scoring(evaluator, test_cases)

    if not args.skip_baseline:
        test_baseline_comparison(evaluator, test_cases)

    print(f"\n{'='*60}")
    print(f"  ✅ 테스트 완료!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
