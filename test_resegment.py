"""
기존 구음장애 세션 WAV → Whisper 타임스탬프 기반 재세그멘테이션 테스트
==================================================================
목적: 구음장애 데이터의 세션 WAV를 Whisper로 재세그멘테이션 가능한지 검증

원리:
  1. base Whisper가 세션 WAV를 처리 → 세그멘트별 타임스탬프 자동 생성
  2. 원본 Transcript에서 문장 분리 (마침표/b/ 기준)
  3. Whisper 세그멘트 ↔ 원본 문장 매칭 (fuzzy matching)
  4. 매칭된 타임스탬프로 정확한 세그멘트 절단
  
실행:
    CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python test_resegment.py
"""

import json
import os
import re
import torch
from pathlib import Path
from difflib import SequenceMatcher

torch.backends.cudnn.enabled = False

HOME = Path.home()
DATA_ROOT = HOME / "mingly_workspace" / "dataset" / "013.구음장애_음성인식_데이터" / "01.데이터" / "1.Training"
LABEL_DIR = DATA_ROOT / "라벨링데이터_250331_add"
WAV_DIR = DATA_ROOT / "원천데이터"


def find_first_pair():
    """첫 번째 JSON + WAV 쌍 찾기."""
    for subdir in sorted(LABEL_DIR.iterdir()):
        if not subdir.is_dir():
            continue
        for json_file in sorted(subdir.glob("*.json")):
            stem = json_file.stem
            # WAV 찾기 (같은 서브폴더 이름의 원천데이터)
            wav_candidates = list(WAV_DIR.rglob(f"{stem}.wav"))
            if wav_candidates:
                return json_file, wav_candidates[0]
    return None, None


def parse_transcript(transcript_text):
    """Transcript 텍스트에서 문장 분리."""
    # b/ 마커 제거
    text = re.sub(r'b/', '', transcript_text)
    # 연속 공백 정리
    text = re.sub(r'\s+', ' ', text).strip()
    # 마침표로 문장 분리
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip() and len(s.strip()) > 2]
    return sentences


def test_whisper_timestamps(wav_path, transcript_text):
    """Whisper로 세션 WAV 처리 → 타임스탬프 세그멘트 추출."""
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import librosa

    print(f"\n📥 Base Whisper 로드: openai/whisper-tiny")
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="ko", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print(f"🎵 WAV 로드: {wav_path}")
    audio, sr = librosa.load(str(wav_path), sr=16000, mono=True)
    total_duration = len(audio) / 16000
    print(f"   총 길이: {total_duration:.0f}초 ({total_duration/60:.1f}분)")

    # Whisper 30초 청크 단위 처리
    chunk_sec = 30
    chunk_samples = chunk_sec * 16000
    whisper_segments = []

    num_chunks = (len(audio) + chunk_samples - 1) // chunk_samples
    print(f"\n🔍 Whisper 처리: {num_chunks}개 청크 (각 {chunk_sec}초)")

    for i in range(min(num_chunks, 20)):  # 처음 20개 청크만 (10분)
        start = i * chunk_samples
        end = min(start + chunk_samples, len(audio))
        chunk = audio[start:end]

        if len(chunk) < 16000:  # 1초 미만 건너뛰기
            continue

        input_features = processor.feature_extractor(
            chunk, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)

        with torch.no_grad():
            pred_ids = model.generate(
                input_features,
                max_new_tokens=256,
                language="ko",
                task="transcribe",
                return_timestamps=True,
            )

        # 디코딩
        decoded = processor.tokenizer.decode(pred_ids[0], skip_special_tokens=False)
        text_only = processor.tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()

        offset = i * chunk_sec
        whisper_segments.append({
            "chunk_idx": i,
            "start_sec": offset,
            "end_sec": offset + chunk_sec,
            "text": text_only,
            "raw": decoded[:200],
        })

        if text_only:
            print(f"   [{i:3d}] {offset:5.0f}s~{offset+chunk_sec:5.0f}s: {text_only[:60]}...")

    return whisper_segments


def match_segments(whisper_segments, transcript_sentences):
    """Whisper 세그멘트와 원본 문장 fuzzy matching."""
    print(f"\n{'='*60}")
    print(f"  📊 매칭 분석")
    print(f"{'='*60}")
    print(f"  Whisper 세그멘트: {len(whisper_segments)}개")
    print(f"  원본 문장:        {len(transcript_sentences)}개")

    matched = 0
    for ws in whisper_segments[:10]:  # 처음 10개만
        if not ws["text"]:
            continue
        best_ratio = 0
        best_sent = ""
        for sent in transcript_sentences:
            ratio = SequenceMatcher(None, ws["text"][:30], sent[:30]).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_sent = sent

        status = "✅" if best_ratio > 0.5 else "❌"
        if best_ratio > 0.5:
            matched += 1
        print(f"\n  {status} [{ws['chunk_idx']:2d}] 유사도: {best_ratio:.2f}")
        print(f"      Whisper: {ws['text'][:50]}")
        print(f"      원본:    {best_sent[:50]}")

    print(f"\n  📊 매칭률: {matched}/{min(len(whisper_segments), 10)}")
    return matched


def main():
    print("🔍 재세그멘테이션 가능성 테스트")
    print("="*60)

    # 1. 데이터 찾기
    json_path, wav_path = find_first_pair()
    if not json_path:
        print("❌ JSON+WAV 쌍을 찾을 수 없습니다")
        return

    print(f"📂 JSON: {json_path.name}")
    print(f"🎵 WAV:  {wav_path.name}")

    # 2. Transcript 파싱
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    transcript = data.get("Transcript", "")
    sentences = parse_transcript(transcript)
    print(f"\n📝 원본 문장: {len(sentences)}개")
    for s in sentences[:5]:
        print(f"   - {s[:50]}")

    # 3. Whisper 타임스탬프 추출
    whisper_segments = test_whisper_timestamps(wav_path, transcript)

    # 4. 매칭 분석
    match_segments(whisper_segments, sentences)

    print(f"\n{'='*60}")
    print(f"  결론: 재세그멘테이션 가능 여부는 매칭률로 판단")
    print(f"  50% 이상 → 실용적 / 30% 미만 → 비실용적")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
