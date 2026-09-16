"""
STEP VAD: 긴 WAV 세션 녹음 → 문장 단위 세그멘트 분리
======================================================
Silero-VAD로 음성 구간을 감지하고, 문장 단위로 WAV를 분리합니다.
분리된 세그멘트를 Transcript 문장과 순서대로 매핑합니다.

설치:
    pip install silero-vad pydub

실행:
    python vad_segment.py \
        --wav_dir "C:\\...\\TS02_언어청각장애" \
        --json_dir "C:\\...\\TL02_언어청각장애" \
        --output_dir "./segmented_dataset"
"""

import json
import re
import argparse
from pathlib import Path
from collections import Counter

import torch
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from korean_g2p_nomecab import load_g2p


TARGET_SR  = 16000
MIN_SEG_SEC = 0.5   # 0.5초 미만 세그멘트 제외
MAX_SEG_SEC = 15.0  # 15초 초과 세그멘트 제외


# ────────────────────────────────────────────
# 1. Silero-VAD 로드
# ────────────────────────────────────────────

def load_vad():
    print("📥 Silero-VAD 로딩...")
    model, utils = torch.hub.load(
        repo_or_dir = "snakers4/silero-vad",
        model       = "silero_vad",
        force_reload = False,
        trust_repo  = True,
    )
    get_speech_ts = utils[0]
    print("✅ VAD 로드 완료")
    return model, get_speech_ts


# ────────────────────────────────────────────
# 2. WAV → 음성 구간 타임스탬프 추출
# ────────────────────────────────────────────

def get_speech_segments(wav_path: Path, vad_model, get_speech_ts, n_sentences: int) -> tuple:
    """
    VAD + 균등 분할 하이브리드:
    1. 먼저 VAD로 전체 음성 구간 감지
    2. 감지된 음성 구간을 n_sentences개로 균등 병합
    → 세그멘트 수가 문장 수와 정확히 일치
    """
    audio, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
    audio_tensor = torch.from_numpy(audio)

    # VAD로 음성 구간 감지 (느슨한 파라미터)
    speech_timestamps = get_speech_ts(
        audio_tensor,
        vad_model,
        sampling_rate           = TARGET_SR,
        min_speech_duration_ms  = 200,
        min_silence_duration_ms = 300,
        threshold               = 0.4,
    )

    if not speech_timestamps:
        return [], audio

    # 전체 음성 시작~끝 구간
    total_start = speech_timestamps[0]["start"] / TARGET_SR
    total_end   = speech_timestamps[-1]["end"]   / TARGET_SR
    total_dur   = total_end - total_start

    # n_sentences개로 균등 분할
    seg_dur = total_dur / n_sentences
    segments = []
    for i in range(n_sentences):
        start = total_start + i * seg_dur
        end   = total_start + (i + 1) * seg_dur
        dur   = end - start

        # 너무 짧거나 긴 구간 스킵
        if dur < 0.3:
            continue

        segments.append({
            "start"   : start,
            "end"     : end,
            "duration": dur,
        })

    return segments, audio

# ────────────────────────────────────────────
# 3. Transcript → 문장 단위 분리
# ────────────────────────────────────────────

def split_transcript(transcript: str) -> list[str]:
    """
    Transcript를 문장 단위로 분리.
    - b/ 숨소리·노이즈 마커 제거
    - '.' '?' '!' 기준 분리
    - 빈 문장 및 너무 짧은 문장 제외
    """
    # b/ 마커 제거 (숨소리, 노이즈, 발화 외 구간 표시)
    transcript = re.sub(r'\s*b/\s*', ' ', transcript)
    # 기타 노이즈 마커 제거 (n/, l/, u/ 등)
    transcript = re.sub(r'\s*[a-z]/\s*', ' ', transcript)
    transcript = transcript.strip()

    # 문장 부호 기준 분리
    sentences = re.split(r"[.?!。]\s*", transcript)
    # 빈 문장 및 2음절 미만 제외
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 2]
    return sentences


# ────────────────────────────────────────────
# 4. 세그멘트 ↔ 문장 순서 매핑
# ────────────────────────────────────────────

def map_segments_to_sentences(
    segments  : list[dict],
    sentences : list[str],
) -> list[dict]:
    """
    VAD 세그멘트와 Transcript 문장을 순서대로 매핑.
    세그멘트 수와 문장 수가 다를 경우:
    - 세그멘트 > 문장: 앞에서부터 문장 수만큼만 사용
    - 세그멘트 < 문장: 세그멘트 수만큼만 사용
    """
    n = min(len(segments), len(sentences))
    pairs = []
    for i in range(n):
        pairs.append({
            "segment" : segments[i],
            "sentence": sentences[i],
        })
    return pairs


# ────────────────────────────────────────────
# 5. 세그멘트 WAV 저장
# ────────────────────────────────────────────

def save_segment(
    audio     : np.ndarray,
    start_sec : float,
    end_sec   : float,
    out_path  : Path,
):
    start_sample = int(start_sec * TARGET_SR)
    end_sample   = int(end_sec   * TARGET_SR)
    segment_audio = audio[start_sample:end_sample]
    sf.write(str(out_path), segment_audio, TARGET_SR)


# ────────────────────────────────────────────
# 6. 전체 처리
# ────────────────────────────────────────────

def process_all(
    wav_dir    : Path,
    json_dir   : Path,
    output_dir : Path,
    g2p,
    max_files  : int = 0,  # 0=전체
):
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir_out = output_dir / "wavs"
    wav_dir_out.mkdir(exist_ok=True)

    # VAD 로드
    vad_model, get_speech_ts = load_vad()

    # WAV ↔ JSON 매칭
    wav_index  = {f.stem: f for f in wav_dir.rglob("*.wav")}
    json_files = sorted(json_dir.rglob("*.json"))

    if max_files > 0:
        json_files = json_files[:max_files]

    print(f"\n📂 처리 대상: {len(json_files):,}개 세션")

    all_pairs = []
    stats     = Counter()

    for jp in tqdm(json_files, desc="VAD 세그멘테이션"):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception:
            stats["read_error"] += 1
            continue

        transcript = data.get("Transcript", "").strip()
        if not transcript:
            stats["no_transcript"] += 1
            continue

        wav_stem = Path(data.get("File_id", "")).stem
        wav_path = wav_index.get(wav_stem)
        if wav_path is None:
            stats["no_wav"] += 1
            continue

        # 문장 분리
        sentences = split_transcript(transcript)
        if not sentences:
            stats["no_sentences"] += 1
            continue

        # VAD + 균등 분할: 문장 수에 맞게 세그멘트 생성
        try:
            segments, audio = get_speech_segments(
                wav_path, vad_model, get_speech_ts, n_sentences=len(sentences)
            )
        except Exception as e:
            stats["vad_error"] += 1
            continue

        if not segments:
            stats["no_segments"] += 1
            continue

        # 세그멘트 ↔ 문장 매핑
        pairs = map_segments_to_sentences(segments, sentences)
        if not pairs:
            stats["no_pairs"] += 1
            continue

        # 세그멘트 WAV 저장 + 메타데이터 수집
        patient = data.get("Patient_info", {})
        disease = data.get("Disease_info", {})

        for i, pair in enumerate(pairs):
            seg      = pair["segment"]
            sentence = pair["sentence"]

            # G2P 변환
            try:
                label = g2p(sentence, descriptive=True).strip()
            except Exception:
                label = sentence

            # WAV 파일 저장
            seg_name = f"{wav_stem}_seg{i:04d}.wav"
            seg_path = wav_dir_out / seg_name

            if not seg_path.exists():  # 이어받기
                save_segment(audio, seg["start"], seg["end"], seg_path)

            all_pairs.append({
                "wav_path"      : str(seg_path),
                "transcript"    : sentence,
                "label"         : label,
                "duration"      : seg["duration"],
                "speaker_id"    : wav_stem,
                "segment_idx"   : i,
                "disease_type"  : disease.get("Type", ""),
                "subcategory"   : disease.get("Subcategory1", ""),
                "sex"           : patient.get("Sex", ""),
                "age"           : patient.get("Age", ""),
            })

        stats["processed"] += 1
        stats["total_segments"] += len(pairs)

    # 결과 저장
    _save_results(all_pairs, output_dir)

    # 통계 출력
    print(f"\n{'='*50}")
    print(f"  VAD 세그멘테이션 완료")
    print(f"{'='*50}")
    print(f"  처리된 세션:     {stats['processed']:,}개")
    print(f"  생성된 세그멘트: {stats['total_segments']:,}개")
    print(f"  평균 세그멘트/세션: {stats['total_segments'] / max(stats['processed'], 1):.1f}개")
    for k, v in stats.items():
        if k not in ("processed", "total_segments") and v > 0:
            print(f"  ⚠️  {k}: {v:,}개")

    if all_pairs:
        durations = [p["duration"] for p in all_pairs]
        print(f"\n  세그멘트 길이 분포:")
        print(f"    평균: {np.mean(durations):.1f}초")
        print(f"    최소: {np.min(durations):.1f}초")
        print(f"    최대: {np.max(durations):.1f}초")

    return all_pairs


def _save_results(pairs: list[dict], output_dir: Path):
    """train/val/test 분할 후 JSONL 저장"""
    import random
    from collections import defaultdict

    random.seed(42)
    speaker_map = defaultdict(list)
    for p in pairs:
        speaker_map[p["speaker_id"]].append(p)

    speakers = list(speaker_map.keys())
    random.shuffle(speakers)
    n = len(speakers)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)
    train_sp = set(speakers[:n_train])
    val_sp   = set(speakers[n_train:n_train + n_val])

    split = {"train": [], "validation": [], "test": []}
    for sp, sp_pairs in speaker_map.items():
        key = "train" if sp in train_sp else ("validation" if sp in val_sp else "test")
        split[key].extend(sp_pairs)

    print(f"\n✂️  화자 단위 분할:")
    for name, data in split.items():
        out_path = output_dir / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {name:12s}: {len(data):,}개 → {out_path.name}")


# ────────────────────────────────────────────
# 7. 메인
# ────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wav_dir",
        default=r"C:\Users\User\Voice-Model-Test\구음장애 음성인식 데이터\01.데이터\1.Training\원천데이터\TS02_언어청각장애",
    )
    parser.add_argument(
        "--json_dir",
        default=r"C:\Users\User\Voice-Model-Test\구음장애 음성인식 데이터\01.데이터\1.Training\라벨링데이터_250331_add\TL02_언어청각장애",
    )
    parser.add_argument("--output_dir", default="./segmented_dataset")
    parser.add_argument("--max_files",  type=int, default=0,
                        help="테스트용: 처음 N개 세션만 처리 (0=전체)")
    args = parser.parse_args()

    print("🔤 G2P 로딩...")
    g2p = load_g2p()

    process_all(
        wav_dir    = Path(args.wav_dir),
        json_dir   = Path(args.json_dir),
        output_dir = Path(args.output_dir),
        g2p        = g2p,
        max_files  = args.max_files,
    )

    print("\n✅ 완료! 다음 단계:")
    print("   python finetune_simple.py \\")
    print("     --wav_dir ./segmented_dataset/wavs \\")
    print("     --json_dir ./segmented_dataset")
    print("   (finetune_simple.py의 DATA_ROOT 경로도 수정 필요)")
