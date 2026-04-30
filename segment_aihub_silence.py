"""
AIHub 구음장애 — 침묵 기반 VAD 세그멘테이션 (라벨 정렬 보장)
==============================================================
기존 vad_segment.py는 음성 구간을 첫/끝 시점만 추출한 뒤
n_sentences개로 *시간 균등 분할* 했기 때문에 segment[i] ≠ sentence[i] 였음.

본 스크립트는:
  1) silero-VAD를 *문장 경계 친화적* 파라미터로 실행 → 실제 침묵으로
     구분된 N개 발화 구간을 얻는다 (균등 분할 X)
  2) Transcript를 문장 부호로 분리 → M개 문장
  3) **품질 게이트**: |N - M| / M < tol 인 세션만 사용
     (tol 초과 시 정렬 정확도 보장 불가 → 세션 통째로 스킵)
  4) 통과한 세션은 순서대로 1:1 매핑 → wav 잘라 저장
  5) 화자 단위 train/val/test 분할 후 JSONL 출력

실행:
    python segment_aihub_silence.py \\
        --wav_dir  "/hdd/.../원천데이터/TS02_언어청각장애" \\
        --json_dir "/hdd/.../라벨링데이터_250331_add" \\
        --output_dir ./segmented_dataset_v2 \\
        --tol 0.10 \\
        --min_silence_ms 800
"""

import json
import re
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import torch
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

from korean_g2p_nomecab import load_g2p

TARGET_SR = 16000
MIN_SEG_SEC = 0.5
MAX_SEG_SEC = 20.0


# ────────────────────────────────────────────
# VAD 로드
# ────────────────────────────────────────────
def load_vad():
    print("📥 Silero-VAD 로딩...")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        trust_repo=True,
    )
    get_speech_ts = utils[0]
    print("✅ VAD 로드 완료")
    return model, get_speech_ts


# ────────────────────────────────────────────
# Transcript 문장 분리
# ────────────────────────────────────────────
def split_transcript(transcript: str) -> list[str]:
    """문장 부호 기준 분리 + 노이즈 마커 제거."""
    transcript = re.sub(r"\s*b/\s*", " ", transcript)
    transcript = re.sub(r"\s*[a-z]/\s*", " ", transcript)
    transcript = transcript.strip()
    sentences = re.split(r"[.?!。]\s*", transcript)
    sentences = [s.strip() for s in sentences if len(s.strip()) >= 2]
    return sentences


# ────────────────────────────────────────────
# 침묵 기반 발화 구간 추출
# ────────────────────────────────────────────
def get_speech_regions(audio_tensor, vad_model, get_speech_ts,
                       min_speech_ms=300, min_silence_ms=800, threshold=0.5):
    """침묵을 *문장 경계급*으로 잡아 N개의 발화 구간을 그대로 반환.

    기존 균등 분할과 달리, 실제 VAD가 감지한 [start, end]를 그대로 사용.
    """
    speech_ts = get_speech_ts(
        audio_tensor,
        vad_model,
        sampling_rate=TARGET_SR,
        min_speech_duration_ms=min_speech_ms,
        min_silence_duration_ms=min_silence_ms,
        threshold=threshold,
    )
    regions = []
    for ts in speech_ts:
        start = ts["start"] / TARGET_SR
        end = ts["end"] / TARGET_SR
        dur = end - start
        regions.append({"start": start, "end": end, "duration": dur})
    return regions


# ────────────────────────────────────────────
# 세션 처리
# ────────────────────────────────────────────
def process_session(json_path, wav_index, vad_model, get_speech_ts, g2p,
                    out_wav_dir, tol, min_silence_ms, stats):
    """단일 세션 처리. 품질 게이트 통과 시 segment 리스트 반환."""
    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except Exception:
        stats["read_error"] += 1
        return []

    transcript = (data.get("Transcript") or "").strip()
    if not transcript:
        stats["no_transcript"] += 1
        return []

    wav_stem = Path(data.get("File_id", "")).stem
    wav_path = wav_index.get(wav_stem)
    if wav_path is None:
        stats["no_wav"] += 1
        return []

    sentences = split_transcript(transcript)
    if not sentences:
        stats["no_sentences"] += 1
        return []

    # 오디오 로드
    try:
        audio, _ = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
    except Exception:
        stats["load_error"] += 1
        return []

    # VAD: 실제 침묵 기반 발화 구간 (균등 분할 X)
    audio_tensor = torch.from_numpy(audio)
    try:
        regions = get_speech_regions(
            audio_tensor, vad_model, get_speech_ts,
            min_silence_ms=min_silence_ms,
        )
    except Exception:
        stats["vad_error"] += 1
        return []

    if not regions:
        stats["no_regions"] += 1
        return []

    n_reg = len(regions)
    n_sent = len(sentences)

    # 품질 게이트: 발화 구간 수 ≈ 문장 수
    diff_ratio = abs(n_reg - n_sent) / max(n_sent, 1)
    if diff_ratio > tol:
        stats["count_mismatch"] += 1
        stats[f"mismatch_n{n_reg}_m{n_sent}"] += 1
        return []

    # 순서대로 매핑 (둘 중 작은 쪽 길이까지)
    n = min(n_reg, n_sent)
    patient = data.get("Patient_info", {})
    disease = data.get("Disease_info", {})

    out_pairs = []
    for i in range(n):
        reg = regions[i]
        sent = sentences[i]

        # 너무 짧거나 긴 세그먼트 제외
        if reg["duration"] < MIN_SEG_SEC or reg["duration"] > MAX_SEG_SEC:
            stats["bad_duration"] += 1
            continue

        # WAV 슬라이스
        start_s = int(reg["start"] * TARGET_SR)
        end_s = int(reg["end"] * TARGET_SR)
        seg_audio = audio[start_s:end_s]

        seg_name = f"{wav_stem}_seg{i:04d}.wav"
        seg_path = out_wav_dir / seg_name
        if not seg_path.exists():
            sf.write(str(seg_path), seg_audio, TARGET_SR)

        # G2P 라벨 생성
        try:
            label = g2p(sent, descriptive=True).strip()
        except Exception:
            label = sent

        out_pairs.append({
            "wav_path": str(seg_path),
            "transcript": sent,
            "label": label,
            "duration": reg["duration"],
            "speaker_id": wav_stem,
            "segment_idx": i,
            "n_regions": n_reg,
            "n_sentences": n_sent,
            "disease_type": disease.get("Type", ""),
            "subcategory": disease.get("Subcategory2", ""),
            "sex": patient.get("Sex", ""),
            "age": patient.get("Age", ""),
        })

    if out_pairs:
        stats["passed_sessions"] += 1
        stats["total_segments"] += len(out_pairs)

    return out_pairs


# ────────────────────────────────────────────
# Train/Val/Test 분할 (화자 단위)
# ────────────────────────────────────────────
def save_splits(pairs, output_dir):
    import random
    random.seed(42)

    by_speaker = defaultdict(list)
    for p in pairs:
        by_speaker[p["speaker_id"]].append(p)

    speakers = list(by_speaker.keys())
    random.shuffle(speakers)
    n = len(speakers)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train_sp = set(speakers[:n_train])
    val_sp = set(speakers[n_train:n_train + n_val])

    splits = {"train": [], "validation": [], "test": []}
    for sp, sp_pairs in by_speaker.items():
        key = ("train" if sp in train_sp
               else "validation" if sp in val_sp else "test")
        splits[key].extend(sp_pairs)

    print(f"\n✂️  화자 단위 분할:")
    for name, data in splits.items():
        out_path = Path(output_dir) / f"{name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {name:12s}: {len(data):,}개 → {out_path.name}")


# ────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="AIHub 구음장애 — 침묵 기반 VAD 세그멘테이션 (라벨 정렬 보장)"
    )
    parser.add_argument("--wav_dir", required=True, help="원천 WAV 디렉토리")
    parser.add_argument("--json_dir", required=True, help="라벨 JSON 디렉토리")
    parser.add_argument("--output_dir", default="./segmented_dataset_v2")
    parser.add_argument("--max_files", type=int, default=0,
                        help="처리할 세션 수 제한 (0=전체)")
    parser.add_argument("--tol", type=float, default=0.10,
                        help="품질 게이트: |N-M|/M 허용 오차 (기본 10%)")
    parser.add_argument("--min_silence_ms", type=int, default=800,
                        help="문장 경계로 인정할 최소 침묵 길이 (ms)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_wav_dir = output_dir / "wavs"
    out_wav_dir.mkdir(exist_ok=True)

    # G2P + VAD 로드
    print("🔤 G2P 로딩...")
    g2p = load_g2p()
    vad_model, get_speech_ts = load_vad()

    # WAV/JSON 인덱스
    print(f"\n📂 WAV 인덱싱: {args.wav_dir}")
    wav_index = {f.stem: f for f in Path(args.wav_dir).rglob("*.wav")}
    print(f"   WAV: {len(wav_index):,}개")

    print(f"📂 JSON 인덱싱: {args.json_dir}")
    json_files = sorted(Path(args.json_dir).rglob("*.json"))
    if args.max_files > 0:
        json_files = json_files[:args.max_files]
    print(f"   JSON: {len(json_files):,}개")

    # 처리
    print(f"\n🚀 세그멘테이션 시작 (tol={args.tol}, "
          f"min_silence={args.min_silence_ms}ms)")
    all_pairs = []
    stats = Counter()
    for jp in tqdm(json_files, desc="세션"):
        pairs = process_session(
            jp, wav_index, vad_model, get_speech_ts, g2p,
            out_wav_dir, args.tol, args.min_silence_ms, stats,
        )
        all_pairs.extend(pairs)

    # 통계
    print(f"\n{'='*60}")
    print(f"  세그멘테이션 완료")
    print(f"{'='*60}")
    n_pass = stats["passed_sessions"]
    n_total = len(json_files)
    n_seg = stats["total_segments"]
    print(f"  통과 세션:     {n_pass:,} / {n_total:,} "
          f"({100*n_pass/max(n_total,1):.1f}%)")
    print(f"  생성 세그먼트: {n_seg:,}개")
    if n_pass:
        print(f"  평균 세그먼트/세션: {n_seg/n_pass:.1f}개")

    print(f"\n  탈락 사유:")
    for k, v in stats.most_common():
        if k.startswith("mismatch_n") or k in ("passed_sessions", "total_segments"):
            continue
        if v > 0:
            print(f"    {k}: {v:,}")

    if all_pairs:
        durs = [p["duration"] for p in all_pairs]
        print(f"\n  세그먼트 길이:")
        print(f"    평균 {np.mean(durs):.2f}s  "
              f"중간값 {np.median(durs):.2f}s  "
              f"최소 {np.min(durs):.2f}s  "
              f"최대 {np.max(durs):.2f}s")

        save_splits(all_pairs, output_dir)
    else:
        print("\n⚠️ 통과한 세션이 없습니다. --tol 을 완화하거나 "
              "--min_silence_ms 를 조정해보세요.")


if __name__ == "__main__":
    main()