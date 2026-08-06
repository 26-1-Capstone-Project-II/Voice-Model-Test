"""
뉴스 대본 및 앵커 음성 데이터(AI Hub, dataSetSn=71557) → Whisper 학습 JSONL
=========================================================================
외래어/저빈도어 어휘 확장(v3.1)의 세 번째 축. TTS(melo)만으로는 재합성
변동성·품질 한계가 있고, KsponSpeech(자유대화)엔 IT/기술 어휘가 거의 안 나온다.
이 데이터는 아나운서(준비생 포함)가 실제 IT과학 뉴스 원고를 낭독한 진짜 사람
음성이라, 목표 어휘(외래어/전문용어)가 실제 문맥·자연스러운 발음으로 등장한다.

라벨 구조(라벨 JSON, script 필드):
  - script.text        : 발화 텍스트(표준 표기, 철자형) — G2P 입력
  - script.press_field  : 카테고리("IT과학", "정치", "사회" 등) — 이 스크립트로 필터링
  - file_information.audio_duration : 오디오 길이(초). wav 파일 자체가 이미
    발화 단위로 잘려 있어(1문장=1파일) utterance_start/end로 재슬라이싱 불필요
    (실측 확인: JSON 값과 wav 실제 길이가 소수점까지 일치).

원천/라벨 폴더 대응: <label_root>/SPK*/<id>/<name>.json ↔ <audio_root>/SPK*/<id>/<name>.wav
(같은 상대경로, 확장자만 다름)

경로 이식성: 이 스크립트는 Windows(원천 다운로드 위치)에서 실행하고,
결과(리샘플된 wav + jsonl)만 Linux 서버로 옮기는 걸 전제로 한다.
JSONL 의 wav_path 필드는 로컬 절대경로가 아니라 --wav_path_prefix 로
지정한 "서버에 옮겨질 최종 경로"를 기준으로 기록한다(옮긴 뒤 경로 재작성 불필요).

실행(Windows, Git Bash/WSL):
  python prepare_news_it.py \
      --label_root "138.뉴스대본_라벨" \
      --audio_root "138_뉴스대본_오디오" \
      --category IT과학 \
      --output_dir ./news_it_dataset \
      --wav_path_prefix /home/slim/mingly_workspace/Voice-Model-Test/news_it_dataset/wavs

이후 output_dir 전체(특히 wavs/ 와 *.jsonl)를 서버의
/home/slim/mingly_workspace/Voice-Model-Test/news_it_dataset 로 그대로 복사하면
wav_path 가 이미 맞아떨어진다.
"""

import os
import re
import json
import argparse
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

TARGET_SR = 16000
MIN_SEC = 0.5
MAX_SEC = 25.0


def load_g2p_optional():
    """g2pk 설치돼 있으면 사용, 없으면 None(라벨은 원문 그대로 저장 —
    finetune_whisper.py 가 --apply_g2p 로 어차피 transcript 에서 재생성한다)."""
    try:
        from g2pk import G2p
        g2p = G2p()
        print("✅ G2P: g2pk 사용")
        return lambda text: g2p(text, descriptive=True)
    except Exception:
        print("⏭️  g2pk 없음 — label 은 원문 그대로 저장(학습 시 --apply_g2p 로 재생성됨)")
        return None


def collect_pairs(label_root, audio_root, categories):
    """카테고리(press_field)로 필터링한 (json_path, wav_path) 쌍 목록."""
    label_root = Path(label_root)
    audio_root = Path(audio_root)
    pairs = []
    skipped_category = 0
    skipped_no_audio = 0
    for json_path in label_root.rglob("*.json"):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        script = data.get("script", {})
        if script.get("press_field") not in categories:
            skipped_category += 1
            continue
        rel = json_path.relative_to(label_root).with_suffix(".wav")
        wav_path = audio_root / rel
        if not wav_path.exists():
            skipped_no_audio += 1
            continue
        pairs.append((wav_path, script))
    print(f"  📂 라벨 스캔 완료: 매칭 {len(pairs):,}개 "
          f"(카테고리 불일치 {skipped_category:,}, 오디오 없음 {skipped_no_audio:,})")
    return pairs


def main():
    ap = argparse.ArgumentParser(description="뉴스 대본/앵커 음성 데이터(카테고리 필터) → 학습 JSONL")
    ap.add_argument("--label_root", required=True, help="라벨 JSON 루트(압축 해제한 138.*_라벨)")
    ap.add_argument("--audio_root", required=True, help="원천 wav 루트(압축 해제한 138_*_오디오)")
    ap.add_argument("--category", default="IT과학",
                     help="press_field 필터, 콤마로 여러 개 가능 (예: IT과학,경제)")
    ap.add_argument("--output_dir", default="./news_it_dataset")
    ap.add_argument("--wav_path_prefix", default=None,
                     help="JSONL 에 기록할 wav 경로 접두사(서버 최종 경로). "
                          "미지정 시 output_dir/wavs 의 로컬 절대경로를 그대로 씀"
                          "(같은 머신에서 바로 쓸 때만 유효 — 다른 머신으로 옮기면 --wav_path_prefix 필수).")
    ap.add_argument("--val_ratio", type=float, default=0.05)
    ap.add_argument("--test_ratio", type=float, default=0.0,
                     help="§9 회귀 기준선은 Zeroth 가 담당하므로 기본 0(전부 train/validation).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    categories = {c.strip() for c in args.category.split(",") if c.strip()}
    out_dir = Path(args.output_dir)
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    wav_path_prefix = args.wav_path_prefix.rstrip("/\\") if args.wav_path_prefix else str(wav_dir.resolve())

    print(f"🔎 라벨 스캔 중... (카테고리: {', '.join(sorted(categories))})")
    pairs = collect_pairs(args.label_root, args.audio_root, categories)
    if not pairs:
        raise SystemExit("❌ 매칭되는 항목이 0개 — label_root/audio_root/category 확인 필요")

    g2p = load_g2p_optional()

    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n = len(pairs)
    n_test = int(n * args.test_ratio)
    n_val = int(n * args.val_ratio)

    writers = {sp: open(out_dir / f"{sp}.jsonl", "w", encoding="utf-8")
               for sp in ("train", "validation", "test")}
    counts = {sp: 0 for sp in writers}
    skipped_short = skipped_long = skipped_load_fail = 0

    try:
        for i, (wav_path, script) in enumerate(tqdm(pairs, desc="변환")):
            sp = "test" if i < n_test else "validation" if i < n_test + n_val else "train"

            text = (script.get("text") or "").strip()
            if not text:
                continue

            try:
                audio, sr = sf.read(str(wav_path))
            except Exception:
                skipped_load_fail += 1
                continue
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != TARGET_SR:
                audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
            duration = len(audio) / TARGET_SR
            if duration < MIN_SEC:
                skipped_short += 1
                continue
            if duration > MAX_SEC:
                skipped_long += 1
                continue

            out_name = f"{sp}_{i:06d}.wav"
            sf.write(str(wav_dir / out_name), audio.astype(np.float32), TARGET_SR)

            label = g2p(text).strip() if g2p else text
            obj = {
                "wav_path": f"{wav_path_prefix}/{out_name}",
                "transcript": text,
                "label": label,
                "duration": float(duration),
                "source": "news_it",
                "press_field": script.get("press_field"),
            }
            writers[sp].write(json.dumps(obj, ensure_ascii=False) + "\n")
            counts[sp] += 1
    finally:
        for w in writers.values():
            w.close()

    total = sum(counts.values())
    if total == 0:
        raise SystemExit("❌ 변환 결과 0개 — 오디오 로드/길이 필터 확인 필요")

    print(f"\n✅ 완료 → {out_dir}")
    for sp in ("train", "validation", "test"):
        print(f"   {sp:11s}: {counts[sp]:,}개")
    if skipped_short or skipped_long or skipped_load_fail:
        print(f"   🧹 스킵 — 너무짧음:{skipped_short:,} 너무김:{skipped_long:,} 로드실패:{skipped_load_fail:,}")
    print(f"\nwav_path 접두사: {wav_path_prefix}")
    print(f"다음: {out_dir} 전체(wavs/ + *.jsonl)를 서버의 위 경로로 복사한 뒤,")
    print(f"      finetune_whisper.py --extra_json_dirs 에 이 디렉터리를 추가하세요.")


if __name__ == "__main__":
    main()