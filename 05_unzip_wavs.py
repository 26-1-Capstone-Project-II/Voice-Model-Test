"""
WAV 원천데이터 전체 압축 해제 스크립트
========================================
- 진행률 실시간 표시
- 이미 해제된 파일은 스킵 (이어받기 지원)
- 압축 해제 후 JSON과 WAV 매칭 검증

실행:
    python 05_unzip_wavs.py --data_root "C:\\Users\\User\\Voice-Model-Test\\구음장애 음성인식 데이터"
"""

import zipfile
import argparse
import time
from pathlib import Path


SOURCE_KEYWORDS = ["TS", "VS", "원천데이터"]


def is_source_zip(path: Path) -> bool:
    name = path.name
    for kw in SOURCE_KEYWORDS:
        if kw in name:
            return True
    return False


def unzip_with_progress(zip_path: Path, out_dir: Path):
    """압축 해제 + 실시간 진행률 출력"""
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        total = len(members)

        # 이미 해제된 파일 스킵
        to_extract = []
        for m in members:
            target = out_dir / m
            if not target.exists():
                to_extract.append(m)

        skipped = total - len(to_extract)
        if skipped > 0:
            print(f"     ↩️  이미 해제된 파일 {skipped:,}개 스킵")

        if not to_extract:
            print(f"     ✅ 이미 완료됨\n")
            return

        print(f"     총 {len(to_extract):,}개 파일 해제 시작...")
        start = time.time()

        for i, member in enumerate(to_extract, 1):
            zf.extract(member, out_dir)

            # 100개마다 진행률 출력
            if i % 100 == 0 or i == len(to_extract):
                elapsed = time.time() - start
                rate    = i / elapsed if elapsed > 0 else 0
                eta     = (len(to_extract) - i) / rate if rate > 0 else 0
                pct     = i / len(to_extract) * 100
                bar     = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                print(f"     [{bar}] {pct:5.1f}%  {i:,}/{len(to_extract):,}  "
                      f"ETA: {eta/60:.1f}분", end="\r")

        print(f"\n     ✅ 완료 ({time.time()-start:.0f}초)\n")


def verify_wav_json_pairs(data_root: Path):
    """해제 후 WAV ↔ JSON 매칭 검증"""
    print("\n🔍 WAV ↔ JSON 매칭 검증 중...")

    wav_files  = {f.stem: f for f in data_root.rglob("*.wav")}
    json_files = {f.stem: f for f in data_root.rglob("*.json")}

    # File_id는 .wav 확장자 포함이므로 stem 기준 매칭
    matched   = set(wav_files.keys()) & set(json_files.keys())
    wav_only  = set(wav_files.keys()) - set(json_files.keys())
    json_only = set(json_files.keys()) - set(wav_files.keys())

    print(f"  WAV  파일: {len(wav_files):,}개")
    print(f"  JSON 파일: {len(json_files):,}개")
    print(f"  ✅ 매칭됨:  {len(matched):,}개")
    if wav_only:
        print(f"  ⚠️  JSON 없는 WAV:  {len(wav_only):,}개")
    if json_only:
        print(f"  ⚠️  WAV 없는 JSON:  {len(json_only):,}개")

    return len(matched)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--verify_only", action="store_true", help="압축 해제 없이 매칭 검증만")
    args = parser.parse_args()

    data_root = Path(args.data_root)

    if args.verify_only:
        verify_wav_json_pairs(data_root)
        return

    zip_files    = sorted(data_root.rglob("*.zip"))
    source_zips  = [z for z in zip_files if is_source_zip(z)]

    print(f"\n📦 원천데이터 ZIP: {len(source_zips)}개")
    total_gb = sum(z.stat().st_size for z in source_zips) / (1024**3)
    print(f"   총 용량: {total_gb:.1f} GB\n")

    for zip_path in source_zips:
        size_gb = zip_path.stat().st_size / (1024**3)
        out_dir = zip_path.parent / zip_path.stem
        print(f"  📂 {zip_path.name}  ({size_gb:.1f} GB)")
        print(f"     → {out_dir}/")
        unzip_with_progress(zip_path, out_dir)

    verify_wav_json_pairs(data_root)

    print("\n🎉 전체 압축 해제 완료!")
    print("   다음 단계: python 06_prepare_asr_data.py 실행")


if __name__ == "__main__":
    main()
