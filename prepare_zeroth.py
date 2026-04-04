"""
HuggingFace Zeroth-Korean 데이터셋 다운로드 및 전처리 스크립트
============================================================
AIHub 대안으로 사용할 수 있는 오픈소스 한국어 낭독체 데이터셋(51시간).
회원가입 없이 즉시 다운로드 가능하며, 완벽하게 문장 단위로 분할되어 있습니다.

실행:
    python prepare_zeroth.py
"""

import os
import json
import soundfile as sf
from pathlib import Path

def main():
    try:
        from datasets import load_dataset, Audio
        from tqdm import tqdm
    except ImportError:
        print("❌ 라이브러리가 없습니다. 먼저 설치해주세요:")
        print("   pip install datasets tqdm soundfile")
        return

    # 저장 경로 설정
    out_dir = Path.home() / "mingly_workspace" / "Voice-Model-Test" / "zeroth_dataset"
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    
    print("📥 [1/3] HuggingFace에서 Zeroth-Korean 데이터셋 로드 중...")
    ds = load_dataset("Bingsu/zeroth-korean")
    
    # torchcodec 오류를 우회하기 위해 Audio 내장 디코딩 비활성화
    ds = ds.cast_column("audio", Audio(decode=False))
    
    # 분할 (train 2.2만개, test 4천개)
    train_data = list(ds["train"])
    test_data = list(ds["test"])
    
    import random
    random.seed(42)
    random.shuffle(train_data)
    
    # Train에서 500개를 떼어내 Validation으로 사용
    val_size = 500
    val_data = train_data[:val_size]
    train_data = train_data[val_size:]
    
    splits = {
        "train": train_data,
        "validation": val_data,
        "test": test_data
    }
    
    print(f"\n📊 [2/3] 데이터 세그멘테이션 완료:")
    print(f"   Train: {len(train_data):,}개")
    print(f"   Val:   {len(val_data):,}개")
    print(f"   Test:  {len(test_data):,}개")
    
    print("\n⚙️ [3/3] 오디오 파일(WAV) 추출 및 JSONL 생성 시작...")
    
    import io
    for split_name, records in splits.items():
        jsonl_path = out_dir / f"{split_name}.jsonl"
        
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for i, rec in enumerate(tqdm(records, desc=f"{split_name[:5]:5s}")):
                text = rec["text"]
                audio_info = rec["audio"]
                
                # 수동 디코딩 (torchcodec 우회)
                if "bytes" in audio_info and audio_info["bytes"] is not None:
                    audio_array, sr = sf.read(io.BytesIO(audio_info["bytes"]))
                else:
                    audio_array, sr = sf.read(audio_info["path"])
                
                # 개별 WAV 파일로 추출하여 저장 (finetune_whisper.py와 완벽 호환)
                wav_name = f"{split_name}_{i:06d}.wav"
                wav_path = wav_dir / wav_name
                
                sf.write(str(wav_path), audio_array, sr)
                
                duration = len(audio_array) / sr
                
                # JSONL 기록
                # (label은 finetune_whisper.py에서 --apply_g2p 적용 시 자동 발음기호로 덮어씌워집니다)
                obj = {
                    "wav_path": str(wav_path),
                    "transcript": text,
                    "label": text, 
                    "duration": float(duration)
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                
    print(f"\n✅ 완벽하게 준비되었습니다! 경로: {out_dir}")
    print(f"   이제 아래 명령어로 파인튜닝을 시작하세요:")
    print(f"   CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python finetune_whisper.py \\")
    print(f"       --json_dir {out_dir} \\")
    print(f"       --apply_g2p \\")
    print(f"       --lr 2e-5 --num_epochs 3 --batch_size 8 --grad_accum 2")

if __name__ == "__main__":
    main()
