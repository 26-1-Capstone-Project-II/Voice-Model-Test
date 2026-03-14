# test_trimmed.py
import torch
import numpy as np
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

MODEL_ID  = "kresnik/wav2vec2-large-xlsr-korean"
TARGET_SR = 16000

print("[모델 로드 중...]")
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model     = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
model.eval()
print("✅ 모델 로드 완료\n")

def transcribe(audio: np.ndarray) -> str:
    inputs = processor(audio, sampling_rate=TARGET_SR,
                       return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(ids)[0].strip()

def analyze(audio: np.ndarray, label: str):
    duration     = len(audio) / TARGET_SR
    silence_rate = (np.abs(audio) < 0.01).mean() * 100
    print(f"  길이: {duration:.2f}초  |  무음: {silence_rate:.1f}%")

    result = transcribe(audio)
    match  = "✅" if label.replace(" ","") == result.replace(" ","") else "🔍"
    print(f"  정답: {label}")
    print(f"  인식: {result}  {match}")

TEST_CASES = [
    ("test_같이 .wav",  "같이 해볼까"),
    ("test_안녕하.wav", "안녕하세요"),
    ("test_오늘 .wav",  "오늘 날씨가 좋네요"),
]

print("=" * 54)
print("  [원본] vs [무음제거] 비교")
print("=" * 54)

for wav_path, label in TEST_CASES:
    raw, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)

    # librosa trim: top_db=20 → 20dB 이하 무음 제거
    trimmed, _ = librosa.effects.trim(raw, top_db=20)

    print(f"\n▶ '{label}'")
    print(f"  ── 원본 ──")
    analyze(raw, label)
    print(f"  ── 무음제거 후 ──")
    analyze(trimmed, label)

print("\n" + "=" * 54)