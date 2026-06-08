"""
원거리·소음·긴발화 후반부 환각 — 정량 베이스라인 측정
=====================================================
목적(재학습 todo #1):
  현재 fine-tuned whisper-base가 원거리(잔향)/소음/긴 발화 후반부에서
  얼마나 환각하는지를 "앱과 동일한 디코딩 조건"으로 정량화한다.
  재학습 전후 비교의 기준선(baseline)을 만든다.

핵심 관찰(이슈):
  - 환각이 저신뢰가 아니라 *고신뢰*(avgLogProb 정상)로 발생 → 온디바이스 게이팅 불가.
  - 음성이 끝난 뒤 30초 윈도우의 비음성 꼬리를 fluent 텍스트로 채우는 패턴.
  따라서 본 스크립트는 (1) 열화 조건별 CER/WER 와 (2) 비음성 꼬리 환각률,
  (3) 노이즈-only 순수 환각률, (4) 출력의 avgLogProb 를 함께 측정한다.

조건 매트릭스:
  clean / reverb(원거리) / noise@SNR(0,5,10,15,20) / reverb+noise
  + 각 음성 조건에 "비음성 꼬리(noise tail)" 패딩 변형(--tail_sec)

증강은 augment.py 의 검증된 RIR/소음 코드를 재사용(조건별 SNR 고정).

실행(서버):
    CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python diagnose_farfield_baseline.py \\
        --model_path best_model_whisper/best \\
        --json_dir zeroth_dataset \\
        --num_samples 200 \\
        --apply_g2p \\
        --tail_sec 8 \\
        --output_dir results/farfield_baseline

출력:
    results/farfield_baseline/baseline_results.json   # 조건별 집계 + 샘플별 상세
    results/farfield_baseline/summary.md              # 사람이 읽는 요약표
"""

import os
import re
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import librosa

torch.backends.cudnn.enabled = False

TARGET_SR = 16000
MAX_AUDIO_SEC = 30.0            # Whisper 처리 윈도우
MIN_SEC = 1.0                   # 너무 짧은 세그먼트 제외
MAX_SRC_SEC = 25.0             # 원본 음성 길이 상한(꼬리 패딩 여유 확보)


# ────────────────────────────────────────────
# 1. 텍스트/메트릭 유틸
# ────────────────────────────────────────────
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _edit_ops(ref: str, hyp: str):
    """문자 단위 Levenshtein 연산 카운트 (sub, ins, del)."""
    r, h = list(ref), list(hyp)
    n, m = len(r), len(h)
    # dp[i][j] = (cost, sub, ins, del)
    dp = [[(0, 0, 0, 0)] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        c = dp[i - 1][0]
        dp[i][0] = (c[0] + 1, c[1], c[2], c[3] + 1)
    for j in range(1, m + 1):
        c = dp[0][j - 1]
        dp[0][j] = (c[0] + 1, c[1], c[2] + 1, c[3])
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if r[i - 1] == h[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue
            sub = dp[i - 1][j - 1]
            ins = dp[i][j - 1]
            dele = dp[i - 1][j]
            best = min(sub, ins, dele, key=lambda x: x[0])
            if best is sub:
                dp[i][j] = (sub[0] + 1, sub[1] + 1, sub[2], sub[3])
            elif best is ins:
                dp[i][j] = (ins[0] + 1, ins[1], ins[2] + 1, ins[3])
            else:
                dp[i][j] = (dele[0] + 1, dele[1], dele[2], dele[3] + 1)
    _, sub, ins, dele = dp[n][m]
    return {"sub": sub, "ins": ins, "del": dele, "ref_len": n, "hyp_len": m}


def cer(ref: str, hyp: str) -> float:
    ref, hyp = _norm(ref).replace(" ", ""), _norm(hyp).replace(" ", "")
    if not ref:
        return 0.0 if not hyp else 1.0
    ops = _edit_ops(ref, hyp)
    return (ops["sub"] + ops["ins"] + ops["del"]) / max(1, ops["ref_len"])


def wer(ref: str, hyp: str) -> float:
    r, h = _norm(ref).split(), _norm(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    # 단어 단위 편집거리
    n, m = len(r), len(h)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m] / max(1, n)


def insertion_rate(ref: str, hyp: str) -> float:
    """삽입 문자 / 참조 문자 — 환각(불필요한 추가 출력) 프록시."""
    ref_c, hyp_c = _norm(ref).replace(" ", ""), _norm(hyp).replace(" ", "")
    if not ref_c:
        return float(len(hyp_c) > 0)
    ops = _edit_ops(ref_c, hyp_c)
    return ops["ins"] / max(1, ops["ref_len"])


# ────────────────────────────────────────────
# 2. 데이터 로딩 (zeroth test split)
# ────────────────────────────────────────────
def load_records(json_dir, split, num_samples, apply_g2p):
    json_dir = Path(json_dir)
    path = json_dir / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음")

    g2p = None
    if apply_g2p:
        from korean_g2p_nomecab import load_g2p
        g2p = load_g2p()

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
            except Exception:
                continue
            dur = obj.get("duration", 0)
            if dur < MIN_SEC or dur > MAX_SRC_SEC:
                continue
            wav_path = obj.get("wav_path", "")
            if not wav_path or not Path(wav_path).exists():
                continue
            transcript = _norm(obj.get("transcript", ""))
            if apply_g2p and g2p and transcript:
                label = _norm(g2p(transcript, descriptive=True))
            else:
                label = _norm(obj.get("label", "") or transcript)
            if len(label) < 2:
                continue
            records.append({"wav_path": wav_path, "label": label,
                            "transcript": transcript, "duration": dur})
            if num_samples > 0 and len(records) >= num_samples:
                break
    print(f"  📂 {split}: {len(records):,}개 로드 (g2p={'on' if apply_g2p else 'off'})")
    return records


# ────────────────────────────────────────────
# 3. 모델 래퍼 (앱과 동일한 greedy 디코딩 + avgLogProb)
# ────────────────────────────────────────────
class GreedyDecoder:
    def __init__(self, model_path, device=None, neutral=True, allow_eot=False):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📥 모델 로드: {model_path}  (device={self.device})")
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.neutral = neutral
        if neutral:
            # 앱(WhisperKit)은 HF의 repetition_penalty/no_repeat_ngram 을 적용하지 않는다.
            # 학습 시 config 에 박힌 anti-repeat 를 꺼서 "앱이 실제로 보는" 환각을 재현.
            gc = self.model.generation_config
            gc.repetition_penalty = 1.0
            gc.no_repeat_ngram_size = 0
            print("  🎚️  중립 디코딩: repetition_penalty=1.0, no_repeat_ngram_size=0 (앱 일치)")
        if allow_eot:
            # 진단용: begin_suppress_tokens 에서 EOS(50257) 를 빼 첫 토큰 EOS 를 허용한다.
            # Whisper 기본값 [220, 50257] 은 빈 출력 방지로 첫-EOS 를 막아, 비음성에도
            # 강제로 텍스트를 내게 한다. 이를 풀면 noise-only 환각이 데이터로 검증됨.
            gc = self.model.generation_config
            bs = list(getattr(gc, "begin_suppress_tokens", None) or [])
            gc.begin_suppress_tokens = [t for t in bs if t != self.model.config.eos_token_id]
            print(f"  🔓 begin_suppress EOS 해제: {bs} → {gc.begin_suppress_tokens} (첫 토큰 EOS 허용)")

    @torch.no_grad()
    def transcribe(self, audio: np.ndarray):
        """오디오(파형) → (텍스트, avg_logprob). greedy, temperature 0."""
        feats = self.processor.feature_extractor(
            audio, sampling_rate=TARGET_SR, return_tensors="pt"
        ).input_features.to(self.device)

        out = self.model.generate(
            feats,
            max_new_tokens=256,
            language="ko",
            task="transcribe",
            num_beams=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
        seq = out.sequences
        text = self.processor.tokenizer.batch_decode(seq, skip_special_tokens=True)[0].strip()

        # avgLogProb (생성 토큰 평균 로그확률) — 앱의 avgLogProb 대응
        try:
            trans = self.model.compute_transition_scores(
                seq, out.scores, normalize_logits=True
            )[0]
            trans = trans[torch.isfinite(trans)]
            avg_lp = float(trans.mean().item()) if trans.numel() else float("nan")
        except Exception:
            avg_lp = float("nan")
        return text, avg_lp


# ────────────────────────────────────────────
# 4. 오디오 열화 (augment.py 재사용)
# ────────────────────────────────────────────
def build_augmentor(noise_root, rir_root):
    from augment import FarFieldAugmentor
    # 파일 리스트만 로드해 두고, 조건별로 p_*/snr 을 직접 세팅해 재사용한다.
    aug = FarFieldAugmentor(
        noise_root=noise_root, rir_root=rir_root,
        p_reverb=0.0, p_noise=0.0, p_gain=0.0, p_hard_snr=0.0,
    )
    return aug


def degrade(aug, audio, *, reverb=False, snr_db=None):
    """조건별 결정적 열화: 잔향 → 소음(SNR 고정)."""
    a = audio.astype(np.float32)
    if reverb and aug.rir_files:
        a = aug._reverberate(a)
    if snr_db is not None and aug.noise_files:
        aug.snr_db_range = (float(snr_db), float(snr_db))
        aug.p_hard_snr = 0.0
        a = aug._add_noise(a)
    peak = float(np.max(np.abs(a))) if len(a) else 0.0
    if peak > 0.99:
        a = a * (0.99 / peak)
    return a.astype(np.float32)


def append_noise_tail(aug, audio, tail_sec, snr_db=10.0, total_sec=MAX_AUDIO_SEC):
    """
    음성 뒤에 비음성(노이즈) 꼬리를 붙여 '말 끝난 뒤 윈도우 잔여 구간' 재현.
    참조 라벨은 음성 부분만 → 꼬리에서 나오는 출력은 전부 환각.
    """
    from augment import _load_audio, _match_length
    import random
    tail_len = int(tail_sec * TARGET_SR)
    if tail_len <= 0:
        return audio
    if aug.noise_files:
        noise = _match_length(_load_audio(random.choice(aug.noise_files)), tail_len)
        # 꼬리 노이즈를 음성 RMS 대비 snr_db 수준으로 스케일
        sp = float(np.mean(audio ** 2)) + 1e-8
        npow = float(np.mean(noise ** 2)) + 1e-8
        scale = np.sqrt(sp / (10 ** (snr_db / 10)) / npow)
        tail = (scale * noise).astype(np.float32)
    else:
        tail = np.zeros(tail_len, dtype=np.float32)  # 노이즈 없으면 무음 꼬리
    out = np.concatenate([audio, tail])
    max_len = int(total_sec * TARGET_SR)
    return out[:max_len].astype(np.float32)


# ────────────────────────────────────────────
# 5. 조건 정의 & 실행
# ────────────────────────────────────────────
def make_conditions(snr_list, tail_sec):
    conds = [{"name": "clean", "reverb": False, "snr": None, "tail": 0.0}]
    conds.append({"name": "reverb", "reverb": True, "snr": None, "tail": 0.0})
    for snr in snr_list:
        conds.append({"name": f"noise_snr{int(snr)}", "reverb": False, "snr": snr, "tail": 0.0})
    conds.append({"name": "reverb+noise_snr10", "reverb": True, "snr": 10.0, "tail": 0.0})
    if tail_sec and tail_sec > 0:
        # 후반부 환각 집중 조건: 음성 + 비음성 꼬리
        conds.append({"name": f"clean+tail{int(tail_sec)}", "reverb": False, "snr": None, "tail": tail_sec})
        conds.append({"name": f"reverb+noise+tail{int(tail_sec)}", "reverb": True, "snr": 10.0, "tail": tail_sec})
    return conds


def run_condition(decoder, aug, records, cond):
    rows = []
    for rec in records:
        audio, _ = librosa.load(rec["wav_path"], sr=TARGET_SR, mono=True)
        audio = audio[: int(MAX_SRC_SEC * TARGET_SR)]
        a = degrade(aug, audio, reverb=cond["reverb"], snr_db=cond["snr"])
        if cond["tail"] > 0:
            a = append_noise_tail(aug, a, cond["tail"], snr_db=(cond["snr"] or 10.0))
        a = a[: int(MAX_AUDIO_SEC * TARGET_SR)]
        hyp, avg_lp = decoder.transcribe(a)
        ref = rec["label"]
        rows.append({
            "ref": ref, "hyp": hyp,
            "cer": cer(ref, hyp), "wer": wer(ref, hyp),
            "ins_rate": insertion_rate(ref, hyp),
            "len_ratio": len(_norm(hyp).replace(" ", "")) / max(1, len(_norm(ref).replace(" ", ""))),
            "avg_logprob": avg_lp,
        })
    return rows


def run_noise_only(decoder, aug, n_clips, tail_sec):
    """순수 노이즈(음성 없음) → 환각률. 참조는 빈 문자열."""
    from augment import _load_audio, _match_length
    import random
    if not aug.noise_files:
        return None
    rows = []
    length = int((tail_sec if tail_sec else 10.0) * TARGET_SR)
    for _ in range(n_clips):
        noise = _match_length(_load_audio(random.choice(aug.noise_files)), length)
        peak = float(np.max(np.abs(noise))) + 1e-8
        noise = (0.3 * noise / peak).astype(np.float32)  # 적당한 레벨
        hyp, avg_lp = decoder.transcribe(noise)
        rows.append({"hyp": hyp, "n_chars": len(_norm(hyp).replace(" ", "")),
                    "avg_logprob": avg_lp})
    return rows


def aggregate(rows):
    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else float("nan")
    # 꼬리/소음 환각 발생 비율: 삽입률이 큰 샘플 비율
    halluc = [r for r in rows if r["ins_rate"] >= 0.30 or r["len_ratio"] >= 1.30]
    return {
        "n": len(rows),
        "cer": mean("cer"),
        "wer": mean("wer"),
        "ins_rate": mean("ins_rate"),
        "len_ratio": mean("len_ratio"),
        "avg_logprob": mean("avg_logprob"),
        "halluc_rate": len(halluc) / max(1, len(rows)),
    }


# ────────────────────────────────────────────
# 6. 메인
# ────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="원거리/소음/긴발화 후반부 환각 베이스라인 측정")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--json_dir", default="zeroth_dataset")
    ap.add_argument("--split", default="test")
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--apply_g2p", action="store_true", help="참조 라벨을 transcript에서 g2p 재생성")
    ap.add_argument("--snr_list", default="0,5,10,15,20")
    ap.add_argument("--tail_sec", type=float, default=8.0, help="비음성 꼬리 길이(초). 0이면 비활성")
    ap.add_argument("--noise_root", default=os.environ.get("MUSAN_NOISE_DIR", "/data/musan/noise"))
    ap.add_argument("--rir_root", default=os.environ.get("RIR_DIR", "/data/RIRS_NOISES/simulated_rirs"))
    ap.add_argument("--noise_only_clips", type=int, default=50)
    ap.add_argument("--no_neutral", action="store_true", help="학습 config 그대로(anti-repeat 유지)")
    ap.add_argument("--allow_eot", action="store_true",
                    help="진단: begin_suppress 에서 EOS 제거 → 첫 토큰 EOS 허용(비음성→빈 출력 가능). "
                         "noise-only 환각의 원인이 디코딩(begin_suppress)임을 검증")
    ap.add_argument("--output_dir", default="results/farfield_baseline")
    args = ap.parse_args()

    snr_list = [float(x) for x in args.snr_list.split(",") if x.strip() != ""]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    decoder = GreedyDecoder(args.model_path, neutral=not args.no_neutral, allow_eot=args.allow_eot)
    aug = build_augmentor(args.noise_root, args.rir_root)
    records = load_records(args.json_dir, args.split, args.num_samples, args.apply_g2p)

    conditions = make_conditions(snr_list, args.tail_sec)
    results = {"config": vars(args), "conditions": {}}

    print(f"\n{'='*72}\n  베이스라인 측정 시작 — {len(records)}개 × {len(conditions)}조건\n{'='*72}")
    for cond in conditions:
        rows = run_condition(decoder, aug, records, cond)
        agg = aggregate(rows)
        results["conditions"][cond["name"]] = {"agg": agg, "samples": rows[:20]}
        print(f"  [{cond['name']:>24}]  CER={agg['cer']:.3f}  WER={agg['wer']:.3f}  "
              f"ins={agg['ins_rate']:.3f}  len×{agg['len_ratio']:.2f}  "
              f"halluc={agg['halluc_rate']:.1%}  lp={agg['avg_logprob']:.2f}")

    # 순수 노이즈 환각
    noise_rows = run_noise_only(decoder, aug, args.noise_only_clips, args.tail_sec)
    if noise_rows is not None:
        nonempty = [r for r in noise_rows if r["n_chars"] >= 2]
        results["noise_only"] = {
            "n": len(noise_rows),
            "halluc_rate": len(nonempty) / max(1, len(noise_rows)),
            "avg_logprob_when_halluc": float(np.nanmean([r["avg_logprob"] for r in nonempty])) if nonempty else float("nan"),
            "samples": noise_rows[:20],
        }
        no = results["noise_only"]
        print(f"\n  [noise-only 순수환각]  환각률={no['halluc_rate']:.1%}  "
              f"환각시 avgLogProb={no['avg_logprob_when_halluc']:.2f}  (고신뢰면 게이팅 불가 입증)")

    # 저장
    (out_dir / "baseline_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary(out_dir / "summary.md", results, conditions)
    print(f"\n✅ 저장: {out_dir}/baseline_results.json , summary.md")


def _write_summary(path, results, conditions):
    lines = ["# 원거리/소음/긴발화 후반부 환각 — 베이스라인\n"]
    lines.append("| 조건 | CER | WER | 삽입률 | 길이비 | 환각률 | avgLogProb |")
    lines.append("|------|----:|----:|------:|------:|------:|----------:|")
    for cond in conditions:
        a = results["conditions"][cond["name"]]["agg"]
        lines.append(f"| {cond['name']} | {a['cer']:.3f} | {a['wer']:.3f} | "
                     f"{a['ins_rate']:.3f} | {a['len_ratio']:.2f} | "
                     f"{a['halluc_rate']:.1%} | {a['avg_logprob']:.2f} |")
    if "noise_only" in results:
        no = results["noise_only"]
        lines.append(f"\n**noise-only 순수 환각률**: {no['halluc_rate']:.1%} "
                     f"(환각 시 avgLogProb={no['avg_logprob_when_halluc']:.2f})\n")
        lines.append("> avgLogProb 가 정상 수준이면, 온디바이스 신뢰도 게이팅으로는 "
                     "환각을 차단할 수 없음 → 데이터 분포(증강 재학습)로만 교정 가능.\n")
    lines.append("\n해석: `clean` 대비 열화 조건에서 CER/삽입률/길이비/환각률이 "
                 "오르고 꼬리(tail) 조건에서 두드러지면, 후반부 환각이 데이터로 재현된 것. "
                 "재학습 후 동일 스크립트로 비교한다.\n")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
