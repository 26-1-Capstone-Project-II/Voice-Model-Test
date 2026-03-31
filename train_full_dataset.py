import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback
from peft import prepare_model_for_int8_training, LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model

# ==========================================
# 1. 설정 (100% 학습에 맞게 변경된 부분)
# ==========================================
model_name_or_path = "openai/whisper-small" # 사용하시는 모델 크기에 맞게 변경하세요
dataset_name = "your_dataset_name_or_path"  # 실제 데이터셋 경로
output_dir = "./lora_adapter_100/best"      # ⭐️ 기존 10% 모델을 덮어쓰지 않도록 경로 변경!

print("🚀 100% 전체 데이터셋 ASR(음성 인식) 파인튜닝 스크립트를 시작합니다!")

# ==========================================
# 2. 프로세서, 토크나이저, 특징 추출기 로드
# ==========================================
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name_or_path)
tokenizer = WhisperTokenizer.from_pretrained(model_name_or_path, language="Korean", task="transcribe")
processor = WhisperProcessor.from_pretrained(model_name_or_path, language="Korean", task="transcribe")

# ==========================================
# 3. 데이터셋 로드 (100% 전체)
# ==========================================
print("📦 전체 데이터셋을 로드하고 있습니다...")
# ⭐️ 10% 슬라이싱(예: select 또는 [:10%])을 모두 제거하고 전체를 불러옵니다.
dataset = load_dataset(dataset_name)

# 오디오 샘플링 레이트를 Whisper 모델이 요구하는 16kHz로 맞춥니다.
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch):
    # 1. 오디오 데이터 로드 및 특징 추출
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    
    # 2. 정답(Label) 텍스트 토큰화 
    # (발음 기호 그대로 전사된 텍스트가 'sentence' 또는 'text' 컬럼에 있다고 가정)
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

print("🛠️ 전체 데이터 전처리를 시작합니다. 시간이 다소 소요될 수 있습니다...")
# 전체 데이터 맵핑
dataset = dataset.map(prepare_dataset, remove_columns=dataset.column_names["train"], num_proc=4)

# ==========================================
# 4. 데이터 콜레이터 정의
# ==========================================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# ==========================================
# 5. 모델 로드 및 LoRA (PEFT) 설정
# ==========================================
model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path, load_in_8bit=True, device_map="auto")
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

# 8bit 정밀도 학습 준비
model = prepare_model_for_int8_training(model)

# LoRA 설정
config = LoraConfig(
    r=32, 
    lora_alpha=64, 
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05, 
    bias="none"
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

# ==========================================
# 6. 학습 인자 설정 (100% 스케일에 맞게 조정)
# ==========================================
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,  # 변경된 저장 경로
    per_device_train_batch_size=8, # VRAM에 맞춰 조절하세요 (OOM 발생 시 4나 2로 줄임)
    gradient_accumulation_steps=2, # batch_size를 줄였다면 이 값을 늘려주세요
    learning_rate=1e-3,
    warmup_steps=500,       # 데이터가 늘었으므로 warmup step도 조금 늘려줍니다
    max_steps=5000,         # 전체 데이터셋 크기에 맞춰 적절한 총 스텝 수 설정 (또는 num_train_epochs 사용)
    evaluation_strategy="steps",
    fp16=True,
    per_device_eval_batch_size=8,
    generation_max_length=225,
    save_steps=1000,        # ⭐️ 너무 자주 저장하면 디스크 용량이 꽉 차므로 저장 주기를 늘립니다 (예: 1000)
    eval_steps=1000,        # ⭐️ 평가 주기도 함께 늘려줍니다
    logging_steps=50,       # 로그를 더 자주 확인하기 위해 50으로 수정 (원래 100)
    report_to=["tensorboard"],
    remove_unused_columns=False,
    label_names=["labels"],
)

# ==========================================
# 7. 트레이너 실행
# ==========================================
# 커스텀 콜백 정의: 백그라운드 학습 시에도 로그를 깔끔하게 출력해주는 역할
class CustomProgressCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            step = state.global_step
            max_steps = state.max_steps
            loss = logs.get("loss", "N/A")
            lr = logs.get("learning_rate", "N/A")
            epoch = logs.get("epoch", "N/A")
            
            # 소수점 자리수 깔끔하게 포매팅
            if isinstance(loss, float): loss = f"{loss:.4f}"
            if isinstance(lr, float): lr = f"{lr:.2e}"
            if isinstance(epoch, float): epoch = f"{epoch:.2f}"
            
            print(f"📊 [Step {step}/{max_steps}] Epoch: {epoch} | Loss: {loss} | Learning Rate: {lr}")

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"], # 평가용 데이터셋이 있다면 지정
    data_collator=data_collator,
    tokenizer=processor.feature_extractor,
    callbacks=[CustomProgressCallback()] # ⭐️ 커스텀 상태 출력 콜백 추가
)

print("🔥 100% 데이터셋 학습을 시작합니다!")
trainer.train()

# 최종 모델 저장
model.save_pretrained(output_dir)
print(f"🎉 100% 학습 완료 및 모델 저장 성공! (경로: {output_dir})")