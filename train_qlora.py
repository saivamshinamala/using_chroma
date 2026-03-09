import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

os.environ["WANDB_DISABLED"] = "true"

# ---------------- PATHS ----------------
BASE_MODEL_PATH = r"E:\llms\Qwen2-7B-Instruct"
DATASET_PATH = r"data\train_aug.jsonl"
OUTPUT_DIR = r"models\qwen_qlora_adapters"

# ---------------- CHECK CUDA ----------------
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠️ CUDA not detected. Training will be very slow on CPU.")

# ---------------- QUANTIZATION CONFIG ----------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# ---------------- LOAD TOKENIZER ----------------
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---------------- LOAD MODEL ----------------
print("\nLoading Qwen model in 4-bit (QLoRA)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True
)

model.config.use_cache = False

# ---------------- LORA CONFIG ----------------
print("\nSetting up LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
)

# ---------------- LOAD DATASET ----------------
print("\nLoading dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.train_test_split(test_size=0.1, seed=42)

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

print("Train samples:", len(train_dataset))
print("Eval samples:", len(eval_dataset))

# ---------------- FORMAT DATA ----------------
def format_example(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

train_dataset = train_dataset.map(format_example)
eval_dataset = eval_dataset.map(format_example)

# ---------------- TRAINING ARGUMENTS ----------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,      # GTX 1650 safe
    gradient_accumulation_steps=8,      # effective batch size = 8
    learning_rate=2e-4,
    fp16=True,
    logging_steps=5,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=2,
    report_to="none",
    optim="paged_adamw_8bit"
)

# ---------------- TRAINER ----------------
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=5,
    save_strategy="epoch",
    eval_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    optim="paged_adamw_8bit",
    max_length=1024,
    packing=False,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    args=sft_config,
)

# ---------------- TRAIN ----------------
print("\n🚀 Starting QLoRA Fine-tuning...\n")
trainer.train()

# ---------------- SAVE ----------------
print("\n💾 Saving adapters + tokenizer...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n✅ Fine-tuning complete!")
print("Adapters saved at:", OUTPUT_DIR)