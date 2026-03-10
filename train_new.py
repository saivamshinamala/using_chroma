import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM

os.environ["WANDB_DISABLED"] = "true"

# ---------------- PATHS ----------------
BASE_MODEL_PATH = r"E:\llms\Qwen2-7B-Instruct"
DATASET_PATH    = r"data\train_aug.jsonl"
OUTPUT_DIR      = r"models\qwen_qlora_adapters"

# ---------------- CHECK CUDA ----------------
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠️ CUDA not detected. Training will be very slow on CPU.")

# ---------------- QUANTIZATION CONFIG ----------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,   # bfloat16 is more stable than float16
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

# ---------------- LOAD TOKENIZER ----------------
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,
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
    trust_remote_code=True,
)

model.config.use_cache = False

# ⭐ IMPROVEMENT 1: Prepare model for k-bit training
# Casts LayerNorm and output head to float32 for training stability
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,   # saves VRAM on GTX 1650
)

# ---------------- LORA CONFIG ----------------
print("\nSetting up LoRA...")
lora_config = LoraConfig(
    # ⭐ IMPROVEMENT 2: Higher rank = more expressive adaptations
    # r=32 gives ~2x more trainable params vs r=16, still fits in 4GB VRAM
    r=32,
    lora_alpha=64,          # keep alpha = 2 * r for stable scaling
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    # ⭐ IMPROVEMENT 3: Added embed_tokens + lm_head for better token learning
    # These layers handle vocabulary mapping — crucial for factual accuracy
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    # ⭐ IMPROVEMENT 4: modules_to_save ensures output layer is fully updated
    modules_to_save=["embed_tokens", "lm_head"],
)

# ---------------- LOAD DATASET ----------------
print("\nLoading dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.train_test_split(test_size=0.1, seed=42)

train_dataset = dataset["train"]
eval_dataset  = dataset["test"]

print("Train samples:", len(train_dataset))
print("Eval samples :", len(eval_dataset))

# ---------------- FORMAT DATA ----------------
def format_example(example):
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

train_dataset = train_dataset.map(format_example)
eval_dataset  = eval_dataset.map(format_example)

# ⭐ IMPROVEMENT 5: Response-only training with DataCollatorForCompletionOnlyLM
# By default SFTTrainer trains on ALL tokens (prompt + response).
# This means the model wastes capacity memorizing the prompt template.
# With completion-only training, loss is computed ONLY on assistant responses
# → model learns to generate good answers, not to copy prompts.
#
# For Qwen2-Instruct, the assistant turn starts with <|im_start|>assistant
response_template = "<|im_start|>assistant"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer,
    mlm=False,
)

# ---------------- TRAINING CONFIG ----------------
# ⭐ IMPROVEMENT 6: Warmup + cosine LR schedule
# Without warmup, the optimizer takes large steps early → unstable loss.
# Cosine decay slowly anneals LR toward zero → better final convergence.
total_steps = (len(train_dataset) // (1 * 8)) * 3   # batch_size=1, accum=8, epochs=3
warmup_steps = max(20, total_steps // 10)            # 10% warmup

sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,          # effective batch = 8
    learning_rate=2e-4,
    lr_scheduler_type="cosine",             # ⭐ cosine decay
    warmup_steps=warmup_steps,              # ⭐ warmup
    fp16=False,                             # ⭐ use bf16 instead (more stable)
    bf16=True,
    logging_steps=5,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,            # ⭐ keep the best checkpoint
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    save_total_limit=2,
    optim="paged_adamw_8bit",
    max_length=1024,
    packing=False,
    report_to="none",
    # ⭐ IMPROVEMENT 7: Gradient clipping prevents exploding gradients
    max_grad_norm=0.3,
    # ⭐ IMPROVEMENT 8: Weight decay acts as regularization → less overfitting
    weight_decay=0.01,
    # Dataset text field
    dataset_text_field="text",
)

# ---------------- TRAINER ----------------
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=lora_config,
    args=sft_config,
    data_collator=collator,       # ⭐ response-only training
)

# ---------------- TRAIN ----------------
print("\n🚀 Starting QLoRA Fine-tuning (with response-only loss)...\n")
print(f"   LoRA rank       : {lora_config.r}")
print(f"   LR scheduler    : cosine with {warmup_steps} warmup steps")
print(f"   Training on     : assistant responses ONLY (completion-only collator)")
print(f"   Precision       : bfloat16\n")

trainer.train()

# ---------------- SAVE ----------------
print("\n💾 Saving adapters + tokenizer...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n✅ Fine-tuning complete!")
print("Adapters saved at:", OUTPUT_DIR)
