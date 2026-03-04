"""
Optimized QLoRA Fine-tuning — Qwen2-7B-Instruct
=================================================
Changes from previous version:
  - Removed duplicate TrainingArguments (was dead code)
  - Increased to 5 epochs with cosine LR scheduler
  - Added warmup_ratio for stable early training
  - Added gradient_checkpointing (critical for GTX 1650)
  - Explicit dataset_text_field to avoid silent TRL issues
  - max_length raised to 1536 to avoid truncating long answers
  - Added per_device_eval_batch_size
  - Added load_best_model_at_end to save best checkpoint
  - Added logging of sample formatted text for sanity check
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

os.environ["WANDB_DISABLED"] = "true"

# ================= PATHS =================
BASE_MODEL_PATH = r"E:\llms\Qwen2-7B-Instruct"
DATASET_PATH    = r"data\train_aug.jsonl"
OUTPUT_DIR      = r"models\qwen_qlora_adapters"

# ================= CHECK CUDA =================
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
else:
    print("⚠️ CUDA not detected. Training will be extremely slow on CPU.")

# ================= QUANTIZATION =================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,   # saves ~0.4GB extra VRAM on 1650
)

# ================= TOKENIZER =================
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    local_files_only=True,
    trust_remote_code=True,
)
if tokenizer.pad_token is None:
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ── Important: pad on right for causal LM training ──────────────────────────
tokenizer.padding_side = "right"

# ================= MODEL =================
print("\nLoading Qwen2-7B in 4-bit (QLoRA)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True,
)

model.config.use_cache = False

# Critical for GTX 1650 — reduces VRAM peak by ~30% during backward pass
model.enable_input_require_grads()
model.gradient_checkpointing_enable()

# ================= LORA CONFIG =================
print("\nSetting up LoRA (r=16)...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,          # alpha = 2x rank is standard
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
        "down_proj",
    ],
)

# ================= DATASET =================
print("\nLoading dataset...")
raw = load_dataset("json", data_files=DATASET_PATH, split="train")
raw = raw.train_test_split(test_size=0.1, seed=42)

train_dataset = raw["train"]
eval_dataset  = raw["test"]

print(f"Train samples : {len(train_dataset)}")
print(f"Eval  samples : {len(eval_dataset)}")

# ================= FORMAT =================
def format_example(example):
    """
    Apply Qwen chat template to convert messages → single training string.
    add_generation_prompt=False so the model learns to generate the full
    assistant turn including the EOS token.
    """
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}

train_dataset = train_dataset.map(format_example, remove_columns=["messages"])
eval_dataset  = eval_dataset.map(format_example,  remove_columns=["messages"])

# ── Sanity check: print one formatted sample ────────────────────────────────
print("\n📋 Sample formatted training text:")
print("-" * 50)
sample_text = train_dataset[0]["text"]
print(sample_text[:600] + "..." if len(sample_text) > 600 else sample_text)
print("-" * 50)

# Check max token length in dataset to set max_length correctly
print("\n🔍 Checking token lengths...")
lengths = []
for ex in train_dataset.select(range(min(100, len(train_dataset)))):
    toks = tokenizer(ex["text"], return_length=True)
    lengths.append(toks["length"])

p95_length = sorted(lengths)[int(len(lengths) * 0.95)]
print(f"  95th percentile token length: {p95_length}")
print(f"  Max token length in sample  : {max(lengths)}")

# Set max_length to cover 95th percentile, capped at 2048
MAX_LENGTH = min(max(p95_length + 64, 512), 2048)
print(f"  Using max_length            : {MAX_LENGTH}")

# ================= SFT CONFIG =================
# KEY CHANGES:
#   - num_train_epochs  : 3 → 5   (more learning on small dataset)
#   - warmup_ratio      : added   (stabilizes early training)
#   - lr_scheduler_type : cosine  (better than linear for small datasets)
#   - load_best_model   : True    (saves checkpoint with lowest eval loss)
#   - metric_for_best   : eval_loss
#   - dataset_text_field: explicit (avoids silent TRL field detection bugs)

sft_config = SFTConfig(
    output_dir                  = OUTPUT_DIR,

    # ── Epochs & batch ───────────────────────────────────────────────────
    num_train_epochs            = 5,       # was 3 — more passes on 850 samples
    per_device_train_batch_size = 1,       # GTX 1650 safe
    per_device_eval_batch_size  = 1,
    gradient_accumulation_steps = 8,       # effective batch = 8

    # ── Learning rate schedule ────────────────────────────────────────────
    learning_rate               = 2e-4,
    warmup_ratio                = 0.05,    # 5% of steps = gentle warmup
    lr_scheduler_type           = "cosine",# cosine decay > linear for small data

    # ── Precision & memory ────────────────────────────────────────────────
    fp16                        = True,
    optim                       = "paged_adamw_8bit",  # 8-bit optimizer saves VRAM

    # ── Sequence length ───────────────────────────────────────────────────
    max_seq_length              = MAX_LENGTH,   # dynamic, covers your data
    packing                     = False,        # keep False — packing can mix Q&A pairs

    # ── Logging & saving ─────────────────────────────────────────────────
    logging_steps               = 5,
    save_strategy               = "epoch",
    eval_strategy               = "epoch",
    save_total_limit            = 2,
    load_best_model_at_end      = True,    # saves best checkpoint by eval loss
    metric_for_best_model       = "eval_loss",
    greater_is_better           = False,

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset_text_field          = "text",  # explicit — avoids silent TRL bugs

    report_to                   = "none",
)

# ================= TRAINER =================
print("\nInitializing SFTTrainer...")
trainer = SFTTrainer(
    model         = model,
    train_dataset = train_dataset,
    eval_dataset  = eval_dataset,
    peft_config   = lora_config,
    args          = sft_config,
)

# ── Print trainable parameter count ─────────────────────────────────────────
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"\n📊 Trainable params : {trainable:,} ({100*trainable/total:.2f}% of total)")

# ================= TRAIN =================
print("\n🚀 Starting QLoRA Fine-tuning...\n")
print(f"   Epochs          : {sft_config.num_train_epochs}")
print(f"   Effective batch : {sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps}")
print(f"   Learning rate   : {sft_config.learning_rate}")
print(f"   LR scheduler    : {sft_config.lr_scheduler_type}")
print(f"   Max seq length  : {MAX_LENGTH}")
print(f"   Train samples   : {len(train_dataset)}")
print()

trainer.train()

# ================= SAVE =================
print("\n💾 Saving best adapter + tokenizer...")
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\n✅ Fine-tuning complete!")
print(f"   Adapters saved at : {OUTPUT_DIR}")

# ── Print final eval loss ────────────────────────────────────────────────────
if trainer.state.best_metric is not None:
    print(f"   Best eval loss    : {trainer.state.best_metric:.4f}")
