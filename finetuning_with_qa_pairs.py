# qlora_finetune_verbose.py

import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq, TrainerCallback, TrainerState, TrainerControl
from peft import LoraConfig, get_peft_model, TaskType
import torch

# ------------------------------
# 1️⃣ Load your JSONL Q&A data
# ------------------------------
jsonl_file = r"D:\RAG Projects\vamshi\RAFT\data\QAPairs\qa_pairs.jsonl"  # Path to your JSONL file

data = []
with open(jsonl_file, "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        prompt = f"Question: {item['question']}\nAnswer:"
        completion = f" {item['answer']}"
        data.append({"input_text": prompt, "target_text": completion})

dataset = Dataset.from_list(data)

print(f"✅ Total examples loaded: {len(dataset)}")
print(f"Sample Q&A pair:\n{dataset[0]}")

# ------------------------------
# 2️⃣ Load tokenizer and model locally
# ------------------------------
model_path = r"D:/Machine Learning and LLMs/LLMs/Mistral-7B-Instruct-v0.2"  # Path to your downloaded Mistral model
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    load_in_8bit=True  # QLoRA works with 8-bit or 4-bit
)

print(f"✅ Model loaded: {model.__class__.__name__}")
print(f"✅ CUDA available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}")

# ------------------------------
# 3️⃣ Set up QLoRA LoRA config
# ------------------------------
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    dtype="float16"
)

model = get_peft_model(model, lora_config)
print("✅ LoRA adapters applied.")
model.print_trainable_parameters()

# ------------------------------
# 4️⃣ Tokenize dataset
# ------------------------------
max_input_length = 512
max_target_length = 256

def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=max_input_length,
        truncation=True
    )
    labels = tokenizer(
        examples["target_text"],
        max_length=max_target_length,
        truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["input_text", "target_text"])
print(f"✅ Tokenization complete. Sample tokenized example:\n{tokenized_dataset[0]}")

# ------------------------------
# 5️⃣ Data collator
# ------------------------------
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# ------------------------------
# 6️⃣ Custom callback for per-step loss
# ------------------------------
class PrintLossCallback(TrainerCallback):
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.global_step % 1 == 0:  # print every step
            logs = {k: v for k, v in state.log_history[-1].items()} if state.log_history else {}
            print(f"Step {state.global_step}, Logs: {logs}")

# ------------------------------
# 7️⃣ Training arguments
# ------------------------------
training_args = TrainingArguments(
    output_dir="./mistral_qlora_qa",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=True,
    logging_steps=1,   # log every step
    save_steps=500,
    save_total_limit=2,
    remove_unused_columns=False,
    report_to="none"
)

# ------------------------------
# 8️⃣ Trainer
# ------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[PrintLossCallback]
)

print("🚀 Starting QLoRA fine-tuning now...")
trainer.train()

# ------------------------------
# 9️⃣ Save LoRA adapters
# ------------------------------
model.save_pretrained("./mistral_qlora_qa")
tokenizer.save_pretrained("./mistral_qlora_qa")

print("✅ QLoRA fine-tuning complete! LoRA adapters saved in './mistral_qlora_qa'")
