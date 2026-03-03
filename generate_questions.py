"""
Local Training Data Augmentation (No API Required)
=====================================================
Uses your already-downloaded Qwen2 model to augment your JSONL training data.
No internet, no API key, no cost.

Two modes:
  --mode qwen     : Use your fine-tuned / base Qwen model (recommended)
  --mode template : Pure rule-based, no model needed (fastest, good baseline)

Usage:
    # Mode 1: Use Qwen model (best quality)
    python augment_local.py --input train.jsonl --output train_aug.jsonl --mode qwen

    # Mode 2: Template-based only (no GPU needed, instant)
    python augment_local.py --input train.jsonl --output train_aug.jsonl --mode template

    # Control number of variants
    python augment_local.py --input train.jsonl --output train_aug.jsonl --mode qwen --variants 6
"""

import json
import re
import random
import argparse
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"]       = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"]               = "./hf_cache"
os.environ["TRANSFORMERS_CACHE"]    = "./hf_cache"

# ── Paths — same as your rag_pipeline.py ─────────────────────────────────────
BASE_MODEL_PATH = r"E:\llms\Qwen2-1.5B-Instruct"
ADAPTER_PATH    = str(Path(__file__).resolve().parent / "models" / "qwen_qlora_adapters")

# ═════════════════════════════════════════════════════════════════════════════
#  PART 1 — TEMPLATE-BASED AUGMENTATION (no model, works instantly)
# ═════════════════════════════════════════════════════════════════════════════

# Question opener templates
QUESTION_OPENERS = [
    "What is {topic}?",
    "What are {topic}?",
    "Explain {topic}.",
    "Explain {topic} in detail.",
    "Describe {topic}.",
    "Tell me about {topic}.",
    "Give an overview of {topic}.",
    "Can you explain {topic}?",
    "Provide information about {topic}.",
    "Elaborate on {topic}.",
    "What do you know about {topic}?",
    "Define {topic}.",
    "Summarize {topic}.",
]

POINTS_OPENERS = [
    "Explain {topic} in points.",
    "List the key aspects of {topic}.",
    "Explain {topic} as a numbered list.",
    "What are the main points about {topic}?",
    "Describe {topic} point by point.",
    "Give me {topic} in bullet points.",
    "Break down {topic} into points.",
    "List everything about {topic}.",
    "Enumerate the features of {topic}.",
    "What are the components of {topic}? List them.",
]


def extract_topic_from_question(question: str) -> str:
    """
    Heuristically extract the core topic from a question.
    e.g. "What is the system configuration of Shakti EW?" -> "the system configuration of Shakti EW"
    """
    q = question.strip().rstrip("?.")

    # Remove common question starters
    patterns = [
        r"^(what is|what are|explain|describe|tell me about|define|"
        r"give an overview of|how does|how do|why is|elaborate on|"
        r"can you explain|provide information about|summarize)\s+",
    ]
    for pat in patterns:
        q = re.sub(pat, "", q, flags=re.IGNORECASE).strip()

    return q if q else question


def answer_to_points(answer: str) -> str:
    """
    Convert a paragraph answer into a numbered list.
    Splits on sentences and formats as 1. 2. 3. ...
    """
    # If already a numbered list, return as-is
    if re.match(r'^\s*\d+\.', answer.strip()):
        return answer

    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    sentences = [s.strip() for s in sentences if s.strip() and len(s.split()) > 4]

    if not sentences:
        return answer

    # Merge very short sentences with the previous one
    merged = []
    buffer = ""
    for s in sentences:
        if len(s.split()) < 6 and buffer:
            buffer += " " + s
        else:
            if buffer:
                merged.append(buffer)
            buffer = s
    if buffer:
        merged.append(buffer)

    if len(merged) < 2:
        # Can't make a meaningful list — try splitting on commas/semicolons
        parts = re.split(r'[;,]\s+', answer)
        parts = [p.strip().capitalize() for p in parts if len(p.split()) > 3]
        if len(parts) >= 2:
            merged = parts

    if len(merged) < 2:
        return answer  # Not splittable — return original

    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(merged))
    return numbered


def make_concise(answer: str) -> str:
    """
    Return just the first 1-2 sentences as a concise answer.
    """
    sentences = re.split(r'(?<=[.!?])\s+', answer.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    return " ".join(sentences[:2]) if sentences else answer


def make_detailed(answer: str) -> str:
    """
    Expand the answer slightly by adding a closing summary sentence.
    """
    if not answer.strip().endswith('.'):
        answer = answer.strip() + "."
    # Add a simple closing line
    closing_options = [
        " This covers the essential aspects as defined in the system documentation.",
        " These details are as specified in the technical reference.",
        " The above information summarizes the key specifications.",
    ]
    return answer + random.choice(closing_options)


def template_augment_single(
    question: str,
    answer: str,
    system_prompt: str,
    num_variants: int = 6
) -> List[Dict]:
    """
    Generate variants using pure templates — no model needed.
    Always produces:
      - 2 points/bullet variants
      - 1 concise variant
      - 1 detailed variant
      - remaining: rephrased question variants
    """
    topic    = extract_topic_from_question(question)
    variants = []

    # ── 2 point-format variants ───────────────────────────────────────────
    points_answer = answer_to_points(answer)
    for opener in random.sample(POINTS_OPENERS, min(2, len(POINTS_OPENERS))):
        q = opener.format(topic=topic)
        variants.append(_make_record(system_prompt, q, points_answer))

    # ── 1 concise variant ─────────────────────────────────────────────────
    concise_q = random.choice([
        f"Briefly, what is {topic}?",
        f"In short, explain {topic}.",
        f"Give a brief answer: what is {topic}?",
    ])
    variants.append(_make_record(system_prompt, concise_q, make_concise(answer)))

    # ── 1 detailed variant ────────────────────────────────────────────────
    detailed_q = random.choice([
        f"Explain {topic} in detail.",
        f"Provide a comprehensive explanation of {topic}.",
        f"Describe {topic} thoroughly.",
    ])
    variants.append(_make_record(system_prompt, detailed_q, make_detailed(answer)))

    # ── Fill remaining slots with rephrased question + original answer ────
    remaining = num_variants - len(variants)
    openers   = [o for o in QUESTION_OPENERS if o.format(topic=topic).lower() != question.lower()]
    random.shuffle(openers)

    for opener in openers[:remaining]:
        q = opener.format(topic=topic)
        variants.append(_make_record(system_prompt, q, answer))

    return variants[:num_variants]


def _make_record(system_prompt: str, question: str, answer: str) -> Dict:
    return {
        "messages": [
            {"role": "system",    "content": system_prompt},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


# ═════════════════════════════════════════════════════════════════════════════
#  PART 2 — QWEN MODEL AUGMENTATION (better quality, requires GPU)
# ═════════════════════════════════════════════════════════════════════════════

def load_qwen_model():
    """Load Qwen model + tokenizer for local augmentation."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Qwen on {device}...")

    use_adapter = os.path.exists(ADAPTER_PATH)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH, local_files_only=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if use_adapter:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH, quantization_config=bnb,
            device_map="auto", local_files_only=True, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, ADAPTER_PATH)
        print("✅ Fine-tuned Qwen loaded")
    else:
        import torch
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH, torch_dtype=torch.float16,
            device_map="auto", local_files_only=True, trust_remote_code=True
        )
        print("✅ Base Qwen loaded")

    return model, tokenizer, device


def qwen_generate(model, tokenizer, device, prompt: str, max_new_tokens: int = 600) -> str:
    """Run one generation pass with Qwen."""
    import torch
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,   # Higher temp for more variety
            top_p=0.92,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text


def build_qwen_augment_prompt(tokenizer, question: str, answer: str, variant_instruction: str) -> str:
    """Build a prompt asking Qwen to rewrite a Q&A pair."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a training data assistant. You rephrase Q&A pairs "
                "without changing the facts. Follow the instruction exactly. "
                "Return ONLY the new question and answer — nothing else."
            )
        },
        {
            "role": "user",
            "content": (
                f"Original Question: {question}\n"
                f"Original Answer: {answer}\n\n"
                f"Instruction: {variant_instruction}\n\n"
                "Return in this exact format:\n"
                "QUESTION: <your rephrased question>\n"
                "ANSWER: <your reformatted answer>"
            )
        }
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


# Variant instructions passed to Qwen
QWEN_VARIANT_INSTRUCTIONS = [
    "Rephrase the question starting with 'Explain' and keep the same answer.",
    "Rephrase the question starting with 'Describe' and keep the same answer.",
    "Rephrase the question as 'Tell me about...' and keep the same answer.",
    "Rephrase the question starting with 'Give an overview of' and keep the same answer.",
    "Rewrite the question asking for 'a numbered list' or 'in points', and reformat the answer as a numbered list (1. 2. 3. ...).",
    "Rewrite the question asking 'Explain step by step' and reformat the answer as numbered steps.",
    "Make the question very concise (under 8 words) and shorten the answer to 1-2 sentences.",
    "Rephrase the question starting with 'What are the key features of' and keep the same answer.",
    "Rephrase the question starting with 'Can you elaborate on' and keep the same answer.",
    "Rephrase the question as 'How would you describe' and keep the same answer.",
]


def parse_qwen_output(text: str) -> Optional[Tuple[str, str]]:
    """Parse QUESTION: / ANSWER: format from Qwen output."""
    q_match = re.search(r'QUESTION:\s*(.+?)(?=\nANSWER:|\Z)', text, re.DOTALL | re.IGNORECASE)
    a_match = re.search(r'ANSWER:\s*(.+?)$', text, re.DOTALL | re.IGNORECASE)

    if q_match and a_match:
        q = q_match.group(1).strip()
        a = a_match.group(1).strip()
        if q and a and len(a.split()) > 3:
            return q, a

    return None


def qwen_augment_single(
    model, tokenizer, device,
    question: str, answer: str, system_prompt: str,
    num_variants: int = 6
) -> List[Dict]:
    """
    Use Qwen to generate augmented variants.
    Falls back to template augmentation if Qwen output can't be parsed.
    """
    variants = []
    instructions = random.sample(
        QWEN_VARIANT_INSTRUCTIONS,
        min(num_variants, len(QWEN_VARIANT_INSTRUCTIONS))
    )

    # Always include at least 2 points variants
    points_instructions = [i for i in QWEN_VARIANT_INSTRUCTIONS if "numbered" in i or "points" in i or "step" in i]
    other_instructions  = [i for i in instructions if i not in points_instructions]
    final_instructions  = points_instructions[:2] + other_instructions[:max(0, num_variants-2)]
    final_instructions  = final_instructions[:num_variants]

    for instruction in final_instructions:
        prompt = build_qwen_augment_prompt(tokenizer, question, answer, instruction)

        try:
            output = qwen_generate(model, tokenizer, device, prompt, max_new_tokens=400)
            parsed = parse_qwen_output(output)

            if parsed:
                new_q, new_a = parsed
                variants.append(_make_record(system_prompt, new_q, new_a))
            else:
                # Fallback: use template for this slot
                topic = extract_topic_from_question(question)
                if "numbered" in instruction or "points" in instruction:
                    fb_q = f"Explain {topic} in points."
                    fb_a = answer_to_points(answer)
                else:
                    fb_q = random.choice(QUESTION_OPENERS).format(topic=topic)
                    fb_a = answer
                variants.append(_make_record(system_prompt, fb_q, fb_a))

        except Exception as e:
            print(f"      ⚠ Qwen generation error: {e} — using template fallback")
            topic = extract_topic_from_question(question)
            fb_q  = random.choice(QUESTION_OPENERS).format(topic=topic)
            variants.append(_make_record(system_prompt, fb_q, answer))

    return variants


# ═════════════════════════════════════════════════════════════════════════════
#  PART 3 — SHARED UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def extract_qa(record: Dict) -> Optional[Tuple[str, str]]:
    messages = record.get("messages", [])
    question = None
    answer   = None
    for msg in messages:
        role    = msg.get("role", "")
        content = msg.get("content", "").strip()
        if role == "user" and not question:
            question = content
        elif role == "assistant" and not answer:
            answer = content
    return (question, answer) if (question and answer) else None


def extract_system_prompt(record: Dict) -> str:
    for msg in record.get("messages", []):
        if msg.get("role") == "system":
            return msg["content"]
    return "You are a helpful technical assistant. Answer using only the provided context."


def validate_output(output_path: str):
    path = Path(output_path)
    if not path.exists():
        return
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except:
                    pass
    valid = sum(
        1 for r in records
        if any(m["role"] == "user"      for m in r.get("messages", []))
        and any(m["role"] == "assistant" for m in r.get("messages", []))
    )
    print(f"\n🔍 Validation: {len(records)} total records")
    print(f"   ✅ Valid    : {valid}")
    print(f"   ❌ Invalid  : {len(records) - valid}")
    if valid == len(records):
        print("   All records are fine — ready for fine-tuning!\n")


# ═════════════════════════════════════════════════════════════════════════════
#  PART 4 — MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def augment_dataset(
    input_path: str,
    output_path: str,
    mode: str = "template",
    num_variants: int = 6,
    keep_originals: bool = True,
):
    input_file  = Path(input_path)
    output_file = Path(output_path)

    if not input_file.exists():
        print(f"❌ Input file not found: {input_path}")
        return

    # Load records
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"⚠ Skipping line {line_num}: {e}")

    print(f"\n{'='*60}")
    print(f"  LOCAL TRAINING DATA AUGMENTATION")
    print(f"{'='*60}")
    print(f"  Mode           : {mode.upper()}")
    print(f"  Input records  : {len(records)}")
    print(f"  Variants each  : {num_variants}")
    print(f"  Expected output: ~{len(records) * (num_variants + 1)} records")
    print(f"{'='*60}\n")

    # Load model if needed
    model = tokenizer = device = None
    if mode == "qwen":
        try:
            model, tokenizer, device = load_qwen_model()
        except Exception as e:
            print(f"⚠ Could not load Qwen ({e}). Falling back to template mode.")
            mode = "template"

    all_output = []
    skipped    = 0

    for idx, record in enumerate(records, 1):
        qa = extract_qa(record)
        if not qa:
            print(f"[{idx:>3}/{len(records)}] ⚠ Skipping — could not extract Q&A")
            skipped += 1
            continue

        question, answer = qa
        system_prompt    = extract_system_prompt(record)

        print(f"[{idx:>3}/{len(records)}] {question[:70]}{'...' if len(question)>70 else ''}")

        if keep_originals:
            all_output.append(record)

        if mode == "qwen":
            variants = qwen_augment_single(
                model, tokenizer, device,
                question, answer, system_prompt, num_variants
            )
        else:
            variants = template_augment_single(
                question, answer, system_prompt, num_variants
            )

        all_output.extend(variants)
        print(f"           ✅ +{len(variants)} variants  |  total: {len(all_output)}")

    # Write output
    with open(output_file, "w", encoding="utf-8") as f:
        for r in all_output:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n{'='*60}")
    print(f"  AUGMENTATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Input          : {len(records)} records")
    print(f"  Skipped        : {skipped}")
    print(f"  Output         : {len(all_output)} records")
    print(f"  Expansion      : {len(all_output)/max(len(records),1):.1f}x")
    print(f"  Saved to       : {output_path}")
    print(f"{'='*60}")

    validate_output(output_path)

    # Show 2 sample augmented records
    augmented = [r for r in all_output if r not in records[:1]]
    if augmented:
        print("\n📋 Sample augmented records:\n")
        for sample in augmented[:2]:
            print("-" * 50)
            for msg in sample["messages"]:
                role    = msg["role"].upper()
                content = msg["content"]
                preview = content[:160] + "..." if len(content) > 160 else content
                print(f"[{role}]\n{preview}\n")


# ═════════════════════════════════════════════════════════════════════════════
#  CLI
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Augment JSONL training data locally — no API needed"
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Path to input .jsonl file")
    parser.add_argument("--output", "-o", required=True,
                        help="Path to output .jsonl file")
    parser.add_argument("--mode", "-m", choices=["qwen", "template"], default="template",
                        help="qwen = use your local Qwen model (better quality) | "
                             "template = rule-based only, no GPU needed (default: template)")
    parser.add_argument("--variants", "-v", type=int, default=6,
                        help="Variants per Q&A pair (default: 6)")
    parser.add_argument("--no-originals", action="store_true",
                        help="Exclude originals from output")

    args = parser.parse_args()

    augment_dataset(
        input_path    = args.input,
        output_path   = args.output,
        mode          = args.mode,
        num_variants  = args.variants,
        keep_originals= not args.no_originals,
    )