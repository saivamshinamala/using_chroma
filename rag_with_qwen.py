# ================= OFFLINE SETTINGS (MUST BE FIRST) =================
import os

os.environ["TRANSFORMERS_OFFLINE"]   = "1"
os.environ["HF_HUB_OFFLINE"]         = "1"
os.environ["HF_DATASETS_OFFLINE"]    = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"]                = "./hf_cache"
os.environ["TRANSFORMERS_CACHE"]     = "./hf_cache"

# ================= IMPORTS =================
import re
import sys
import numpy as np
import torch
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path
from typing import List, Tuple, Optional

# ================= PATHS =================
BASE_DIR = Path(__file__).resolve().parent

BASE_MODEL_PATH    = r"E:\llms\Qwen2-1.5B-Instruct"
ADAPTER_PATH       = str(BASE_DIR / "models" / "qwen_qlora_adapters")
EMBED_PATH         = r"D:\Machine Learning and LLMs\LLMs\all-MiniLM-L6-v2"
CROSS_ENCODER_PATH = r"D:\Machine Learning and LLMs\LLMs\cross-encoder-ms-marco-MiniLM-L-6-v2"
CHROMA_DIR         = str(BASE_DIR / "chroma_store")
COLLECTION_NAME    = "pdf_markdown_embeddings"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ================= LOAD BI-ENCODER =================
print("Loading bi-encoder embedder (OFFLINE)...")
embedder = SentenceTransformer(EMBED_PATH, device="cuda", local_files_only=True)
print("✅ Bi-encoder loaded")

# ================= LOAD CROSS-ENCODER =================
print("Loading cross-encoder reranker (OFFLINE)...")
cross_encoder_available = False
reranker = None

try:
    reranker = CrossEncoder(
        CROSS_ENCODER_PATH,
        max_length=512,
        device=device,
    )
    cross_encoder_available = True
    print("✅ Cross-encoder reranker loaded")
except Exception as e:
    print(f"⚠  Cross-encoder not loaded: {e}")
    print("   Falling back to keyword-boost reranking.")

# ================= LOAD CHROMA =================
print("Loading ChromaDB...")
client     = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION_NAME)
print(f"✅ Collection loaded — {collection.count()} chunks")

space = (collection.metadata or {}).get("hnsw:space", "l2")
print(f"   Distance metric: {space}")
if space != "cosine":
    print(f"\n⛔  FATAL: Collection metric is '{space}', expected 'cosine'.\n")
    sys.exit(1)

# ================= LOAD FINE-TUNED LLM =================
print("Loading Qwen model...")
use_finetuned = os.path.exists(ADAPTER_PATH)

if use_finetuned:
    print("✓ Adapters found — loading with QLoRA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("✅ Fine-tuned Qwen loaded")
else:
    print("⚠  No adapters — loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
    )
    print("✅ Base Qwen loaded")

# ================= TOKENIZER =================
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH, local_files_only=True, trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id


# =============================================================================
#  FIX 4 — QUESTION TYPE DETECTION (procedure keyword checked BEFORE definition)
# =============================================================================

def detect_question_type(question: str) -> str:
    """
    Fixed order: procedure/list/comparison checked BEFORE definition.
    Previously "What is the procedure for..." was wrongly typed as 'definition'
    because 'what is' was checked first.
    """
    q = question.lower()

    # ── Explicit format requests — highest priority ───────────────────────
    if any(p in q for p in [
        "in points", "as points", "point by point", "in bullet", "as bullets",
        "bullet points", "numbered list", "as a list", "list format",
        "step by step", "in steps", "one by one", "enumerate",
    ]):
        return "points"

    # ── Procedure — checked BEFORE definition to catch "What is the procedure" ──
    if any(w in q for w in [
        "procedure", "process", "steps to", "how to", "how do", "how does",
        "switching on", "switch on", "turn on", "power on", "startup", "start up",
        "shut down", "shutdown", "reboot", "restart",
    ]):
        return "procedure"

    # ── List / safety — checked BEFORE definition ────────────────────────
    if any(w in q for w in [
        "list", "mention all", "what are all", "give all", "types of",
        "kinds of", "all precautions", "all safety", "all measures",
        "safety", "precautions", "guidelines", "requirements", "rules",
        "colour codes", "color codes", "codes for",
    ]):
        return "list"

    # ── Comparison ────────────────────────────────────────────────────────
    if any(w in q for w in ["compare", "difference", "vs", "versus", "distinguish", "contrast"]):
        return "comparison"

    # ── Definition ───────────────────────────────────────────────────────
    if any(q.startswith(p) for p in ["what is", "what are", "define", "meaning of"]):
        return "definition"

    # ── Explanation ──────────────────────────────────────────────────────
    if any(q.startswith(p) for p in ["explain", "describe", "tell me about", "overview", "elaborate"]):
        return "explanation"

    # ── Reason ───────────────────────────────────────────────────────────
    if any(w in q for w in ["why", "reason", "cause", "purpose", "benefit", "advantage"]):
        return "explanation"

    return "general"


# =============================================================================
#  RETRIEVAL — Bi-encoder stage 1
# =============================================================================

def biencoder_retrieve(
    query: str,
    k: int = 20,
    verbose: bool = True,
) -> Tuple[List[str], List[float], List[dict]]:
    q_vec = embedder.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True,
    ).tolist()

    if verbose:
        print(f"  Bi-encoder query vector norm: {np.linalg.norm(q_vec[0]):.4f}")

    results = collection.query(
        query_embeddings=q_vec,
        n_results=k,
        include=["documents", "distances", "metadatas"],
    )

    docs      = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results.get("metadatas", [[{}] * len(docs)])[0]
    sims      = [1.0 - (d / 2.0) for d in distances]

    if verbose:
        print(f"  Retrieved {len(docs)} candidates from ChromaDB")

    return docs, sims, metadatas


# =============================================================================
#  RETRIEVAL — Cross-encoder stage 2
# =============================================================================

def crossencoder_rerank(
    query: str,
    docs: List[str],
    bi_scores: List[float],
    top_k: int = 5,
    verbose: bool = True,
) -> Tuple[List[str], List[float]]:
    if not docs:
        return [], []

    pairs     = [(query, doc) for doc in docs]
    ce_scores = reranker.predict(pairs, show_progress_bar=False)

    if verbose:
        print(f"\n{'─'*70}")
        print(f"  {'Rank':<5} {'CE Score':<12} {'Bi-Sim':<10} {'Document preview':<40}")
        print(f"{'─'*70}")

    combined = []
    for doc, ce_score, bi_score in zip(docs, ce_scores, bi_scores):
        combined.append({
            "doc":      doc,
            "ce_score": float(ce_score),
            "bi_score": float(bi_score),
            "final":    float(ce_score) * 0.85 + float(bi_score) * 0.15,
        })

    combined.sort(key=lambda x: x["final"], reverse=True)

    if verbose:
        for rank, item in enumerate(combined[:top_k], 1):
            preview = item["doc"][:42] + "..." if len(item["doc"]) > 42 else item["doc"]
            print(f"  {rank:<5} {item['ce_score']:<12.4f} {item['bi_score']:<10.4f} {preview}")
        print(f"{'─'*70}")

    top_docs   = [item["doc"]      for item in combined[:top_k]]
    top_scores = [item["ce_score"] for item in combined[:top_k]]
    return top_docs, top_scores


def biencoder_rerank(
    query: str,
    docs: List[str],
    bi_scores: List[float],
    top_k: int = 5,
    threshold: float = 0.45,
    verbose: bool = True,
) -> Tuple[List[str], List[float]]:
    query_terms = set(query.lower().split())
    candidates  = []

    for doc, score in zip(docs, bi_scores):
        doc_lower = doc.lower()
        matches   = sum(1 for t in query_terms if len(t) > 3 and t in doc_lower)
        boost     = min(0.08 * matches, 0.2)
        candidates.append({"doc": doc, "score": min(score + boost, 1.0), "raw": score})

    candidates.sort(key=lambda x: x["score"], reverse=True)

    if verbose:
        print(f"\n{'─'*70}")
        for i, c in enumerate(candidates[:top_k], 1):
            preview = c["doc"][:42] + "..."
            print(f"  {i:<5} {c['score']:<10.4f} {c['raw']:<10.4f} {preview}")
        print(f"{'─'*70}")

    filtered = [c for c in candidates[:top_k] if c["score"] >= threshold]
    return [c["doc"] for c in filtered], [c["score"] for c in filtered]


def retrieve_and_rerank(
    query: str,
    candidate_k: int = 20,
    top_k: int = 5,
    threshold: float = 0.45,
    verbose: bool = True,
) -> Tuple[List[str], List[float]]:
    print(f"\n📡 Stage 1 — Bi-encoder retrieval (k={candidate_k})...")
    docs, bi_scores, _ = biencoder_retrieve(query, k=candidate_k, verbose=verbose)

    if not docs:
        return [], []

    if bi_scores and bi_scores[0] < 0.35:
        print(f"⚠  Very low bi-encoder score ({bi_scores[0]:.3f})")
        return [], []

    if cross_encoder_available:
        print(f"🎯 Stage 2 — Cross-encoder reranking (top_k={top_k})...")
        reranked_docs, reranked_scores = crossencoder_rerank(
            query, docs, bi_scores, top_k=top_k, verbose=verbose
        )
        # Drop only extremely low CE scores (< -5 = clearly irrelevant)
        filtered_docs   = [d for d, s in zip(reranked_docs, reranked_scores) if s > -5.0]
        filtered_scores = [s for s in reranked_scores if s > -5.0]
        print(f"✅ Cross-encoder selected {len(filtered_docs)} chunks")
        return filtered_docs, filtered_scores
    else:
        print(f"🔄 Stage 2 — Keyword-boost reranking (fallback)...")
        return biencoder_rerank(query, docs, bi_scores, top_k=top_k, threshold=threshold, verbose=verbose)


# =============================================================================
#  CONTEXT CLEANING
# =============================================================================

def clean_context(context: str, preserve_structure: bool = True) -> str:
    cleaned    = []
    lines      = context.splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            if preserve_structure and cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue
        lower = line.lower()
        if any(p in lower for p in [
            "page intentionally left blank",
            "this page intentionally",
            "continued on next page",
        ]):
            continue
        if re.match(r'^(\d+\.|\*|-|•|[A-Z][A-Z\s]+:)', line):
            cleaned.append(line)
            continue
        if any(c in line for c in ['=', ':', '(', ')', '%', '@']):
            cleaned.append(line)
            continue
        if len(line) < 20 and not any(c.isdigit() for c in line):
            continue
        cleaned.append(line)

    final      = []
    prev_empty = False
    for line in cleaned:
        if line == "":
            if not prev_empty:
                final.append(line)
            prev_empty = True
        else:
            final.append(line)
            prev_empty = False

    return "\n".join(final)


# =============================================================================
#  FIX 2 — CONTEXT RELEVANCE (relaxed for short single-term queries)
# =============================================================================

def validate_context_relevance(context: str, question: str) -> float:
    """
    Fixed: single-term queries like 'What is shakti?' or 'What is ES AHU-1?'
    were failing because only 1 meaningful term existed and it wasn't found
    verbatim in the chunk text.

    New approach:
    - If fewer than 2 meaningful terms in question, skip the check (return 1.0)
    - Use partial/substring matching for technical terms like "AHU-1"
    - Lower bar to 1 match being sufficient for short questions
    """
    question_terms = [t for t in question.lower().split() if len(t) > 3]
    context_lower  = context.lower()

    # Too short to judge — don't block on relevance
    if len(question_terms) < 2:
        return 1.0

    matches = sum(1 for t in question_terms if t in context_lower)

    # For short questions (2-3 terms), 1 match is enough
    if len(question_terms) <= 3:
        return 1.0 if matches >= 1 else 0.0

    return matches / len(question_terms)


# =============================================================================
#  PROMPT BUILDER
# =============================================================================

def build_prompt(
    context: str,
    question: str,
    qtype: str,
    similarity_score: Optional[float] = None,
) -> str:
    format_instructions = {
        "definition": (
            "Write a clear, precise definition. "
            "Use exact terms from the context. Keep it concise (2-4 sentences)."
        ),
        "explanation": (
            "Write a detailed explanation in clear paragraphs. "
            "Cover all key aspects mentioned in the context."
        ),
        "procedure": (
            "List EVERY step as a numbered list in the exact order described.\n"
            "1. First step\n2. Second step\n...\n"
            "Include ALL steps from the context — do not summarize or skip any."
        ),
        "points": (
            "Answer using a numbered list. Each point must be a complete, "
            "informative sentence drawn directly from the context.\n"
            "Format:\n1. [Point one]\n2. [Point two]\n3. [Point three]\n...\n"
            "Do NOT write paragraphs — use ONLY the list format. "
            "Include ALL relevant points from the context."
        ),
        "list": (
            "List ALL relevant items mentioned in the context without omitting any. "
            "Use a numbered format with a brief explanation for each item. "
            "Do NOT stop early — include every item the context mentions."
        ),
        "comparison": (
            "Compare the items by highlighting their key differences and similarities "
            "as described in the context. Use structured points."
        ),
        "general": (
            "Answer comprehensively using specific information from the context. "
            "Be direct and complete."
        ),
    }

    style_instruction = format_instructions.get(qtype, format_instructions["general"])

    confidence_note = ""
    if similarity_score is not None and similarity_score < 0.55:
        confidence_note = (
            "\nNote: Context relevance is moderate. "
            "Answer only what is explicitly supported."
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise technical assistant. Answer questions using "
                "ONLY the information in the provided context.\n\n"
                "STRICT RULES:\n"
                "1. Use ONLY facts explicitly stated in the context.\n"
                "2. Do NOT add outside knowledge or assumptions.\n"
                "3. Do NOT repeat yourself — each point or sentence must be unique.\n"
                "4. Stop generating as soon as you have covered all information "
                "   from the context. Do NOT pad or loop.\n"
                "5. If context is insufficient, say: 'The available context does not "
                "   contain sufficient information to answer this.'\n"
                "6. Mirror the vocabulary and terminology used in the context.\n\n"
                f"RESPONSE FORMAT: {style_instruction}"
                f"{confidence_note}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION: {question}\n\n"
                "ANSWER:"
            ),
        },
    ]

    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


# =============================================================================
#  FIX 3 — ANSWER CLEANING (preserve full lists, don't truncate at first period)
# =============================================================================

def clean_generated_answer(answer: str) -> str:
    # Strip chat template artifacts
    for marker in ["<|im_start|>assistant", "<|im_start|>", "<|im_end|>"]:
        if marker in answer:
            parts  = answer.split(marker)
            answer = parts[-1] if marker == "<|im_start|>assistant" else parts[0]

    answer = re.sub(r'^(assistant|system|user)\s*[:\n]', '', answer, flags=re.IGNORECASE).strip()

    # Remove question echo at start
    lines         = answer.splitlines()
    cleaned_lines = []
    skip_echo     = True
    for line in lines:
        stripped = line.strip()
        if skip_echo and (not stripped or stripped.endswith("?")):
            continue
        skip_echo = False
        cleaned_lines.append(line)

    answer = "\n".join(cleaned_lines).strip()
    answer = re.sub(r'\n{3,}', '\n\n', answer)
    return answer.strip()


# =============================================================================
#  FIX 1 — REPETITION DETECTION & REMOVAL
# =============================================================================

def remove_repetition(text: str) -> str:
    """
    Detects and removes looping repetitions in generated text.
    Handles both sentence-level and phrase-level loops.
    """
    if not text:
        return text

    # ── Sentence-level deduplication ─────────────────────────────────────
    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    seen_sentences = []
    unique_sentences = []

    for sentence in sentences:
        # Normalize for comparison (lowercase, strip whitespace)
        normalized = re.sub(r'\s+', ' ', sentence.lower().strip())
        if normalized and normalized not in seen_sentences:
            seen_sentences.append(normalized)
            unique_sentences.append(sentence)
        # Stop if we see the same sentence twice — everything after is likely a loop
        elif normalized and seen_sentences.count(normalized) >= 1:
            break

    text = " ".join(unique_sentences)

    # ── List item deduplication ───────────────────────────────────────────
    # Handle numbered lists: detect when item content repeats
    lines = text.splitlines()
    seen_content = set()
    deduped_lines = []

    for line in lines:
        stripped = line.strip()
        # For numbered list items, extract just the content after the number
        content_match = re.match(r'^\d+\.\s*(.+)', stripped)
        if content_match:
            content = content_match.group(1).lower().strip()
            # Truncate content for comparison (first 60 chars = enough to detect repeats)
            key = content[:60]
            if key not in seen_content:
                seen_content.add(key)
                deduped_lines.append(line)
            # else: skip this duplicate list item
        else:
            deduped_lines.append(line)

    text = "\n".join(deduped_lines)

    # ── Phrase-level loop detection ───────────────────────────────────────
    # If the same 8-word phrase appears 3+ times, truncate at first repetition
    words = text.split()
    if len(words) > 50:
        phrase_len  = 8
        phrase_seen = {}
        for i in range(len(words) - phrase_len):
            phrase = " ".join(words[i:i + phrase_len]).lower()
            if phrase in phrase_seen:
                count = phrase_seen[phrase]
                if count >= 2:
                    # Truncate at this point — everything after is a loop
                    text = " ".join(words[:i]).strip()
                    # Clean up trailing incomplete sentence
                    last_punct = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
                    if last_punct > len(text) * 0.5:
                        text = text[:last_punct + 1]
                    break
                phrase_seen[phrase] = count + 1
            else:
                phrase_seen[phrase] = 1

    return text.strip()


def validate_answer(answer: str) -> bool:
    if not answer or len(answer.split()) < 5:
        return False
    refusal_phrases = [
        "as an ai language model",
        "i don't have access to",
        "based on my training data",
        "i cannot provide information",
        "my knowledge cutoff",
    ]
    if any(p in answer.lower() for p in refusal_phrases):
        return False
    if len(answer.strip()) < 20:
        return False
    return True


def ensure_complete_sentence(text: str) -> str:
    """
    Only add period if needed. NEVER truncate — this was causing list answers
    to be cut short at the first sentence boundary.
    """
    if not text:
        return text
    text = text.strip()
    if text and text[-1] not in ".!?":
        # Don't truncate — just add a period if text is substantial
        if len(text.split()) > 5:
            return text + "."
    return text


# =============================================================================
#  FIX 1 — GENERATION (stronger repetition control)
# =============================================================================

def generate_answer(
    context: str,
    question: str,
    similarity_score: Optional[float] = None,
) -> str:
    """
    Fixed generation parameters:
    - repetition_penalty raised from 1.1 → 1.3 (stronger anti-loop)
    - temperature raised from 0.4 → 0.5 (more variety, less token lock-in)
    - no_repeat_ngram_size=4 added (prevents any 4-gram from repeating)
    - Post-generation repetition removal as safety net
    """
    qtype   = detect_question_type(question)
    context = clean_context(context, preserve_structure=True)

    if not context.strip() or len(context.split()) < 15:
        return "Insufficient context available to answer this question."

    relevance = validate_context_relevance(context, question)
    if relevance < 0.05:
        return "The retrieved context does not appear to contain relevant information for this question."

    prompt = build_prompt(context, question, qtype, similarity_score)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=600,

            # ── Anti-repetition (KEY FIXES) ───────────────────────────────
            repetition_penalty=1.3,       # was 1.1 — much stronger penalty
            no_repeat_ngram_size=4,       # ADDED — no 4-gram can repeat at all

            # ── Sampling ─────────────────────────────────────────────────
            do_sample=True,
            temperature=0.5,              # was 0.4 — slightly more variety
            top_p=0.90,
            top_k=50,

            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # Clean chat template artifacts
    answer = clean_generated_answer(answer)

    # Post-processing: remove any remaining repetition loops
    answer = remove_repetition(answer)

    if not validate_answer(answer):
        return "Unable to provide a reliable answer based on the available context."

    return answer


# =============================================================================
#  MAIN RAG PIPELINE
# =============================================================================

def ask(
    question: str,
    candidate_k: int = 20,
    top_k: int = 5,
    threshold: float = 0.45,
) -> str:
    print(f"\n{'='*60}")
    print(f"QUERY : {question}")
    print(f"TYPE  : {detect_question_type(question)}")
    print(f"RERANK: {'Cross-encoder' if cross_encoder_available else 'Keyword-boost (fallback)'}")
    print('='*60)

    docs, scores = retrieve_and_rerank(
        question,
        candidate_k=candidate_k,
        top_k=top_k,
        threshold=threshold,
        verbose=True,
    )

    if not docs:
        print(f"⚠  No results — retrying with larger pool (k={candidate_k + 10})...")
        docs, scores = retrieve_and_rerank(
            question,
            candidate_k=candidate_k + 10,
            top_k=top_k + 3,
            threshold=max(threshold - 0.10, 0.25),
            verbose=False,
        )

    if not docs:
        return "I don't have sufficient information in the knowledge base to answer this question."

    # Deduplicate chunks
    seen        = set()
    unique_docs = []
    for doc in docs:
        key = doc[:100]
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    context = "\n\n---\n\n".join(unique_docs)

    if len(context.split()) < 20:
        return "The retrieved information is too limited to provide a reliable answer."

    avg_score = sum(scores) / len(scores) if scores else 0
    answer    = generate_answer(context, question, avg_score)

    return ensure_complete_sentence(answer)


# =============================================================================
#  STARTUP DIAGNOSTIC
# =============================================================================

def run_startup_diagnostic():
    print("\n" + "="*60)
    print(" STARTUP DIAGNOSTIC")
    print("="*60)

    try:
        sample = collection.get(limit=10, include=["embeddings", "documents"])
        if not sample["embeddings"]:
            print("❌ ERROR: No embeddings found!")
            return

        embed_dim = len(sample["embeddings"][0])
        norms     = [np.linalg.norm(e) for e in sample["embeddings"]]
        print(f"✓ Embedding dimension : {embed_dim}")
        print(f"✓ Vector norms        : mean={np.mean(norms):.4f}, std={np.std(norms):.4f}")

        if not all(0.95 < n < 1.05 for n in norms[:5]):
            print("⚠ Vectors may not be properly normalized!")
        else:
            print("✓ Vectors are properly normalized")

        print("\n📊 Self-Retrieval Test:")
        test_doc     = sample["documents"][0]
        test_snippet = test_doc[:200]
        q_vec = embedder.encode(
            [test_snippet], normalize_embeddings=True, convert_to_numpy=True
        ).tolist()
        res = collection.query(
            query_embeddings=q_vec, n_results=3,
            include=["documents", "distances"]
        )
        if res["documents"][0]:
            dist = res["distances"][0][0]
            sim  = 1.0 - (dist / 2.0)
            print(f"  Distance: {dist:.4f}  |  Similarity: {sim:.4f}")
            if res["documents"][0][0][:100] == test_doc[:100]:
                print("✓ Exact match — retrieval working correctly")
            elif sim > 0.9:
                print("✓ High similarity match — retrieval working")
            else:
                print("⚠ Low self-retrieval similarity — check embedding model")

        if cross_encoder_available:
            print("\n🎯 Cross-Encoder Test:")
            test_pairs = [
                ("What is the system configuration?", test_doc[:300]),
                ("What is the system configuration?", "Unrelated text about apples."),
            ]
            ce_scores = reranker.predict(test_pairs, show_progress_bar=False)
            print(f"  Relevant doc score   : {ce_scores[0]:.4f}")
            print(f"  Irrelevant doc score : {ce_scores[1]:.4f}")
            if ce_scores[0] > ce_scores[1]:
                print("✓ Cross-encoder correctly ranks relevant doc higher")
            else:
                print("⚠ Cross-encoder ranking unexpected — check model")
        else:
            print("\n⚠  Cross-encoder not available — using fallback reranker")

        print("\n" + "="*60)
        print(" DIAGNOSTIC COMPLETE")
        print("="*60 + "\n")

    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")


# =============================================================================
#  MAIN
# =============================================================================

if __name__ == "__main__":
    run_startup_diagnostic()

    model_type  = "Fine-tuned" if use_finetuned else "Base"
    rerank_type = "Cross-encoder" if cross_encoder_available else "Keyword-boost (fallback)"

    print(f"\n✅ RAG System Ready")
    print(f"   Model    : {model_type} Qwen2-1.5B")
    print(f"   Device   : {device.upper()}")
    print(f"   Chunks   : {collection.count()}")
    print(f"   Reranker : {rerank_type}")
    print("\nType 'exit' or 'quit' to stop.")
    print("Type 'help' for usage tips.\n")

    while True:
        try:
            q = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not q:
            continue
        if q.lower() in ("exit", "quit", "q"):
            print("Goodbye!")
            break
        if q.lower() == "help":
            print("\nUsage Tips:")
            print("  - Ask specific questions about the content in your PDF/docs")
            print("  - You can ask 'explain in points', 'list all', 'step by step'")
            print("  - Use keywords from your training material\n")
            continue

        answer = ask(q, candidate_k=20, top_k=5, threshold=0.45)
        print(f"\n{'─'*60}")
        print(f"Answer:\n{answer}")
        print(f"{'─'*60}\n")
