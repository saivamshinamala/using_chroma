# ================= OFFLINE SETTINGS (MUST BE FIRST) =================
import os

os.environ["TRANSFORMERS_OFFLINE"]  = "1"
os.environ["HF_HUB_OFFLINE"]        = "1"
os.environ["HF_DATASETS_OFFLINE"]   = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"]               = "./hf_cache"
os.environ["TRANSFORMERS_CACHE"]    = "./hf_cache"

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

BASE_MODEL_PATH    = r"E:\llms\Qwen2-7B-Instruct"
ADAPTER_PATH       = str(BASE_DIR / "models" / "qwen_qlora_adapters")
EMBED_PATH         = r"D:\Machine Learning and LLMs\LLMs\bge-large-en-v1.5"
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
    reranker = CrossEncoder(CROSS_ENCODER_PATH, max_length=512, device=device)
    cross_encoder_available = True
    print("✅ Cross-encoder reranker loaded")
except Exception as e:
    print(f"⚠  Cross-encoder not loaded: {e}")
    print("   Falling back to bi-encoder + keyword boost reranking.")

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
#  ⭐ IMPROVEMENT 1: QUERY EXPANSION
#  Generate 2 alternative phrasings of the user's question before retrieval.
#  Different phrasings hit different parts of the embedding space →
#  broader recall, especially for short or ambiguous questions.
# =============================================================================

def expand_query(question: str) -> List[str]:
    """
    Returns [original_question, variant1, variant2].
    Uses a lightweight local prompt — no extra model needed.
    """
    expansion_prompt = [
        {
            "role": "system",
            "content": (
                "You are a query rewriter. Given a question, output exactly 2 "
                "alternative phrasings that preserve the meaning but use different words. "
                "Output ONLY the 2 alternatives, one per line. No numbering, no explanation."
            ),
        },
        {"role": "user", "content": f"Question: {question}"},
    ]

    prompt = tokenizer.apply_chat_template(
        expansion_prompt, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,          # greedy — fast + deterministic
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_tokens = out[0][inputs["input_ids"].shape[-1]:]
    raw        = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    variants = [line.strip() for line in raw.splitlines() if line.strip()][:2]

    # Always include the original; pad if model gave fewer than 2 variants
    queries = [question] + variants
    if len(queries) < 3:
        queries.append(question)   # duplicate original as fallback

    print(f"  Expanded queries: {queries}")
    return queries[:3]


# =============================================================================
#  ⭐ IMPROVEMENT 2: HyDE  (Hypothetical Document Embedding)
#  Generate a short hypothetical answer, then embed IT instead of the question.
#  This bridges the query-document vocabulary gap — embedding a fake answer
#  lands much closer to real answers in the embedding space than the question alone.
# =============================================================================

def generate_hypothetical_answer(question: str) -> str:
    """Generate a short hypothetical answer for HyDE retrieval."""
    hyde_prompt = [
        {
            "role": "system",
            "content": (
                "Write a short, factual paragraph (3-5 sentences) that would be "
                "a plausible answer to the question below. "
                "Use technical language typical of documentation or manuals."
            ),
        },
        {"role": "user", "content": question},
    ]

    prompt = tokenizer.apply_chat_template(
        hyde_prompt, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.5,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen_tokens = out[0][inputs["input_ids"].shape[-1]:]
    hyp_answer = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    print(f"  HyDE answer (preview): {hyp_answer[:100]}...")
    return hyp_answer


# =============================================================================
#  RETRIEVAL STAGE 1 — Multi-query bi-encoder retrieval
# =============================================================================

def biencoder_retrieve(
    queries: List[str],          # ⭐ accepts multiple queries now
    k: int = 20,
    verbose: bool = True,
) -> Tuple[List[str], List[float], List[dict]]:
    """
    Stage 1: Encode multiple query variants and merge their results.
    Deduplication ensures each chunk appears at most once (best score kept).
    """
    # Encode all queries in one batch
    q_vecs = embedder.encode(
        queries,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    merged_docs     = {}   # doc_key → (doc, best_similarity, metadata)

    for q_vec in q_vecs:
        results    = collection.query(
            query_embeddings=[q_vec],
            n_results=k,
            include=["documents", "distances", "metadatas"],
        )
        docs      = results["documents"][0]
        distances = results["distances"][0]
        metadatas = results.get("metadatas", [[{}] * len(docs)])[0]

        for doc, dist, meta in zip(docs, distances, metadatas):
            sim = 1.0 - (dist / 2.0)
            key = doc[:120]
            if key not in merged_docs or sim > merged_docs[key][1]:
                merged_docs[key] = (doc, sim, meta)

    # Sort by best score across all queries
    sorted_items = sorted(merged_docs.values(), key=lambda x: x[1], reverse=True)
    docs      = [x[0] for x in sorted_items]
    sims      = [x[1] for x in sorted_items]
    metas     = [x[2] for x in sorted_items]

    if verbose:
        print(f"  Multi-query retrieved {len(docs)} unique candidates")

    return docs, sims, metas


# =============================================================================
#  RETRIEVAL STAGE 2 — Cross-encoder reranking (unchanged logic, improved input)
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
        print(f"\n{'─'*70}")
        print(f"  {'Rank':<5} {'CE Score':<12} {'Bi-Sim':<10} {'Document preview':<40}")
        print(f"{'─'*70}")
        for rank, item in enumerate(combined[:top_k], 1):
            preview = item["doc"][:42] + "..." if len(item["doc"]) > 42 else item["doc"]
            print(f"  {rank:<5} {item['ce_score']:<12.4f} {item['bi_score']:<10.4f} {preview}")
        print(f"{'─'*70}")

    top_docs   = [item["doc"]      for item in combined[:top_k]]
    top_scores = [item["ce_score"] for item in combined[:top_k]]
    return top_docs, top_scores


# =============================================================================
#  RETRIEVAL STAGE 2 (fallback) — keyword-boosted bi-encoder reranking
# =============================================================================

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
            preview = c["doc"][:42] + "..." if len(c["doc"]) > 42 else c["doc"]
            print(f"  {i:<5} {c['score']:<10.4f} {c['raw']:<10.4f} {preview}")
        print(f"{'─'*70}")

    filtered = [c for c in candidates[:top_k] if c["score"] >= threshold]
    return [c["doc"] for c in filtered], [c["score"] for c in filtered]


# =============================================================================
#  ⭐ IMPROVEMENT 3: MMR (Maximal Marginal Relevance) for context diversity
#  Standard retrieval returns the top-k MOST SIMILAR chunks — which often means
#  highly redundant chunks from the same passage.
#  MMR balances relevance (similar to query) vs diversity (dissimilar to
#  already-selected chunks) → each chunk adds new information.
# =============================================================================

def mmr_select(
    query: str,
    docs: List[str],
    scores: List[float],
    top_k: int = 5,
    lambda_param: float = 0.6,   # 0 = max diversity, 1 = max relevance
) -> Tuple[List[str], List[float]]:
    """
    Maximal Marginal Relevance selection.
    lambda_param=0.6 means 60% relevance, 40% diversity.
    """
    if not docs or top_k >= len(docs):
        return docs[:top_k], scores[:top_k]

    # Encode all docs at once
    doc_vecs   = embedder.encode(docs, normalize_embeddings=True, convert_to_numpy=True)
    query_vec  = embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]

    # Relevance scores from bi/cross encoder (already computed)
    rel_scores = np.array(scores)

    selected_idx = []
    remaining    = list(range(len(docs)))

    for _ in range(min(top_k, len(docs))):
        if not remaining:
            break

        if not selected_idx:
            # First selection: pick highest relevance
            best = max(remaining, key=lambda i: rel_scores[i])
        else:
            # MMR score = lambda * relevance - (1-lambda) * max_similarity_to_selected
            sel_vecs = doc_vecs[selected_idx]   # shape: (n_selected, dim)
            mmr_scores = []
            for i in remaining:
                # Similarity to already-selected docs (max over selected)
                sim_to_sel = float(np.max(doc_vecs[i] @ sel_vecs.T))
                mmr_score  = lambda_param * rel_scores[i] - (1 - lambda_param) * sim_to_sel
                mmr_scores.append((i, mmr_score))
            best = max(mmr_scores, key=lambda x: x[1])[0]

        selected_idx.append(best)
        remaining.remove(best)

    selected_docs   = [docs[i]   for i in selected_idx]
    selected_scores = [scores[i] for i in selected_idx]

    print(f"  MMR selected {len(selected_docs)} diverse chunks")
    return selected_docs, selected_scores


# =============================================================================
#  FULL RETRIEVAL PIPELINE  (multi-query bi-encoder → cross-encoder → MMR)
# =============================================================================

def retrieve_and_rerank(
    question: str,
    candidate_k: int = 20,
    top_k: int = 5,
    threshold: float = 0.45,
    use_hyde: bool = True,
    use_query_expansion: bool = True,
    verbose: bool = True,
) -> Tuple[List[str], List[float]]:
    """
    Enhanced two-stage retrieval:
      Stage 0 — Query expansion + optional HyDE
      Stage 1 — Multi-query bi-encoder fetches candidates from ChromaDB
      Stage 2 — Cross-encoder reranks (or keyword-boost fallback)
      Stage 3 — MMR diversification

    Args:
        question            : user question
        candidate_k         : how many candidates to pull per query variant
        top_k               : final chunks to return
        threshold           : minimum score cutoff
        use_hyde            : use Hypothetical Document Embedding
        use_query_expansion : use query expansion variants
        verbose             : print debug tables
    """
    # ---- Stage 0: Build query pool ----
    queries = [question]

    if use_query_expansion:
        print(f"\n🔄 Stage 0a — Query expansion...")
        queries = expand_query(question)

    if use_hyde:
        print(f"🔮 Stage 0b — HyDE (hypothetical document embedding)...")
        hyp_answer = generate_hypothetical_answer(question)
        queries.append(hyp_answer)

    print(f"\n📡 Stage 1 — Multi-query bi-encoder retrieval (k={candidate_k} per query)...")
    docs, bi_scores, _ = biencoder_retrieve(queries, k=candidate_k, verbose=verbose)

    if not docs:
        return [], []

    if bi_scores and bi_scores[0] < 0.35:
        print(f"⚠  Very low bi-encoder score ({bi_scores[0]:.3f}) — may be out of domain")
        return [], []

    # ---- Stage 2: Reranking ----
    if cross_encoder_available:
        print(f"🎯 Stage 2 — Cross-encoder reranking (top_k={top_k * 2})...")
        # Get 2x candidates before MMR so MMR has more to choose from
        reranked_docs, reranked_scores = crossencoder_rerank(
            question, docs, bi_scores, top_k=top_k * 2, verbose=verbose
        )
        filtered_docs   = [d for d, s in zip(reranked_docs, reranked_scores) if s > -5.0]
        filtered_scores = [s for s in reranked_scores if s > -5.0]
    else:
        print(f"🔄 Stage 2 — Keyword-boosted reranking (fallback)...")
        filtered_docs, filtered_scores = biencoder_rerank(
            question, docs, bi_scores,
            top_k=top_k * 2, threshold=threshold, verbose=verbose
        )

    # ---- Stage 3: MMR diversification ----
    print(f"🎲 Stage 3 — MMR diversity selection (top_k={top_k})...")
    final_docs, final_scores = mmr_select(
        question, filtered_docs, filtered_scores,
        top_k=top_k, lambda_param=0.65,
    )

    print(f"✅ Final: {len(final_docs)} chunks selected")
    return final_docs, final_scores


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
#  QUESTION TYPE DETECTION (unchanged)
# =============================================================================

def detect_question_type(question: str) -> str:
    q = question.lower()

    if any(p in q for p in [
        "in points", "as points", "point by point", "in bullet", "as bullets",
        "bullet points", "numbered list", "as a list", "list format",
        "step by step", "in steps", "one by one", "enumerate",
    ]):
        return "points"

    if any(q.startswith(p) for p in ["how do", "how to", "how does", "procedure", "steps", "process"]):
        return "procedure"

    if any(q.startswith(p) for p in ["what is", "what are", "define", "meaning of"]):
        return "definition"

    if any(q.startswith(p) for p in ["explain", "describe", "tell me about", "overview", "elaborate"]):
        return "explanation"

    if any(w in q for w in ["list", "mention all", "what are all", "give all", "types of", "kinds of"]):
        return "list"

    if any(w in q for w in ["safety", "measures", "precautions", "guidelines", "requirements", "rules"]):
        return "list"

    if any(w in q for w in ["why", "reason", "cause", "purpose", "benefit", "advantage", "disadvantage"]):
        return "explanation"

    if any(w in q for w in ["compare", "difference", "vs", "versus", "distinguish", "contrast"]):
        return "comparison"

    return "general"


# =============================================================================
#  PROMPT BUILDER (improved: context budget management + citation anchors)
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
            "List the steps as a numbered list:\n"
            "1. First step\n2. Second step\n...\n"
            "Use only steps described in the context."
        ),
        "points": (
            "Answer using a numbered list. Each point must be a complete, "
            "informative sentence drawn directly from the context.\n"
            "Format:\n1. [Point one]\n2. [Point two]\n3. [Point three]\n...\n"
            "Do NOT write paragraphs — use ONLY the list format."
        ),
        "list": (
            "List all relevant items mentioned in the context. "
            "Use a numbered or bulleted format with a brief explanation for each."
        ),
        "comparison": (
            "Compare the items by highlighting their key differences and similarities "
            "as described in the context. Use structured points if helpful."
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
            "Answer only what is explicitly supported by the context."
        )

    # ⭐ IMPROVEMENT 4: Stronger grounding instruction
    # The original system prompt didn't explicitly forbid hallucination.
    # This version adds "Do NOT invent or infer facts not stated verbatim."
    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise technical assistant. Answer questions using "
                "ONLY the information in the provided context.\n\n"
                "RULES:\n"
                "1. Use ONLY facts explicitly stated in the context.\n"
                "2. Do NOT invent, infer, or add any information not present in the context.\n"
                "3. Do NOT use your general knowledge — treat the context as the only truth.\n"
                "4. If the context does not contain enough information, say exactly: "
                "'The available context does not contain sufficient information to answer this.'\n"
                "5. Mirror the vocabulary and terminology used in the context precisely.\n"
                "6. Follow the response format instruction exactly — do not deviate.\n\n"
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
#  ANSWER CLEANING & VALIDATION (unchanged)
# =============================================================================

def clean_generated_answer(answer: str) -> str:
    for marker in ["<|im_start|>assistant", "<|im_start|>", "<|im_end|>"]:
        if marker in answer:
            parts  = answer.split(marker)
            answer = parts[-1] if marker == "<|im_start|>assistant" else parts[0]

    answer = re.sub(r'^(assistant|system|user)\s*[:\n]', '', answer, flags=re.IGNORECASE).strip()

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
    if not text:
        return text
    text = text.strip()
    if text and text[-1] not in ".!?":
        last_line = text.splitlines()[-1].strip() if text.splitlines() else ""
        if re.match(r'^\d+\.', last_line) or last_line.startswith(('-', '•', '*')):
            return text
        if len(text.split()) > 5:
            return text + "."
    return text


def validate_context_relevance(context: str, question: str) -> float:
    question_terms = set(question.lower().split())
    context_lower  = context.lower()
    meaningful     = [t for t in question_terms if len(t) > 3]
    if not meaningful:
        return 1.0
    matches = sum(1 for t in meaningful if t in context_lower)
    return matches / len(meaningful)


# =============================================================================
#  ⭐ IMPROVEMENT 5: CONTEXT BUDGET MANAGEMENT
#  Original code assembled ALL retrieved chunks into one big context string.
#  If this exceeds the model's 2048-token window, later chunks get truncated
#  and early chunks dominate. We now:
#   1. Tokenize each chunk individually
#   2. Add chunks greedily until budget exhausted
#   3. Reserve tokens for prompt + answer headroom
# =============================================================================

MAX_INPUT_TOKENS = 3072   # ⭐ raised from 2048 — Qwen2-7B can handle this fine
ANSWER_HEADROOM  = 512    # reserved for generated answer
PROMPT_OVERHEAD  = 350    # approximate tokens for system prompt + chat template


def build_context_within_budget(
    docs: List[str],
    budget_tokens: int = MAX_INPUT_TOKENS - ANSWER_HEADROOM - PROMPT_OVERHEAD,
) -> str:
    """
    Pack as many retrieved chunks as fit within the token budget.
    Chunks are in relevance order (best first).
    """
    selected = []
    used     = 0

    for doc in docs:
        doc_tokens = len(tokenizer.encode(doc, add_special_tokens=False))
        separator  = 3  # tokens for "\n\n---\n\n"
        if used + doc_tokens + separator <= budget_tokens:
            selected.append(doc)
            used += doc_tokens + separator
        else:
            # Try to fit a truncated version of this chunk (at least 100 tokens)
            remaining = budget_tokens - used - separator
            if remaining >= 100:
                truncated_tokens = tokenizer.encode(doc, add_special_tokens=False)[:remaining]
                truncated_text   = tokenizer.decode(truncated_tokens)
                selected.append(truncated_text)
            break

    print(f"  Context budget: {used}/{budget_tokens} tokens, {len(selected)} chunks included")
    return "\n\n---\n\n".join(selected)


# =============================================================================
#  ANSWER GENERATION (improved generation parameters)
# =============================================================================

def generate_answer(
    context: str,
    question: str,
    similarity_score: Optional[float] = None,
) -> str:
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
        max_length=MAX_INPUT_TOKENS,   # ⭐ raised context window
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            # ⭐ IMPROVEMENT 6: Lower temperature for factual RAG
            # Temperature=0.4 introduces unnecessary randomness for factual tasks.
            # 0.1-0.2 keeps answers grounded; use beam search for best accuracy.
            do_sample=False,                # ⭐ greedy/beam for factual tasks
            num_beams=3,                    # ⭐ beam search finds better completions
            early_stopping=True,
            repetition_penalty=1.15,        # ⭐ slightly higher to reduce repetition
            no_repeat_ngram_size=4,         # ⭐ prevents exact phrase repetition
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    answer = clean_generated_answer(answer)

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
    use_hyde: bool = True,
    use_query_expansion: bool = True,
) -> str:
    """
    Full enhanced RAG pipeline:
      0. Query expansion + HyDE
      1. Multi-query bi-encoder retrieval
      2. Cross-encoder reranking (or keyword-boost fallback)
      3. MMR diversity selection
      4. Context budget management
      5. Beam-search generation with tight grounding prompt
    """
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
        use_hyde=use_hyde,
        use_query_expansion=use_query_expansion,
        verbose=True,
    )

    # Retry with looser pool if nothing found
    if not docs:
        print(f"⚠  No results — retrying with larger candidate pool (k={candidate_k + 10})...")
        docs, scores = retrieve_and_rerank(
            question,
            candidate_k=candidate_k + 10,
            top_k=top_k + 3,
            threshold=max(threshold - 0.10, 0.25),
            use_hyde=False,              # skip HyDE on retry (already tried)
            use_query_expansion=False,
            verbose=False,
        )

    if not docs:
        return "I don't have sufficient information in the knowledge base to answer this question."

    # ⭐ Improved deduplication: use full text hash not just first 100 chars
    seen        = set()
    unique_docs = []
    unique_scores = []
    for doc, score in zip(docs, scores):
        key = hash(doc)
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
            unique_scores.append(score)

    # ⭐ Build context with budget management
    context = build_context_within_budget(unique_docs)

    if len(context.split()) < 20:
        return "The retrieved information is too limited to provide a reliable answer."

    avg_score = sum(unique_scores) / len(unique_scores) if unique_scores else 0
    answer    = generate_answer(context, question, avg_score)

    return ensure_complete_sentence(answer)


# =============================================================================
#  STARTUP DIAGNOSTIC (unchanged)
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
        print(f"✓ Embedding dimension : {embed_dim}")

        norms    = [np.linalg.norm(e) for e in sample["embeddings"]]
        avg_norm = np.mean(norms)
        print(f"✓ Vector norms        : mean={avg_norm:.4f}, std={np.std(norms):.4f}")

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
                ("What is the system configuration?", "This is completely unrelated text about apples."),
            ]
            ce_scores = reranker.predict(test_pairs, show_progress_bar=False)
            print(f"  Relevant doc score   : {ce_scores[0]:.4f}")
            print(f"  Irrelevant doc score : {ce_scores[1]:.4f}")
            if ce_scores[0] > ce_scores[1]:
                print("✓ Cross-encoder correctly ranks relevant doc higher")
            else:
                print("⚠ Cross-encoder ranking unexpected — check model")

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
    print(f"   Model      : {model_type} Qwen2-7B")
    print(f"   Device     : {device.upper()}")
    print(f"   Chunks     : {collection.count()}")
    print(f"   Reranker   : {rerank_type}")
    print(f"   HyDE       : enabled")
    print(f"   Query Exp. : enabled")
    print(f"   MMR        : enabled")
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
            print("  - Use keywords from your training material")
            print("  - Add --no-hyde or --no-expand flags to disable expansions\n")
            continue

        # Optional flags to disable expensive stages for quick queries
        use_hyde_flag   = "--no-hyde"   not in q
        use_expand_flag = "--no-expand" not in q
        clean_q = q.replace("--no-hyde", "").replace("--no-expand", "").strip()

        answer = ask(
            clean_q,
            candidate_k=20,
            top_k=5,
            threshold=0.45,
            use_hyde=use_hyde_flag,
            use_query_expansion=use_expand_flag,
        )

        print(f"\n{'─'*60}")
        print(f"Answer:\n{answer}")
        print(f"{'─'*60}\n")
