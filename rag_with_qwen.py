# ================= OFFLINE SETTINGS (MUST BE FIRST) =================
import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HOME"] = "./hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "./hf_cache"

# ================= IMPORTS =================
import re
import sys
import numpy as np
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from pathlib import Path
from typing import List, Tuple, Optional

# ================= PATHS =================
BASE_DIR = Path(__file__).resolve().parent

BASE_MODEL_PATH = r"E:\llms\Qwen2-1.5B-Instruct"
ADAPTER_PATH    = str(BASE_DIR / "models" / "qwen_qlora_adapters")
EMBED_PATH      = r"D:\Machine Learning and LLMs\LLMs\all-MiniLM-L6-v2"
CHROMA_DIR      = str(BASE_DIR / "chroma_store")
COLLECTION_NAME = "pdf_markdown_embeddings"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ================= LOAD EMBEDDER =================
print("Loading embedder (OFFLINE)...")
embedder = SentenceTransformer(EMBED_PATH, device="cuda", local_files_only=True)
print("✅ Embedder loaded")

# ================= LOAD CHROMA =================
print("Loading ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION_NAME)
print(f"✅ Collection loaded — {collection.count()} chunks")

# ── Validate distance metric at startup (fail fast)
space = (collection.metadata or {}).get("hnsw:space", "l2")
print(f"   Distance metric: {space}")
if space != "cosine":
    print(
        f"\n⛔  FATAL: Collection metric is '{space}', expected 'cosine'.\n"
        "    Re-run embeddings.py to recreate the collection with the correct metric.\n"
        "    Do NOT continue — retrieval results will be meaningless.\n"
    )
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

# ================= QUESTION TYPE DETECTION =================
def detect_question_type(question: str) -> str:
    """
    Enhanced question type detection including point/list format requests.
    """
    q = question.lower()

    # Explicit formatting requests — check FIRST before semantic type
    if any(p in q for p in [
        "in points", "as points", "point by point", "in bullet", "as bullets",
        "bullet points", "numbered list", "as a list", "list format",
        "step by step", "in steps", "one by one", "enumerate"
    ]):
        return "points"

    # Procedure / how-to
    if any(q.startswith(p) for p in ["how do", "how to", "how does", "procedure", "steps", "process"]):
        return "procedure"

    # Definition
    if any(q.startswith(p) for p in ["what is", "what are", "define", "meaning of"]):
        return "definition"

    # Explanation
    if any(q.startswith(p) for p in ["explain", "describe", "tell me about", "overview", "elaborate"]):
        return "explanation"

    # List / enumeration
    if any(w in q for w in ["list", "mention all", "what are all", "give all", "types of", "kinds of"]):
        return "list"

    # Safety / guidelines
    if any(w in q for w in ["safety", "measures", "precautions", "guidelines", "requirements", "rules"]):
        return "list"

    # Reason / cause
    if any(w in q for w in ["why", "reason", "cause", "purpose", "benefit", "advantage", "disadvantage"]):
        return "explanation"

    # Comparison
    if any(w in q for w in ["compare", "difference", "vs", "versus", "distinguish", "contrast"]):
        return "comparison"

    return "general"


# ================= ENHANCED RETRIEVAL =================
def retrieve_with_reranking(
    query: str,
    k: int = 10,
    top_k: int = 5,
    threshold: float = 0.45,
    verbose: bool = True
) -> Tuple[List[str], List[float]]:
    """
    Enhanced retrieval with semantic reranking and keyword boost.
    """
    q_vec = embedder.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    if verbose:
        print(f"Query vector norm: {np.linalg.norm(q_vec[0]):.4f}")

    results = collection.query(
        query_embeddings=q_vec,
        n_results=k,
        include=["documents", "distances", "metadatas"],
    )

    docs      = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results.get("metadatas", [[{}] * len(docs)])[0]

    candidates = []
    query_terms = set(query.lower().split())

    for doc, dist, meta in zip(docs, distances, metadatas):
        similarity = 1.0 - (dist / 2.0)

        doc_lower = doc.lower()
        keyword_matches = sum(1 for term in query_terms if len(term) > 3 and term in doc_lower)
        keyword_boost = min(0.08 * keyword_matches, 0.2)

        boosted_similarity = min(similarity + keyword_boost, 1.0)

        candidates.append({
            'doc': doc,
            'similarity': similarity,
            'boosted_similarity': boosted_similarity,
            'metadata': meta
        })

    candidates.sort(key=lambda x: x['boosted_similarity'], reverse=True)

    if verbose:
        print("\n📊 Retrieval Results:")
        print(f"{'Rank':<5} {'Dist':<10} {'Sim':<8} {'Boosted':<8} {'Content':<50}")
        print("-" * 80)
        for i, cand in enumerate(candidates[:top_k], 1):
            orig_sim  = cand['similarity']
            boost_sim = cand['boosted_similarity']
            content   = cand['doc'][:45] + "..." if len(cand['doc']) > 45 else cand['doc']
            print(f"{i:<5} {(2*(1-orig_sim)):<10.4f} {orig_sim:<8.4f} {boost_sim:<8.4f} {content:<50}")

    # Require top result to be meaningful
    if candidates and candidates[0]['boosted_similarity'] < 0.45:
        print(f"\n⚠ Weak retrieval (top similarity={candidates[0]['boosted_similarity']:.3f})")
        return [], []

    filtered = []
    scores   = []
    for cand in candidates[:top_k]:
        if cand['boosted_similarity'] >= threshold:
            filtered.append(cand['doc'])
            scores.append(cand['boosted_similarity'])

    print(f"\n🔎 Retrieved {len(filtered)} / {top_k} chunks (threshold={threshold})")
    return filtered, scores


# ================= CONTEXT CLEANING =================
def clean_context(context: str, preserve_structure: bool = True) -> str:
    """
    Clean context while preserving structural information.
    """
    cleaned    = []
    lines      = context.splitlines()

    for line in lines:
        line = line.strip()

        if not line:
            if preserve_structure and cleaned and cleaned[-1] != "":
                cleaned.append("")
            continue

        lower = line.lower()
        if any(phrase in lower for phrase in [
            "page intentionally left blank",
            "this page intentionally",
            "continued on next page"
        ]):
            continue

        # Always keep headers, bullets, numbered items
        if re.match(r'^(\d+\.|\*|-|•|[A-Z][A-Z\s]+:)', line):
            cleaned.append(line)
            continue

        # Keep lines with technical markers
        if any(char in line for char in ['=', ':', '(', ')', '%', '@']):
            cleaned.append(line)
            continue

        # Drop very short non-numeric lines
        if len(line) < 20 and not any(c.isdigit() for c in line):
            continue

        cleaned.append(line)

    # Collapse multiple blank lines
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


# ================= PROMPT BUILDER =================
def build_prompt(context: str, question: str, qtype: str, similarity_score: float = None) -> str:
    """
    Build a prompt tailored to question type, especially for point/list formats.
    """

    # ── Format instructions per question type ──────────────────────────────
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
            "1. First step\n"
            "2. Second step\n"
            "...\n"
            "Use only steps described in the context."
        ),
        "points": (
            "Answer using a numbered or bulleted list. Each point must be a complete, "
            "informative sentence drawn directly from the context. "
            "Format:\n1. [Point one]\n2. [Point two]\n3. [Point three]\n..."
            "\nDo NOT write paragraphs — use ONLY the list format."
        ),
        "list": (
            "List all relevant items mentioned in the context. "
            "Use a numbered or bulleted format with a brief explanation for each item."
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

    # ── Confidence hint ────────────────────────────────────────────────────
    confidence_note = ""
    if similarity_score is not None:
        if similarity_score < 0.55:
            confidence_note = (
                "\nNote: Context relevance is moderate. Answer only what is "
                "explicitly supported by the context."
            )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise technical assistant. Your job is to answer questions "
                "using ONLY the information in the provided context.\n\n"
                "RULES:\n"
                "1. Use ONLY facts explicitly stated in the context.\n"
                "2. Do NOT add outside knowledge, assumptions, or general knowledge.\n"
                "3. If the context does not contain enough information, say: "
                "   'The available context does not contain sufficient information to answer this.'\n"
                "4. Mirror the vocabulary and terminology used in the context.\n"
                "5. Follow the response format instruction exactly.\n\n"
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


# ================= ANSWER CLEANING =================
def clean_generated_answer(answer: str) -> str:
    """
    Minimal cleaning — only remove chat template artifacts.
    Preserve lists, bullets, and formatting from the model output.
    """
    # Remove chat template leakage
    for marker in ["<|im_start|>assistant", "<|im_start|>", "<|im_end|>"]:
        if marker in answer:
            parts = answer.split(marker)
            answer = parts[-1] if marker == "<|im_start|>assistant" else parts[0]

    # Remove leading role labels if present
    answer = re.sub(r'^(assistant|system|user)\s*[:\n]', '', answer, flags=re.IGNORECASE).strip()

    # Remove repeated question echo at the start
    lines = answer.splitlines()
    cleaned_lines = []
    skip_echo = True
    for line in lines:
        stripped = line.strip()
        if skip_echo and (not stripped or stripped.endswith("?")):
            continue
        skip_echo = False
        cleaned_lines.append(line)

    answer = "\n".join(cleaned_lines).strip()

    # Collapse excessive blank lines (keep max 1)
    answer = re.sub(r'\n{3,}', '\n\n', answer)

    return answer.strip()


# ================= ANSWER VALIDATION =================
def validate_answer(answer: str, context: str, question: str) -> bool:
    """
    Relaxed validation — checks answer is non-trivial and not a hallucination refusal.
    Does NOT enforce strict word-overlap (this was causing list/point answers to fail).
    """
    if not answer or len(answer.split()) < 5:
        return False

    # Reject if model refused to use context and fell back to general AI behaviour
    refusal_phrases = [
        "as an ai language model",
        "i don't have access to",
        "based on my training data",
        "i cannot provide information",
        "my knowledge cutoff",
    ]
    answer_lower = answer.lower()
    if any(phrase in answer_lower for phrase in refusal_phrases):
        return False

    # Reject completely empty / one-word answers
    if len(answer.strip()) < 20:
        return False

    return True


# ================= CONTEXT RELEVANCE CHECK =================
def validate_context_relevance(context: str, question: str) -> float:
    """
    Loose check: returns fraction of meaningful question terms found in context.
    """
    question_terms = set(question.lower().split())
    context_lower  = context.lower()
    matches = sum(1 for term in question_terms if len(term) > 3 and term in context_lower)
    return matches / max(len([t for t in question_terms if len(t) > 3]), 1)


# ================= GENERATION =================
def generate_answer(context: str, question: str, similarity_score: float = None) -> str:
    """
    Generate answer with format-aware prompting and relaxed validation.
    """
    qtype   = detect_question_type(question)
    context = clean_context(context, preserve_structure=True)

    if not context.strip() or len(context.split()) < 15:
        return "Insufficient context available to answer this question."

    # Only hard-reject if context has zero semantic relation to question
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

    # ── Generation parameters ────────────────────────────────────────────
    # temperature=0.4 gives slightly more flexibility for reformatting
    # top_p=0.9, top_k=50 gives richer vocabulary for list items
    # max_new_tokens=512 allows full point-format answers
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.4,
            top_p=0.90,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    answer = clean_generated_answer(answer)

    if not validate_answer(answer, context, question):
        return "Unable to provide a reliable answer based on the available context."

    return answer


# ================= COMPLETE SENTENCE HELPER =================
def ensure_complete_sentence(text: str) -> str:
    """
    Only add a trailing period if text doesn't end with punctuation.
    Does NOT truncate — preserves lists and bullet points fully.
    """
    if not text:
        return text
    text = text.strip()
    if text and text[-1] not in ".!?":
        # Check if last line is a list item — don't truncate
        last_line = text.splitlines()[-1].strip() if text.splitlines() else ""
        if re.match(r'^\d+\.', last_line) or last_line.startswith(('-', '•', '*')):
            return text  # Leave list as-is
        if len(text.split()) > 5:
            return text + "."
    return text


# ================= MAIN RAG PIPELINE =================
def ask(question: str, threshold: float = 0.45, use_reranking: bool = True) -> str:
    """
    Full RAG pipeline with format-aware answer generation.
    """
    print(f"\n{'='*60}")
    print(f"QUERY: {question}")
    qtype = detect_question_type(question)
    print(f"TYPE:  {qtype}")
    print('='*60)

    if use_reranking:
        docs, scores = retrieve_with_reranking(
            question,
            k=12,
            top_k=6,
            threshold=threshold
        )
    else:
        docs   = retrieve(question, k=8, threshold=threshold)
        scores = [threshold] * len(docs)

    if not docs:
        fallback_threshold = threshold - 0.15
        print(f"⚠  No results at threshold={threshold}. Trying {fallback_threshold:.2f}...")
        if use_reranking:
            docs, scores = retrieve_with_reranking(
                question, k=15, top_k=8, threshold=fallback_threshold, verbose=False
            )
        else:
            docs   = retrieve(question, k=10, threshold=fallback_threshold, verbose=False)
            scores = [fallback_threshold] * len(docs)

        if not docs:
            return "I don't have sufficient information in the knowledge base to answer this question."

    # Deduplicate chunks
    seen        = set()
    unique_docs = []
    for doc in docs:
        key = doc[:100] if len(doc) > 100 else doc
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)

    context = "\n\n---\n\n".join(unique_docs)

    if len(context.split()) < 20:
        return "The retrieved information is too limited to provide a reliable answer."

    avg_score = sum(scores) / len(scores) if scores else 0
    answer    = generate_answer(context, question, avg_score)

    return ensure_complete_sentence(answer)


# ================= ORIGINAL RETRIEVE (BACKWARD COMPAT) =================
def retrieve(
    query: str,
    k: int = 5,
    threshold: float = 0.35,
    verbose: bool = True
) -> List[str]:
    """Original retrieve — kept for backward compatibility."""
    q_vec = embedder.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).tolist()

    if verbose:
        print(f"Query vector norm: {np.linalg.norm(q_vec[0]):.4f}")

    results = collection.query(
        query_embeddings=q_vec,
        n_results=k,
        include=["documents", "distances"],
    )

    docs      = results["documents"][0]
    distances = results["distances"][0]

    similarities    = []
    filtered_docs   = []

    for rank, (doc, dist) in enumerate(zip(docs, distances), 1):
        similarity = 1.0 - (dist / 2.0)
        similarities.append(similarity)
        if verbose:
            print(f"{rank:<5} {dist:>8.4f} {similarity:>11.4f}  {doc[:55]}...")

    top_similarity = similarities[0] if similarities else 0
    if top_similarity < 0.50:
        print(f"⚠ Weak retrieval (top similarity={top_similarity:.3f})")
        return []

    for doc, sim in zip(docs, similarities):
        if sim >= threshold:
            filtered_docs.append(doc)

    print(f"\n🔎 Retrieved {len(filtered_docs)} / {len(docs)} chunks (threshold={threshold})")
    return filtered_docs


# ================= STARTUP DIAGNOSTIC =================
def run_startup_diagnostic():
    """Check embedding quality and retrieval health at startup."""
    print("\n" + "="*60)
    print(" STARTUP DIAGNOSTIC")
    print("="*60)

    try:
        sample = collection.get(limit=10, include=["embeddings", "documents"])

        if not sample["embeddings"]:
            print("❌ ERROR: No embeddings found in collection!")
            return

        embed_dim = len(sample["embeddings"][0])
        print(f"✓ Embedding dimension: {embed_dim}")

        norms    = [np.linalg.norm(e) for e in sample["embeddings"]]
        avg_norm = np.mean(norms)
        std_norm = np.std(norms)
        print(f"✓ Vector norms: mean={avg_norm:.4f}, std={std_norm:.4f}")

        if not all(0.95 < n < 1.05 for n in norms[:5]):
            print("⚠ WARNING: Vectors may not be properly normalized!")
        else:
            print("✓ Vectors are properly normalized")

        # Self-retrieval test
        print("\n📊 Self-Retrieval Test:")
        test_doc     = sample["documents"][0]
        test_snippet = test_doc[:200] if len(test_doc) > 200 else test_doc

        q_vec = embedder.encode(
            [test_snippet], normalize_embeddings=True, convert_to_numpy=True
        ).tolist()

        res = collection.query(
            query_embeddings=q_vec,
            n_results=3,
            include=["documents", "distances"]
        )

        if res["documents"][0]:
            first_match = res["documents"][0][0]
            dist        = res["distances"][0][0]
            sim         = 1.0 - (dist / 2.0)
            print(f"  Distance: {dist:.4f}  |  Similarity: {sim:.4f}")
            if first_match[:100] == test_doc[:100]:
                print("✓ Exact match — retrieval working correctly")
            elif sim > 0.9:
                print("✓ High similarity match — retrieval working")
            else:
                print("⚠ WARNING: Self-retrieval similarity is low — check embedding model")

        # Consistency test
        print("\n🔍 Embedding Consistency Test:")
        test_text   = "This is a test sentence for embedding consistency"
        emb1        = embedder.encode([test_text], normalize_embeddings=True)
        emb2        = embedder.encode([test_text], normalize_embeddings=True)
        consistency = np.dot(emb1[0], emb2[0])
        print(f"  Consistency score: {consistency:.6f}")
        if consistency > 0.9999:
            print("✓ Embeddings are deterministic and consistent")
        else:
            print("⚠ WARNING: Embeddings may not be deterministic!")

        print("\n" + "="*60)
        print(" DIAGNOSTIC COMPLETE")
        print("="*60 + "\n")

    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")


# ================= MAIN =================
if __name__ == "__main__":
    run_startup_diagnostic()

    model_type = "Fine-tuned" if use_finetuned else "Base"
    print(f"\n✅ Enhanced RAG System Ready")
    print(f"   Model: {model_type} Qwen2-1.5B")
    print(f"   Device: {device.upper()}")
    print(f"   Chunks: {collection.count()}")
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
            print("  - Rephrased questions work — e.g. 'what does X do?' vs 'explain X'\n")
            continue

        answer = ask(q, threshold=0.45, use_reranking=True)

        print(f"\n{'─'*60}")
        print(f"Answer:\n{answer}")
        print(f"{'─'*60}\n")