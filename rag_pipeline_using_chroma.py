import json
import re
import torch
import os
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------- PATHS ----------------
LLM_PATH = r"D:/Machine Learning and LLMs/LLMs/Mistral-7B-Instruct-v0.2"
EMBED_PATH = r"D:/Machine Learning and LLMs/LLMs/all-MiniLM-L6-v2"
CHROMA_DIR = r"D:/Interns/pdf_extraction_using_chromadb/chroma_store"
COLLECTION_NAME = "pdf_markdown_embeddings"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD EMBEDDER ----------------
print("Loading embedder...")
embedder = SentenceTransformer(EMBED_PATH, device=device)

# ---------------- LOAD LLM ----------------
print("Loading LLM...")
tokenizer = AutoTokenizer.from_pretrained(LLM_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    LLM_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    do_sample=False
)

# ---------------- LOAD CHROMA ----------------
print("Loading ChromaDB...")
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION_NAME)

print(f"Loaded {collection.count()} chunks from ChromaDB")

# ---------------- RETRIEVAL (CHROMA) ----------------
def retrieve(query, k=25):
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    return results["documents"][0]

# ---------------- QUESTION TYPE DETECTION ----------------
def detect_question_type(question: str) -> str:
    q = question.lower()

    if q.startswith(("what is", "define", "meaning of")):
        return "definition"
    if q.startswith(("explain", "describe", "overview")):
        return "explanation"
    if q.startswith(("how", "procedure", "steps", "process")):
        return "procedure"
    if any(word in q for word in ["safety", "measures", "precautions", "guidelines"]):
        return "list"
    return "general"

# ---------------- CONTEXT CLEANING ----------------
def clean_context(context: str) -> str:
    lines = []
    for line in context.splitlines():
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith(("figure", "table")) and len(line.split()) < 6:
            continue
        lines.append(line)
    return "\n".join(lines)

# ---------------- SENTENCE COMPLETION ----------------
def ensure_complete_sentence(text: str) -> str:
    text = text.strip()
    if text.endswith((".", "!", "?")):
        return text
    matches = list(re.finditer(r"[.!?]", text))
    if matches:
        return text[:matches[-1].end()].strip()
    return text

# ---------------- PROMPT BUILDER ----------------
def build_prompt(context: str, question: str, qtype: str) -> str:
    style_map = {
        "definition": "Give a short, precise definition in one paragraph.",
        "explanation": "Provide a clear and complete explanation in 5–7 sentences.",
        "procedure": "Explain step by step in a clear sequence.",
        "list": "Answer using clear bullet points.",
        "general": "Provide a concise, factual explanation."
    }

    return f"""[INST]
You are a technical assistant.

Answer using ONLY the context provided.

Rules:
- {style_map[qtype]}
- Do NOT mention chapters, figures, or tables.
- Do NOT invent information.
- If the context does not contain the answer, say exactly:
  "I don't have information on that."

### CONTEXT:
{context}

### QUESTION:
{question}

### ANSWER:
[/INST]"""

# ---------------- ANSWER GENERATION ----------------
def generate_answer(context: str, question: str) -> str:
    qtype = detect_question_type(question)
    context = clean_context(context)
    prompt = build_prompt(context, question, qtype)

    response = llm_pipeline(
        prompt,
        max_new_tokens=400,
        eos_token_id=tokenizer.eos_token_id
    )

    answer = response[0]["generated_text"]
    answer = answer.replace("[INST]", "").replace("[/INST]", "").strip()

    if "### ANSWER:" in answer:
        answer = answer.split("### ANSWER:")[-1].strip()

    answer = ensure_complete_sentence(answer)

    if not answer or len(answer.split()) < 5:
        return "I don't have information on that."

    return answer

# ---------------- RAG PIPELINE ----------------
def ask(question: str) -> str:
    docs = retrieve(question)
    if not docs:
        return "I don't have information on that."
    context = "\n\n".join(docs)
    return generate_answer(context, question)

# ---------------- CHAT ----------------
print("\n✅ RAG READY — ChromaDB Backend, Offline, GPU-Accelerated\n")

while True:
    q = input(">>> ")
    if q.lower() in ["exit", "quit"]:
        break
    print("\n", ask(q), "\n")
