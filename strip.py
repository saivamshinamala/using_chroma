
import json


data = []
jsonl_file=r"D:\RAG Projects\vamshi\RAFT\data\QAPairs\qa_pairs.jsonl"  # Path to your JSONL file
with open(jsonl_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()  # remove leading/trailing whitespace
        if not line:         # skip empty lines
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"⚠️ Skipping invalid line: {line}\nError: {e}")
            continue
        prompt = f"Question: {item['question']}\nAnswer:"
        completion = f" {item['answer']}"
        data.append({"input_text": prompt, "target_text": completion})
