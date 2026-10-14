import json
from pathlib import Path

from evals import load_gold
from fineprint import load_corpus, run_agent

ANSWERS = Path(__file__).parent / "answers.jsonl"

if __name__ == "__main__":
    corpus = load_corpus()
    with ANSWERS.open("w") as f:
        for row in load_gold():
            answer = run_agent(row["question"], corpus)
            f.write(json.dumps({"id": row["id"], "answer": answer}) + "\n")
            print(f"  {row['id']} done")
