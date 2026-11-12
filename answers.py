import json
from pathlib import Path

from evals import load_gold
from fineprint import load_index, run_agent, usage_line

ANSWERS = Path(__file__).parent / "answers.jsonl"

if __name__ == "__main__":
    corpus, vectors = load_index()
    total = 0.0
    with ANSWERS.open("w") as f:
        for row in load_gold():
            run = run_agent(row["question"], corpus, vectors)
            f.write(json.dumps({"id": row["id"], **run}) + "\n")
            total += run["cost_usd"]
            print(f"  {row['id']} done: {usage_line(run)}")
    print(f"total: ${total:.4f}")
