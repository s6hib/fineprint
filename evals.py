import json
from pathlib import Path

GOLD = Path(__file__).parent / "gold.jsonl"


def load_gold() -> list[dict]:
    """The gold questions, one per line of gold.jsonl."""
    return [json.loads(line) for line in GOLD.read_text().splitlines()]


def hit(row: dict, chunks: list[dict]) -> bool:
    """True if any chunk comes from the file and page the gold answer lives on."""
    return any(c["file"] in row["files"] and c["page"] in row["pages"] for c in chunks)


if __name__ == "__main__":
    from fineprint import (
        CANDIDATES,
        build_index,
        keyword_search,
        load_index,
        rerank,
        rrf,
        vector_search,
    )

    gold = [
        r for r in load_gold() if r["files"]
    ]  # out-of-corpus rows have no page to find
    corpus, vectors = load_index()
    db = build_index(corpus)

    fused_hits = rerank_hits = 0
    for row in gold:
        q = row["question"]
        fused = rrf(
            [keyword_search(q, db, CANDIDATES), vector_search(q, vectors, CANDIDATES)],
            CANDIDATES,
        )
        top3 = rerank(q, fused, corpus, 3)
        fused_hit = hit(row, [corpus[i] for i in fused])
        rerank_hit = hit(row, [corpus[i] for i in top3])
        fused_hits += fused_hit
        rerank_hits += rerank_hit
        if not rerank_hit:
            print(
                f"  miss {row['id']} ({row['kind']}): fused {'yes' if fused_hit else 'no'}"
            )

    print(f"recall@10 after rrf:   {fused_hits}/{len(gold)}")
    print(f"recall@3 after rerank: {rerank_hits}/{len(gold)}")
