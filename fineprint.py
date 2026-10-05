import re
from pathlib import Path
import anthropic
import pymupdf
import voyageai
from collections import Counter
import sqlite3
from pydantic import BaseModel

pymupdf.no_recommend_layout()  # stop the pymupdf_layout hint on every run

DATA_DIR = Path(__file__).parent / "data"


class PolicyMeta(BaseModel):
    """What page 1 of a policy says about itself."""

    payer: str
    title: str
    policy_number: str
    effective_date: str


def load_corpus(data_dir: Path = DATA_DIR) -> list[dict]:
    """One entry per chunk. File, page, heading and policy metadata travel with the text."""
    chunks = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        doc = pymupdf.open(pdf_path)
        first = doc[0]
        meta = extract_metadata(first.get_text(textpage=ocr_textpage(first)))
        doc_chunks = chunk_by_heading(pdf_path.name, doc) + table_chunks(
            pdf_path.name, doc
        )
        for chunk in doc_chunks:
            chunk.update(meta.model_dump())
        chunks.extend(doc_chunks)
    return chunks


HEADING_RATIO = 1.15  # a block this much bigger than body text is a heading
BOLD_FLAG = 16

MIN_CHARS_PER_PAGE = 50  # less than this and the page is a scan, not text


def ocr_textpage(page):
    """OCR a page that is an image with nothing under it. None means use the text layer."""
    if len(page.get_text().strip()) >= MIN_CHARS_PER_PAGE or not page.get_images():
        return None
    print(f"  ocr: {page.parent.name} p.{page.number + 1}")
    return page.get_textpage_ocr(language="eng", dpi=300, full=True)


def page_blocks(page) -> list[dict]:
    """One dict per text block: its text, its dominant font size, bold or not."""
    blocks = []
    for block in page.get_text("dict", textpage=ocr_textpage(page))["blocks"]:
        if block["type"] != 0:  # 0 = text, 1 = image
            continue
        spans = [s for line in block["lines"] for s in line["spans"]]
        text = " ".join(s["text"] for s in spans).strip()
        if not text:
            continue
        size = max(s["size"] for s in spans)
        bold = any(s["flags"] & BOLD_FLAG for s in spans)
        blocks.append({"text": text, "size": size, "bold": bold})
    return blocks


def chunk_by_heading(file: str, doc) -> list[dict]:
    """Cut a document into chunks: each starts at a heading, runs to the next."""
    pages = [(n, page_blocks(p)) for n, p in enumerate(doc, start=1)]
    sizes = Counter()
    for _, blocks in pages:
        for b in blocks:
            sizes[round(b["size"], 1)] += len(b["text"])
    body = sizes.most_common(1)[0][0]

    chunks = []
    current = {"file": file, "page": 1, "heading": "", "text": ""}
    for page_number, blocks in pages:
        if current["text"]:
            chunks.append(current)
        current = {**current, "page": page_number, "text": ""}

        for b in blocks:
            is_heading = (b["size"] > body * HEADING_RATIO or b["bold"]) and len(
                b["text"]
            ) < 120
            if is_heading:
                if current["text"]:
                    chunks.append(current)
                current = {
                    "file": file,
                    "page": page_number,
                    "heading": b["text"],
                    "text": "",
                }
            else:
                current["text"] += b["text"] + "\n"
    if current["text"]:
        chunks.append(current)
    return [piece for c in chunks for piece in split_long(c)]


def table_chunks(file: str, doc) -> list[dict]:
    """One chunk per table. Each row becomes one line: cell | cell | cell."""
    chunks = []
    for page_number, page in enumerate(doc, start=1):
        for table in page.find_tables().tables:
            rows = table.extract()  # list of rows, each a list of cell strings or None
            lines = [
                " | ".join(cell.strip() for cell in row if cell and cell.strip())
                for row in rows
                if any(cell for cell in row)
            ]
            if not lines:
                continue
            chunks.append(
                {
                    "file": file,
                    "page": page_number,
                    "heading": lines[0],
                    "text": "\n".join(lines),
                }
            )
    return chunks


MAX_CHUNK_CHARS = 1500  # ~375 tokens


def split_long(chunk: dict) -> list[dict]:
    """Split a chunk over the cap into pieces at paragraph breaks. Same heading on each."""
    if len(chunk["text"]) <= MAX_CHUNK_CHARS:
        return [chunk]
    pieces, buf = [], ""
    for para in chunk["text"].split("\n"):
        if buf and len(buf) + len(para) > MAX_CHUNK_CHARS:
            pieces.append({**chunk, "text": buf})
            buf = ""
        buf += para + "\n"
    if buf.strip():
        pieces.append({**chunk, "text": buf})
    return pieces


def chunk_text(c: dict) -> str:
    """What gets indexed, embedded and reranked: one line of document context, then the chunk."""
    return f"{c['payer']} | {c['title']} | {c['policy_number']} | {c['heading']}\n{c['text']}"


def build_index(chunks: list[dict]) -> sqlite3.Connection:
    """In-memory FTS5 table over the chunks. rowid = the chunk's position in the list."""
    db = sqlite3.connect(":memory:")
    db.execute("create virtual table chunks using fts5(text)")
    db.executemany(
        "insert into chunks(rowid, text) values (?, ?)",
        [(i, chunk_text(c)) for i, c in enumerate(chunks)],
    )
    return db


def keyword_search(query: str, db: sqlite3.Connection, top_k: int = 3) -> list[int]:
    """BM25 over the index. Returns chunk positions, best first."""
    terms = re.findall(r"\w+", query)
    match = " OR ".join(
        f'"{t}"' for t in terms
    )  # quoted: codes stay whole, AND/OR/NOT stay words
    rows = db.execute(
        "select rowid from chunks where chunks match ? order by rank limit ?",
        (match, top_k),
    )
    return [rowid for (rowid,) in rows]


CANDIDATES = 10  # how many hits each search hands to the fusion


def search_policies(
    query: str,
    corpus: list[dict],
    db: sqlite3.Connection,
    vectors: list[list[float]],
    top_k: int = 3,
) -> str:
    """The tool the model calls. Fuses keyword and vector hits, tags each with its [file p.N]."""
    keyword_hits = keyword_search(query, db, CANDIDATES)
    vector_hits = vector_search(query, vectors, CANDIDATES)
    fused = rrf([keyword_hits, vector_hits], CANDIDATES)
    hits = [corpus[i] for i in rerank(query, fused, corpus, top_k)]

    return (
        "\n\n---\n\n".join(
            f"[{c['file']} p.{c['page']}]\n{c['heading']}\n{c['text'][:4000]}"
            for c in hits
        )
        or "No matching chunks."
    )


EMBED_MODEL = "voyage-4"


def embed(texts: list[str], input_type: str) -> list[list[float]]:
    """One vector per text. input_type is "document" for chunks, "query" for the question."""
    client = voyageai.Client()
    return client.embed(texts, model=EMBED_MODEL, input_type=input_type).embeddings


def vector_search(query: str, vectors: list[list[float]], top_k: int = 3) -> list[int]:
    """Dot product of the query against every chunk vector. Returns chunk positions, best first."""
    q = embed([query], "query")[0]
    scores = [sum(a * b for a, b in zip(q, v)) for v in vectors]
    return sorted(range(len(vectors)), key=lambda i: scores[i], reverse=True)[:top_k]


RRF_K = 60


RERANK_MODEL = "rerank-3"


def rerank(
    query: str, candidates: list[int], corpus: list[dict], top_k: int = 3
) -> list[int]:
    """Score each candidate chunk against the query with a reranker. Chunk positions, best first."""
    docs = [chunk_text(corpus[i]) for i in candidates]
    results = (
        voyageai.Client().rerank(query, docs, model=RERANK_MODEL, top_k=top_k).results
    )
    return [candidates[r.index] for r in results]


def rrf(rankings: list[list[int]], top_k: int = 3) -> list[int]:
    """Fuse ranked lists: each chunk scores 1/(60 + rank) per list, summed. Best first."""
    scores = Counter()
    for ranking in rankings:
        for rank, i in enumerate(ranking, start=1):
            scores[i] += 1 / (RRF_K + rank)
    return [i for i, _ in scores.most_common(top_k)]


TOOLS = [
    {
        "name": "search_policies",
        "description": "Search UnitedHealthcare commercial medical policy PDFs by keyword and by meaning. Returns the top matching chunks, each tagged [filename.pdf p.N].",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to look for: a question, a phrase, or a code such as A0431.",
                }
            },
            "required": ["query"],
        },
    }
]

SYSTEM = (
    "You answer questions about UnitedHealthcare medical policies using only the "
    "search_policies tool. Every claim must cite its source with the exact tag "
    "that heads the chunk, for example [uhc-example-policy.pdf p.3]. "
    "Never write the word 'file' in a citation. "
    "If the chunks don't answer the question, say so."
)

MODEL = "claude-haiku-4-5-20251001"


def extract_metadata(text: str) -> PolicyMeta:
    """Ask the model for payer, title, policy number and effective date from page 1."""
    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=MODEL,
        max_tokens=300,
        messages=[
            {
                "role": "user",
                "content": f"Pull the policy metadata from this first page. Give effective_date as YYYY-MM-DD.\n\n{text}",
            }
        ],
        output_format=PolicyMeta,
    )
    return response.parsed_output


MAX_TURNS = 6


def run_agent(question: str, corpus: list[dict]) -> str:
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": question}]
    db = build_index(corpus)
    vectors = embed([chunk_text(c) for c in corpus], "document")

    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")

        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  -> {block.name}({block.input})")
                output = search_policies(block.input["query"], corpus, db, vectors)
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output}
                )
        messages.append({"role": "user", "content": results})

    return "Stopped: too many tool calls."


if __name__ == "__main__":
    import sys

    question = (
        " ".join(sys.argv[1:])
        or "Does UHC cover bilateral cochlear implants for adults?"
    )
    corpus = load_corpus()
    print(run_agent(question, corpus))
