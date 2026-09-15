import re
from pathlib import Path
import anthropic
import pymupdf
from collections import Counter

DATA_DIR = Path(__file__).parent / "data"


def load_corpus(data_dir: Path = DATA_DIR) -> list[dict]:
    """One entry per page. File and page number travel with the text."""
    pages = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        doc = pymupdf.open(pdf_path)
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text()
            pages.append({"file": pdf_path.name, "page": page_number, "text": text})
    return pages


HEADING_RATIO = 1.15  # a block this much bigger than body text is a heading
BOLD_FLAG = 16


def page_blocks(page) -> list[dict]:
    """One dict per text block: its text, its dominant font size, bold or not."""
    blocks = []
    for block in page.get_text("dict")["blocks"]:
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


def search_policies(query: str, corpus: list[dict], top_k: int = 3) -> str:
    """Keyword search: score each page by how many query words appear in it."""
    words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 2]
    scored = []
    for page in corpus:
        lowered = page["text"].lower()
        score = sum(lowered.count(w) for w in words)
        if score > 0:
            scored.append((score, page))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    hits = [
        f"[{p['file']} p.{p['page']}]\n{p['text'][:4000]}" for _, p in scored[:top_k]
    ]
    return "\n\n---\n\n".join(hits) if hits else "No matching pages."


TOOLS = [
    {
        "name": "search_policies",
        "description": "Search UnitedHealthcare commercial medical policy PDFs. Returns the top matching pages, each tagged [file p.N].",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Keywords to search for."}
            },
            "required": ["query"],
        },
    }
]

SYSTEM = (
    "You answer questions about UnitedHealthcare medical policies using only the "
    "search_policies tool. Every claim must cite its source as [file p.N]. "
    "If the pages don't answer the question, say so."
)

MODEL = "claude-sonnet-5"
MAX_TURNS = 6


def run_agent(question: str, corpus: list[dict]) -> str:
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": question}]

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
                output = search_policies(block.input["query"], corpus)
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
