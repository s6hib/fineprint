import re
from pathlib import Path
import anthropic
import pymupdf
from collections import Counter
import sqlite3

pymupdf.no_recommend_layout()  # stop the pymupdf_layout hint on every run

DATA_DIR = Path(__file__).parent / "data"


def load_corpus(data_dir: Path = DATA_DIR) -> list[dict]:
    """One entry per chunk. File, page and heading travel with the text."""
    chunks = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        doc = pymupdf.open(pdf_path)
        chunks.extend(chunk_by_heading(pdf_path.name, doc))
        chunks.extend(table_chunks(pdf_path.name, doc))
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


def build_index(chunks: list[dict]) -> sqlite3.Connection:
    """In-memory FTS5 table over the chunks. rowid = the chunk's position in the list."""
    db = sqlite3.connect(":memory:")
    db.execute("create virtual table chunks using fts5(heading, text)")
    db.executemany(
        "insert into chunks(rowid, heading, text) values (?, ?, ?)",
        [(i, c["heading"], c["text"]) for i, c in enumerate(chunks)],
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
        "description": "Search UnitedHealthcare commercial medical policy PDFs. Returns the top matching pages, each tagged [filename.pdf p.N].",
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
    "search_policies tool. Every claim must cite its source with the exact tag "
    "that heads the chunk, for example [uhc-example-policy.pdf p.3]. "
    "Never write the word 'file' in a citation. "
    "If the chunks don't answer the question, say so."
)

MODEL = "claude-haiku-4-5-20251001"
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
