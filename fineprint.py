import hashlib
import json
import re
import time
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


META_PATH = Path(__file__).parent / "meta.json"  # page-1 metadata, keyed by the PDF's hash


def load_corpus(data_dir: Path = DATA_DIR) -> list[dict]:
    """One entry per chunk. File, page, heading and policy metadata travel with the text."""
    saved = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    chunks = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        doc = pymupdf.open(pdf_path)
        key = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
        if key not in saved:  # the model reads page 1 once per PDF, not on every rebuild
            first = doc[0]
            meta = extract_metadata(first.get_text(textpage=ocr_textpage(first)))
            saved[key] = meta.model_dump()
            META_PATH.write_text(json.dumps(saved, indent=1))
        doc_chunks, headings = chunk_by_heading(pdf_path.name, doc)
        doc_chunks += table_chunks(pdf_path.name, doc, headings)
        for chunk in doc_chunks:
            chunk.update(saved[key])
        chunks.extend(doc_chunks)
    return chunks


INDEX_PATH = Path(__file__).parent / "index.json"


def fingerprint(data_dir: Path) -> str:
    """Hash of this file, uv.lock and every PDF. If any of them changes, the chunks are cut again."""
    here = Path(__file__)
    h = hashlib.sha256(here.read_bytes())
    h.update((here.parent / "uv.lock").read_bytes())
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        h.update(pdf_path.name.encode())
        h.update(pdf_path.read_bytes())
    return h.hexdigest()


def embed_key(text: str) -> str:
    """Names one embedding: a hash of the embedding model and the exact text sent to it."""
    return hashlib.sha256(f"{EMBED_MODEL}\n{text}".encode()).hexdigest()


def load_index(data_dir: Path = DATA_DIR) -> tuple[list[dict], list[list[float]]]:
    """Chunks and their vectors. Read from index.json if it is current, else built and saved.

    A rebuild cuts the chunks again but embeds only text that was not embedded before.
    A code change that leaves the chunks alone costs no embedding calls.
    """
    current = fingerprint(data_dir)
    known = {}  # embed_key -> saved vector
    try:
        saved = json.loads(INDEX_PATH.read_text())
        if saved["fingerprint"] == current:
            return saved["chunks"], saved["vectors"]
        known = dict(zip(saved["embedded"], saved["vectors"]))
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        pass  # no file, a damaged one, or one from before the keys were saved
    chunks = load_corpus(data_dir)
    texts = [chunk_text(c) for c in chunks]
    keys = [embed_key(t) for t in texts]
    new = {k: t for k, t in zip(keys, texts) if k not in known}
    if new:
        print(f"embedding {len(new)} of {len(texts)} chunks")
        known.update(zip(new, embed(list(new.values()), "document")))
    vectors = [known[k] for k in keys]
    tmp = INDEX_PATH.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {
                "fingerprint": current,
                "embedded": keys,
                "chunks": chunks,
                "vectors": vectors,
            }
        )
    )
    tmp.replace(INDEX_PATH)  # swapped in whole, so a stopped run can't leave half a file
    return chunks, vectors


HEADING_RATIO = 1.15  # a line this much bigger than body text is a heading

SAME_LEVEL = 1.1  # headings within 10% of each other in size are the same level
WRAP_AT = 0.75  # a heading line that reaches this far across the page continues on the next

MIN_CHARS_PER_PAGE = 50  # less than this and the page is a scan, not text


def is_scan(page) -> bool:
    """True for a page that is a picture with no text layer."""
    return len(page.get_text().strip()) < MIN_CHARS_PER_PAGE and bool(page.get_images())


def ocr_textpage(page):
    """OCR a page that is an image with nothing under it. None means use the text layer."""
    if not is_scan(page):
        return None
    print(f"  ocr: {page.parent.name} p.{page.number + 1}")
    return page.get_textpage_ocr(language="eng", dpi=300, full=True)


def page_lines(page) -> list[list[dict]]:
    """Text blocks of a page outside its tables, each a list of lines: text, font size, height on page."""
    tables = [pymupdf.Rect(t.bbox) for t in real_tables(page)]
    blocks = []
    for block in page.get_text("dict", textpage=ocr_textpage(page))["blocks"]:
        if block["type"] != 0:  # 0 = text, 1 = image
            continue
        lines = []
        for line in block["lines"]:
            x0, y0, x1, y1 = line["bbox"]
            middle = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)
            if any(t.contains(middle) for t in tables):
                continue  # this text is already in a table chunk
            text = " ".join(s["text"] for s in line["spans"]).strip()
            if text:
                lines.append(
                    {
                        "text": text,
                        "size": max(s["size"] for s in line["spans"]),
                        "y": y0,
                        "wraps": x1 >= page.rect.width * WRAP_AT,
                    }
                )
        if lines:
            blocks.append(lines)
    return blocks


def mask_digits(text: str) -> str:
    """Swap every number for #, so "Page 3 of 7" and "Page 4 of 7" read the same."""
    return re.sub(r"\d+", "#", text)


def repeated_lines(pages: list) -> set[str]:
    """Lines printed on most pages of a document: its headers and footers, digits masked."""
    if len(pages) < 3:  # too few pages to tell a footer from ordinary text
        return set()
    seen = Counter()
    for _, blocks in pages:
        seen.update({mask_digits(line["text"]) for lines in blocks for line in lines})
    return {
        text
        for text, count in seen.items()
        if count > len(pages) / 2 and len(text) >= 10  # short lines repeat by chance
    }


def chunk_by_heading(file: str, doc) -> tuple[list[dict], list[tuple]]:
    """Cut a document into chunks: each starts at a heading line, runs to the next.

    Also returns where each heading sits, as (page, height on page, heading path),
    so a table can be given the section it is in.
    """
    pages = [(n, page_lines(p)) for n, p in enumerate(doc, start=1)]
    sizes = Counter()
    for _, blocks in pages:
        for lines in blocks:
            for line in lines:
                sizes[round(line["size"], 1)] += len(line["text"])
    body = sizes.most_common(1)[0][0]
    repeated = repeated_lines(pages)

    chunks = []
    headings = []
    current = {"file": file, "page": 1, "heading": "", "text": ""}
    open_headings = []  # the heading we are under at each level, biggest font first
    levels = not any(is_scan(p) for p in doc)  # OCR font sizes are too noisy to tell levels
    for page_number, blocks in pages:
        if current["text"]:
            chunks.append(current)
        current = {**current, "page": page_number, "text": ""}

        for lines in blocks:
            wrapped = None  # the heading line just read, if it ran to the right margin
            for line in lines:
                previous, wrapped = wrapped, None
                if line["size"] > body * HEADING_RATIO and len(line["text"]) < 120:
                    if levels and previous and abs(previous["size"] - line["size"]) < 0.2:
                        # the second line of a heading: add it to the heading, start nothing new
                        joined = open_headings[-1]["text"] + " " + line["text"]
                        open_headings[-1] = {**open_headings[-1], "text": joined}
                        headings.pop()
                    else:
                        if current["text"]:
                            chunks.append(current)
                        # a new heading closes every open one that is not clearly bigger
                        while open_headings and (
                            not levels
                            or open_headings[-1]["size"] <= line["size"] * SAME_LEVEL
                        ):
                            open_headings.pop()
                        open_headings.append(line)
                    # the first level is the document title, which the prefix already has
                    path = open_headings[1:] or open_headings
                    current = {
                        "file": file,
                        "page": page_number,
                        "heading": " > ".join(h["text"] for h in path),
                        "text": "",
                    }
                    headings.append((page_number, open_headings[-1]["y"], current["heading"]))
                    wrapped = line if line["wraps"] else None
                elif mask_digits(line["text"]) not in repeated:
                    current["text"] += line["text"] + " "
            if current["text"]:
                current["text"] = (
                    current["text"].rstrip() + "\n"
                )  # a block ends a paragraph
    if current["text"]:
        chunks.append(current)
    return [piece for c in chunks for piece in split_long(c)], headings


MAX_COLUMN_NAME = 40  # characters; longer than this and the cell is a sentence


def real_tables(page) -> list:
    """Tables on a page that start with a row of short column names.

    PyMuPDF also reports boxes drawn around prose or a sidebar. Those don't.
    """
    found = []
    for table in page.find_tables().tables:
        header = [cell.strip() for cell in table.extract()[0] if cell and cell.strip()]
        if len(header) >= 2 and max(map(len, header)) <= MAX_COLUMN_NAME:
            found.append(table)
    return found


def table_chunks(file: str, doc, headings: list[tuple]) -> list[dict]:
    """A chunk per table, more if it is long. Each row becomes one line: cell | cell | cell.

    The heading is the section the table sits in, then its column names.
    """
    chunks = []
    for page_number, page in enumerate(doc, start=1):
        for table in real_tables(page):
            rows = table.extract()  # list of rows, each a list of cell strings or None
            lines = [
                " | ".join(cell.strip() for cell in row if cell and cell.strip())
                for row in rows
                if any(cell for cell in row)
            ]
            if not lines:
                continue
            above = [h for page, y, h in headings if (page, y) <= (page_number, table.bbox[1])]
            chunks.append(
                {
                    "file": file,
                    "page": page_number,
                    "heading": " > ".join(above[-1:] + [lines[0]]),
                    "text": "\n".join(lines),
                }
            )
    # long tables are cut between rows; the column names ride along as the heading
    return [piece for c in chunks for piece in split_long(c)]


MAX_CHUNK_CHARS = 1500  # ~375 tokens


def split_long(chunk: dict) -> list[dict]:
    """Split a chunk over the cap at paragraph breaks, then sentence ends. Same heading on each."""
    if len(chunk["text"]) <= MAX_CHUNK_CHARS:
        return [chunk]
    units = []
    for para in chunk["text"].splitlines(keepends=True):
        if len(para) > MAX_CHUNK_CHARS:
            units += re.split(
                r"(?<=\. )", para
            )  # a paragraph over the cap breaks at sentence ends
        else:
            units.append(para)
    pieces, buf = [], ""
    for unit in units:
        if buf and len(buf) + len(unit) > MAX_CHUNK_CHARS:
            pieces.append({**chunk, "text": buf})
            buf = ""
        buf += unit
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


CANDIDATES = 20  # how many hits each search hands to the fusion


def search_policies(
    query: str,
    corpus: list[dict],
    db: sqlite3.Connection,
    vectors: list[list[float]],
    top_k: int = 3,
) -> str:
    """The tool the model calls. Fuses keyword and vector hits, tags each with its [file p.N].

    Under the tag each chunk says which policy it is from and when that policy took effect.
    """
    keyword_hits = keyword_search(query, db, CANDIDATES)
    vector_hits = vector_search(query, vectors, CANDIDATES)
    fused = rrf([keyword_hits, vector_hits], CANDIDATES)
    hits = [corpus[i] for i in rerank(query, fused, corpus, top_k)]

    return (
        "\n\n---\n\n".join(f"{chunk_header(c)}\n{c['text'][:4000]}" for c in hits)
        or "No matching chunks."
    )


def chunk_header(c: dict) -> str:
    """The three lines above a chunk's text: its [file p.N] tag, its policy, its heading."""
    return (
        f"[{c['file']} p.{c['page']}]\n"
        f"{c['title']} ({c['payer']}), policy {c['policy_number']}, "
        f"effective {c['effective_date']}\n"
        f"{c['heading']}"
    )


MAX_CODE_HITS = 5  # chunks shown for one code


def has_code(code: str, text: str) -> bool:
    """True when the code is in the text as a whole token, so 43 does not match 43889 or 43.1%."""
    whole = rf"(?<![A-Za-z0-9.]){re.escape(code)}(?![A-Za-z0-9]|\.\d)"
    return re.search(whole, text) is not None


def lookup_code(code: str, corpus: list[dict]) -> str:
    """The second tool. Every chunk that mentions a billing code or modifier, code tables first.

    Plain text matching over the chunks, so it calls no API and costs nothing. The whole
    chunk is returned because a table's group headers and notes change what a row means.
    """
    code = re.sub(r"^(CPT|HCPCS|MODIFIER)\s+", "", code.strip().upper()).strip(" .,;:*()'\"-")
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9.]{0,7}", code):
        return "That is not a billing code. Give one code, such as 43775 or A0431."
    found = [c for c in corpus if has_code(code, c["text"])]
    if not found:
        return f"{code} does not appear in these policies."
    # Code tables first. A number can also turn up in a study or a reference list.
    found.sort(key=lambda c: not c["heading"].startswith("Applicable"))
    shown = "\n\n---\n\n".join(
        f"{chunk_header(c)}\n{c['text']}" for c in found[:MAX_CODE_HITS]
    )
    more = len(found) - MAX_CODE_HITS
    return shown + (f"\n\n({more} more places not shown.)" if more > 0 else "")


MAX_SECTION_CHARS = 10000  # a few thousand tokens, enough for any section that holds rules


def squash(text: str) -> str:
    """Lower case with runs of spaces closed up, so OCR spacing doesn't break a comparison."""
    return re.sub(r"\s+", " ", text).strip().lower()


def get_policy(policy: str, section: str, corpus: list[dict]) -> str:
    """The third tool. One whole section of one policy in page order, or the list of its sections.

    Search returns the three best chunks, which can miss part of a list of criteria.
    This returns the section from start to end. Plain lookups over the chunks, no API call.
    """
    files = {}  # file name -> its chunks, in page order
    for c in sorted(corpus, key=lambda c: c["page"]):
        files.setdefault(c["file"], []).append(c)
    want = squash(policy).removesuffix(".pdf")
    filler = {"the", "policy", "unitedhealthcare", "uhc", "pdf"}
    words = [w for w in re.findall(r"[a-z0-9]+", want) if w not in filler]
    matches = [
        f
        for f, cs in files.items()
        if want in (f.removesuffix(".pdf"), cs[0]["policy_number"].lower())
    ] or [
        f
        for f, cs in files.items()
        if words and all(w in f"{f} {cs[0]['title']}".lower() for w in words)
    ]
    if len(matches) != 1:
        listing = "\n".join(
            f"{f}: {cs[0]['title']}, policy {cs[0]['policy_number']}" for f, cs in files.items()
        )
        problem = "More than one policy matches" if matches else "No policy matches"
        return f"{problem} {policy!r}. Give one of these file names:\n{listing}"
    chunks = files[matches[0]]
    title = squash(chunks[0]["title"])
    want = squash(section)

    def in_section(c: dict) -> bool:
        """The name is in the chunk's heading path, or ends with its section ("Rationale")."""
        levels = [squash(level) for level in c["heading"].split(" > ")]
        if levels[0] == title:  # the page-1 chunk under the policy's own name
            return want == title
        whole = re.compile(rf"(?<![a-z0-9]){re.escape(want)}(?![a-z0-9])")  # whole words only
        return any(whole.search(level) for level in levels) or want.endswith(levels[0])

    picked = [c for c in chunks if want and in_section(c)]
    if not picked:
        pages = {}
        for c in chunks:
            pages.setdefault(c["heading"].split(" > ")[0], []).append(c["page"])
        listing = "\n".join(f"{name} (p.{min(ps)}-{max(ps)})" for name, ps in pages.items())
        problem = f"No section matches {section!r}. " if want else ""
        return f"{problem}Sections of {matches[0]}:\n{listing}"
    first = picked[0]
    out = (
        f"{first['title']} ({first['payer']}), policy {first['policy_number']}, "
        f"effective {first['effective_date']}"
    )
    for c in picked:
        piece = f"\n\n[{c['file']} p.{c['page']}]\n{c['heading']}\n{c['text']}"
        if len(out) + len(piece) > MAX_SECTION_CHARS:
            return out + (
                f"\n\n(Stopped at {MAX_SECTION_CHARS:,} characters. The rest of this "
                "section is not shown. Use search_policies to find what you need in it.)"
            )
        out += piece
    return out


REQUESTS_DIR = Path(__file__).parent / "requests"  # approved drafts. Not in git.
SOURCE = re.compile(r"\[([\w.-]+\.pdf) p\.([1-9][0-9]{0,3})\]", re.ASCII)


def draft_prior_auth(args: dict, corpus: list[dict]) -> str:
    """The text of a prior-authorization request, built from what the model found. Saves nothing.

    Each part is checked against the chunks: the policy must exist, each code must appear
    in it, and each requirement must cite a page of it. A ValueError carries a message for
    the model. The draft has no field for a patient, and each free-text part is kept to
    one line so it cannot add lines of its own. Keeping patient details out of that text
    is left to the system prompt.
    """
    chunks = [c for c in corpus if c["file"] == args.get("policy")]
    if not chunks:
        files = ", ".join(sorted({c["file"] for c in corpus}))
        raise ValueError(f"policy must be one of these file names: {files}")
    service = args.get("service")
    if not isinstance(service, str) or not service.strip():
        raise ValueError("service must say which procedure or service is being requested.")
    codes = args.get("codes") or []
    if not isinstance(codes, list) or not all(isinstance(code, str) for code in codes):
        raise ValueError("codes must be a list of billing codes.")
    codes = [code.strip().upper() for code in codes]
    for code in codes:
        if not code or not any(has_code(code, c["text"]) for c in chunks):
            raise ValueError(f"code {code!r} is not in {args['policy']}. Check it with lookup_code.")
    requirements = args.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        raise ValueError("requirements must list at least one thing the policy asks for.")
    pages = {c["page"] for c in chunks}
    lines = []
    for number, item in enumerate(requirements, start=1):
        text = item.get("text") if isinstance(item, dict) else None
        source = SOURCE.fullmatch(str(item.get("source", "")).strip()) if text else None
        if not isinstance(text, str) or not text.strip() or not source:
            raise ValueError(
                f"requirement {number} needs a text and a source such as [{args['policy']} p.2]."
            )
        if source[1] != args["policy"] or int(source[2]) not in pages:
            raise ValueError(f"requirement {number} cites {source[0]}, which is not a page of this policy.")
        lines.append(f"{number}. {' '.join(text.split())} {source[0]}")
    first = chunks[0]
    return "\n".join(
        [
            "PRIOR AUTHORIZATION REQUEST (DRAFT)",
            f"Policy: {first['title']} ({first['payer']}), policy {first['policy_number']}, "
            f"effective {first['effective_date']}",
            f"Service requested: {' '.join(service.split())}",
            f"Billing codes: {', '.join(codes) or 'none given'}",
            "",
            "What the policy asks for, to be documented by the provider:",
            *lines,
            "",
            "Patient and provider details: to be filled in by the provider's office.",
            "Check every line against the policy before sending.",
        ]
    )


def save_request(text: str) -> Path:
    """Write an approved draft to requests/ and return where it went."""
    REQUESTS_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = REQUESTS_DIR / f"prior-auth-{stamp}.txt"
    count = 1
    while path.exists():  # a second draft in the same second must not replace the first
        count += 1
        path = REQUESTS_DIR / f"prior-auth-{stamp}-{count}.txt"
    path.write_text(text + "\n")
    return path


EMBED_MODEL = "voyage-4"

TIMEOUT = 60  # seconds before one API call is given up on
RETRIES = 3  # more tries after a failure the SDK treats as temporary, such as a rate limit
EMBED_BATCH = 100  # texts per embedding request, so each one finishes well inside TIMEOUT

# Failures that can pass on their own. A bad key or a bug is not one and is left to crash.
VOYAGE_DOWN = (
    voyageai.error.RateLimitError,
    voyageai.error.ServiceUnavailableError,
    voyageai.error.ServerError,
    voyageai.error.Timeout,
    voyageai.error.APIConnectionError,
    voyageai.error.APIError,  # a server error with no known status, such as a proxy's error page
)
CLAUDE_DOWN = (
    anthropic.APIConnectionError,  # includes timeouts
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.OverloadedError,
)


def voyage() -> voyageai.Client:
    """A Voyage client with retries and a shorter timeout. The SDK's defaults are none and 10 minutes.

    The SDK counts attempts, not retries, so RETRIES + 1 gives RETRIES retries.
    """
    return voyageai.Client(max_retries=RETRIES + 1, timeout=TIMEOUT)


def embed(texts: list[str], input_type: str) -> list[list[float]]:
    """One vector per text. input_type is "document" for chunks, "query" for the question."""
    client = voyage()
    vectors = []
    for start in range(0, len(texts), EMBED_BATCH):
        batch = texts[start : start + EMBED_BATCH]
        vectors += client.embed(batch, model=EMBED_MODEL, input_type=input_type).embeddings
    return vectors


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
        voyage().rerank(query, docs, model=RERANK_MODEL, top_k=top_k).results
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
    },
    {
        "name": "lookup_code",
        "description": "Find every place one billing code or modifier (CPT, HCPCS, such as 43775, A0431 or QL) appears in the policies. Returns each chunk that mentions it, code tables first, tagged [filename.pdf p.N]. It tells you what the code is. Most code tables don't say whether the service is covered: use search_policies with the procedure's name for that.",
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "One code or modifier, for example 43889.",
                }
            },
            "required": ["code"],
        },
    },
    {
        "name": "get_policy",
        "description": "Read one whole section of one policy from start to end, for example the Coverage Rationale of the bariatric surgery policy. Use it when the answer depends on a complete list of criteria or conditions, which a search can return only part of. Leave section empty to list the policy's sections. Each part is tagged [filename.pdf p.N].",
        "input_schema": {
            "type": "object",
            "properties": {
                "policy": {
                    "type": "string",
                    "description": "The policy's file name, or a word from its file name or title, for example bariatric.",
                },
                "section": {
                    "type": "string",
                    "description": "The section's name, for example Coverage Rationale, Benefit Considerations or Definitions. Empty lists the sections.",
                },
            },
            "required": ["policy"],
        },
    },
    {
        "name": "draft_prior_auth",
        "description": "Draft a prior-authorization request for one service under one policy, for the user to approve. Use it only when the user asks for a request to be drafted, and only after you have read the policy's requirements. The user is shown the draft and decides whether it is saved: nothing is saved without their approval. The draft has no patient details and you must not add any.",
        "input_schema": {
            "type": "object",
            "properties": {
                "policy": {
                    "type": "string",
                    "description": "The policy's exact file name, for example uhc-bariatric-surgery.pdf.",
                },
                "service": {
                    "type": "string",
                    "description": "The procedure or service being requested.",
                },
                "codes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Billing codes for the service. Each must appear in the policy.",
                },
                "requirements": {
                    "type": "array",
                    "description": "Everything the policy says must be met or documented for this service, one item each.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "One requirement, in the policy's terms."},
                            "source": {"type": "string", "description": "The tag of the page it is on, for example [uhc-bariatric-surgery.pdf p.2]."},
                        },
                        "required": ["text", "source"],
                    },
                },
            },
            "required": ["policy", "service", "requirements"],
        },
    },
]

SYSTEM = """\
You answer questions about UnitedHealthcare commercial medical policies. Your only sources are the search_policies, lookup_code and get_policy tools. What you already know about insurance, billing codes or other insurers is not a source.

How to search
- Search before you answer, and search again when the first results don't settle the question.
- For a question about a billing code, use lookup_code to learn which procedure the code is. If the table doesn't say whether the code is covered, search for the procedure by name.
- Each chunk names its section above the text. Clinical Evidence and References are background: studies and what other organizations recommend. Never present them as UnitedHealthcare's rule. The rule is usually under Coverage Rationale or Benefit Considerations. If background is all you found, search again for the rule.
- A search returns three chunks, which can be only part of a list. When the answer depends on a full set of criteria or conditions, read the whole section with get_policy before you answer.

How to answer
- Start with the direct answer.
- Give every condition that applies. When coverage depends on criteria, list all of them, including general requirements that apply to every patient. An answer that leaves out a condition is wrong.
- Say only what the chunks say. Don't add a billing code, a rule, or a contrast with a case the question didn't ask about unless a chunk states it.
- Answer the question that was asked, then stop. Leave out related rules, other patient groups and background the question didn't ask for, even when a chunk states them.
- Cite every claim with the exact tag that heads its chunk, for example [uhc-example-policy.pdf p.3]. Never write the word 'file' in a citation.
- Put one page in each tag. For a claim that rests on two pages, write two tags, [uhc-example-policy.pdf p.3] [uhc-example-policy.pdf p.4], never a range or a list of pages.
- A file whose name ends in -scan was read by OCR, which can misread a character. If it disagrees with another copy of the same policy, go with the other copy and don't report the difference as a second policy.

Drafting a request
- Only when the user asks you to draft a prior-authorization request, use draft_prior_auth. First read the policy's requirements for that service with get_policy, then put every one of them in the draft with the page it is on.
- The user decides whether the draft is saved. Tell them what happened: saved and where, or not saved.
- Never put a patient's name, date of birth, member number or medical details in a draft, even if the user gives them. Those are filled in by the provider's office.

When the policies don't answer
- If the question is about another insurer, say you only have UnitedHealthcare policies and stop. Don't answer from UnitedHealthcare policy instead.
- If your searches found nothing that answers the question, say the policies you can search don't address it. That is different from "not covered": only say a service is not covered when a chunk says so. Don't guess.
"""

MODEL = "claude-haiku-5-5"

# Dollars per million tokens for MODEL, by the names the API reports usage under.
PRICE = {
    "input_tokens": 0.10,
    "output_tokens": 0.50,
    "cache_creation_input_tokens": 0.125,
    "cache_read_input_tokens": 0.01,
}


def cost_usd(tokens: dict) -> float:
    """What a token count costs at PRICE."""
    return sum(tokens[kind] * PRICE[kind] for kind in PRICE) / 1_000_000


def extract_metadata(text: str) -> PolicyMeta:
    """Ask the model for payer, title, policy number and effective date from page 1."""
    client = anthropic.Anthropic(max_retries=RETRIES, timeout=TIMEOUT)
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
MAX_TOKENS = 2048  # output cap for one model call. The model's thinking counts toward it.


def run_tool(
    block, corpus: list[dict], db: sqlite3.Connection, vectors: list[list[float]], approve=None
) -> dict:
    """Run one tool call and wrap what it returns as a tool_result.

    A bad request from the model, or a search service that is still down after the
    retries, goes back to the model as an error result. It can then fix its request,
    search again or say it could not look. The details are printed here and not shown
    to the model.

    draft_prior_auth is the one tool that changes anything, and only after approve(text)
    returns True. With no approve function the draft is never saved.
    """
    result = {"type": "tool_result", "tool_use_id": block.id}
    schema = next((t["input_schema"] for t in TOOLS if t["name"] == block.name), None)
    if schema is None:
        return {**result, "content": f"There is no tool named {block.name}.", "is_error": True}
    if block.name == "draft_prior_auth":
        try:
            text = draft_prior_auth(block.input, corpus)
        except ValueError as e:
            return {**result, "content": f"The draft was not accepted: {e}", "is_error": True}
        if approve is None:
            content = "No one is here to approve the draft, so it was not saved. Show it to the user."
        elif approve(text) is True:  # nothing but a plain yes saves
            try:
                content = f"The user approved the draft. It was saved to requests/{save_request(text).name}."
            except OSError as e:
                print(f"  save error: {type(e).__name__}: {e}")
                content = "The user approved the draft, but it could not be saved."
        else:
            content = "The user declined the draft. Nothing was saved."
        return {**result, "content": f"{content}\n\n{text}"}
    args = {name: block.input.get(name) for name in schema["properties"]}
    for name, value in args.items():
        if value is None and name not in schema["required"]:
            args[name] = value = ""  # an optional argument left out
        if not isinstance(value, str) or (name in schema["required"] and not value.strip()):
            content = f"{block.name} needs a {name} string."
            return {**result, "content": content, "is_error": True}
    try:
        if block.name == "lookup_code":
            return {**result, "content": lookup_code(args["code"], corpus)}
        if block.name == "get_policy":
            return {**result, "content": get_policy(args["policy"], args["section"], corpus)}
        return {**result, "content": search_policies(args["query"], corpus, db, vectors)}
    except VOYAGE_DOWN as e:
        print(f"  tool error: {type(e).__name__}: {e}")
        return {**result, "content": "The search failed. Try once more.", "is_error": True}


def ask_to_save(text: str) -> bool:
    """Show a draft in the terminal and ask whether to save it. Anything but y or yes is a no."""
    print(f"\n{text}\n")
    try:
        return input("Save this request? [y/N] ").strip().lower() in ("y", "yes")
    except EOFError:  # no one at the keyboard
        return False


def run_agent(
    question: str, corpus: list[dict], vectors: list[list[float]], approve=None
) -> dict:
    """Answer one question. Returns the answer with its model calls, tokens, cost and time.

    approve is called with the text of a drafted request and returns whether to save it.
    Left out, as in the eval runs, nothing is ever saved.
    The cost covers the Claude calls only. Voyage search calls are not counted.
    """
    client = anthropic.Anthropic(max_retries=RETRIES, timeout=TIMEOUT)
    messages = [{"role": "user", "content": question}]
    db = build_index(corpus)
    started = time.perf_counter()
    tokens = dict.fromkeys(PRICE, 0)
    calls = 0
    answer = "Stopped: too many tool calls."

    for turn in range(MAX_TURNS):
        last = turn == MAX_TURNS - 1  # no more searching: answer from what was found
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                # Two cache points. The one on the system prompt covers the tools and the
                # prompt, which are the same for every question. The one on the request as
                # a whole covers the conversation so far, which the next call resends.
                # A cached part is read at a tenth of the input price; writing it costs a
                # quarter more than input.
                system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
                tools=TOOLS,
                tool_choice={"type": "none" if last else "auto"},
                messages=messages,
                cache_control={"type": "ephemeral"},
            )
        except CLAUDE_DOWN as e:  # still failing after the retries
            print(f"  model error: {type(e).__name__}: {e}")
            answer = "Stopped: the model could not be reached."
            break
        calls += 1
        for kind in PRICE:
            tokens[kind] += getattr(response.usage, kind, None) or 0
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            answer = "".join(b.text for b in response.content if b.type == "text")
            if response.stop_reason == "max_tokens":
                print("  cut off at MAX_TOKENS")
                answer = answer or "Stopped: the answer ran past the length limit."
            break

        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  -> {block.name}({block.input})")
                results.append(run_tool(block, corpus, db, vectors, approve))
        messages.append({"role": "user", "content": results})

    return {
        "answer": answer,
        "model_calls": calls,
        **tokens,
        "cost_usd": round(cost_usd(tokens), 6),
        "seconds": round(time.perf_counter() - started, 1),
    }


def usage_line(run: dict) -> str:
    """One line with what a run used."""
    tokens_in = sum(run[kind] for kind in PRICE if kind != "output_tokens")
    return (
        f"{run['model_calls']} model calls, {tokens_in:,} tokens in "
        f"({run['cache_read_input_tokens']:,} from cache), "
        f"{run['output_tokens']:,} out, ${run['cost_usd']:.4f}, {run['seconds']}s"
    )


if __name__ == "__main__":
    import sys

    question = (
        " ".join(sys.argv[1:])
        or "Does UHC cover bilateral cochlear implants for adults?"
    )
    corpus, vectors = load_index()
    run = run_agent(question, corpus, vectors, approve=ask_to_save)
    print(run["answer"])
    print(usage_line(run))
