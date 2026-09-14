import re
from pathlib import Path
import anthropic
from pypdf import PdfReader

DATA_DIR = Path(__file__).parent / "data"


def load_corpus(data_dir: Path = DATA_DIR) -> list[dict]:
    """One entry per page. File and page number travel with the text."""
    pages = []
    for pdf_path in sorted(data_dir.glob("*.pdf")):
        reader = PdfReader(pdf_path)
        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append({"file": pdf_path.name, "page": page_number, "text": text})
    return pages


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
