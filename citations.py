import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pymupdf

from evals import load_gold
from fineprint import DATA_DIR, ocr_textpage

ANSWERS = Path(__file__).parent / "answers.jsonl"

CITATION = re.compile(r"\[([\w.-]+\.pdf) p\.(\d+)(?:-(\d+))?\]")
TAG = re.compile(r"\[[^\[\]\n]*\.pdf[^\[\]\n]*\]")  # anything that looks like a citation tag
QUOTE = re.compile(r'"([^"\n]{15,})"')  # shorter quotes are single terms and match anywhere


@lru_cache(maxsize=None)
def page_text(file: str, n: int) -> str | None:
    """Text of page n of a policy, by OCR if the page is a scan. None if there is no such page."""
    path = DATA_DIR / file
    if not path.exists():
        return None
    doc = pymupdf.open(path)
    if not 1 <= n <= len(doc):
        return None
    page = doc[n - 1]
    return page.get_text(textpage=ocr_textpage(page))


def citations(answer: str) -> list[tuple[int, str, int]]:
    """Every (position, file, page) an answer cites. A range like p.6-7 gives one entry per page."""
    found = []
    for m in CITATION.finditer(answer):
        first, last = int(m[2]), int(m[3] or m[2])
        found += [(m.start(), m[1], n) for n in range(first, last + 1)]
    return found


def unreadable_tags(answer: str) -> list[str]:
    """Tags that name a PDF but aren't in the form CITATION reads, so their pages go unchecked."""
    return [t for t in TAG.findall(answer) if not CITATION.fullmatch(t)]


def squash(text: str) -> str:
    """Lowercase letters and digits only, so spacing, bullets and markdown can't break a match."""
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in text if ch.isalnum())


def quote_misses(answer: str) -> list[str]:
    """Quoted passages that are on none of the pages the answer cites."""
    pages = [squash(page_text(f, n) or "") for _, f, n in citations(answer)]
    misses = []
    for quote in QUOTE.findall(answer):
        parts = [squash(p) for p in re.split(r"\.\.\.|…", quote)]  # "..." joins two excerpts
        if not any(all(p in page for p in parts) for page in pages):
            misses.append(quote)
    return misses


if __name__ == "__main__":
    answers = {
        r["id"]: r["answer"] for r in map(json.loads, ANSWERS.read_text().splitlines())
    }
    gold = [r for r in load_gold() if r["files"]]  # out-of-corpus rows have nothing to cite

    cited = on_gold = real = total = quotes = quotes_found = unreadable = 0
    for row in gold:
        answer = answers[row["id"]]
        for tag in unreadable_tags(answer):
            unreadable += 1
            print(f"  {row['id']}: tag not checked: {tag}")
        cites = citations(answer)
        if not cites:
            print(f"  {row['id']}: no citation")
            continue
        cited += 1
        total += len(cites)
        real += sum(page_text(f, n) is not None for _, f, n in cites)
        if any(f in row["files"] and n in row["pages"] for _, f, n in cites):
            on_gold += 1
        else:
            pages = sorted({f"{f} p.{n}" for _, f, n in cites})
            print(f"  {row['id']}: cites {pages}, gold is p.{row['pages']}")
        misses = quote_misses(answer)
        quotes += len(QUOTE.findall(answer))
        quotes_found += len(QUOTE.findall(answer)) - len(misses)
        for q in misses:
            print(f'  {row["id"]}: quote not on a cited page: "{q[:80]}"')

    print(f"answers with a citation:     {cited}/{len(gold)}")
    print(f"answers citing a gold page:  {on_gold}/{len(gold)}")
    print(f"citations to a real page:    {real}/{total}")
    print(f"quotes found on a cited page: {quotes_found}/{quotes}")
    print(f"tags the check couldn't read: {unreadable}")
