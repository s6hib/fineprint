import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel

from citations import citations, page_text
from evals import load_gold
from fineprint import PRICE, RETRIES, TIMEOUT, cost_usd

HERE = Path(__file__).parent
JUDGMENTS = HERE / "judgments.jsonl"

JUDGE_MODEL = "claude-haiku-5-5"  # priced by PRICE in fineprint.py
EXAMPLE_FAILS = 3  # labeled rows shown to the judge as examples
EXAMPLE_PASSES = 3
SEED = 0

RUBRIC = """You grade answers from a question-answering system over UnitedHealthcare medical policies.

For each case you get the question, policy pages, a reference answer, and the system's answer. The pages are the ones the reference comes from and the ones the answer cites. Label the system's answer pass or fail.

Fail when the answer:
- leaves out a condition or criterion that changes the answer
- says it can't tell, or gives no answer, when the pages do answer the question
- contradicts the pages or the reference
- adds a claim about the policy that the pages don't support, such as an extra code, an extra rule or a second policy
- answers a different question from the one asked. When the reference says the question is outside these policies, the answer must say so and must not answer from UnitedHealthcare policy instead.

Pass when the answer gives what the reference gives, even in other words or with more detail that the pages support. Ignore formatting and citation style.

Give your reason in one sentence, then the label."""


class Verdict(BaseModel):
    """The judge's reason comes first so it is written before the label."""

    why: str
    label: Literal["pass", "fail"]


def load_rows() -> list[dict]:
    """One row per gold question: the question, reference, system answer and human label."""
    answers = {
        r["id"]: r["answer"]
        for r in map(json.loads, (HERE / "answers.jsonl").read_text().splitlines())
    }
    labels = {
        r["id"]: r["label"]
        for r in map(json.loads, (HERE / "labels.jsonl").read_text().splitlines())
    }
    return [
        {**g, "reference": g["answer"], "answer": answers[g["id"]], "label": labels[g["id"]]}
        for g in load_gold()
    ]


def split(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """A few labeled rows become examples for the judge. It is scored only on the rest."""
    rng = random.Random(SEED)
    fails = [r for r in rows if r["label"] == "fail"]
    passes = [r for r in rows if r["label"] == "pass"]
    examples = rng.sample(fails, EXAMPLE_FAILS) + rng.sample(passes, EXAMPLE_PASSES)
    example_ids = {r["id"] for r in examples}
    return examples, [r for r in rows if r["id"] not in example_ids]


def source_text(row: dict) -> str:
    """Text of the pages the gold answer lives on, plus every page the answer cites."""
    wanted = {(f, n) for f in row["files"][:1] for n in row["pages"]}
    wanted |= {(f, n) for _, f, n in citations(row["answer"])}
    pages = [
        f"[{f} p.{n}]\n{text}"
        for f, n in sorted(wanted)
        if (text := page_text(f, n)) is not None
    ]
    return "\n\n".join(pages) or "(none)"


def case_text(row: dict) -> str:
    """One case as the judge sees it."""
    return (
        f"<question>{row['question']}</question>\n"
        f"<pages>\n{source_text(row)}\n</pages>\n"
        f"<reference>{row['reference']}</reference>\n"
        f"<answer>\n{row['answer']}\n</answer>"
    )


def system_prompt(examples: list[dict]) -> str:
    """The rubric, then the labeled examples."""
    shown = "\n\n".join(
        f"<example>\n{case_text(e)}\n<label>{e['label']}</label>\n</example>"
        for e in examples
    )
    return f"{RUBRIC}\n\nLabeled examples:\n\n{shown}"


def judge(case: str, system: str) -> tuple[Verdict, dict]:
    """Ask the judge model for a verdict on one case. Returns it with the tokens used.

    The system prompt is the same long text for every case, so it is marked for caching:
    the first call stores it and later calls read it back at a tenth of the input price.
    """
    client = anthropic.Anthropic(max_retries=RETRIES, timeout=TIMEOUT)
    response = client.messages.parse(
        model=JUDGE_MODEL,
        max_tokens=4000,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": case}],
        output_format=Verdict,
    )
    if response.parsed_output is None:
        raise RuntimeError(f"no verdict: {response.stop_reason}")
    tokens = {kind: getattr(response.usage, kind, None) or 0 for kind in PRICE}
    return response.parsed_output, tokens


if __name__ == "__main__":
    examples, held_out = split(load_rows())
    system = system_prompt(examples)
    cases = [case_text(r) for r in held_out]  # pages are read here: OCR can't run in threads
    first = judge(cases[0], system)  # alone, so the cached prompt is stored once, not 8 times
    with ThreadPoolExecutor(8) as pool:
        results = [first, *pool.map(lambda c: judge(c, system), cases[1:])]
    verdicts = [v for v, _ in results]
    tokens = {kind: sum(t[kind] for _, t in results) for kind in PRICE}

    with JUDGMENTS.open("w") as f:
        for row, v in zip(held_out, verdicts):
            f.write(json.dumps({"id": row["id"], "label": v.label, "why": v.why}) + "\n")
            if v.label != row["label"]:
                print(f"  disagree {row['id']}: labeled {row['label']}, judged {v.label}: {v.why}")

    for label, name in [("pass", "passes kept (TPR)"), ("fail", "fails caught (TNR)")]:
        mine = [v for row, v in zip(held_out, verdicts) if row["label"] == label]
        print(f"{name}: {sum(v.label == label for v in mine)}/{len(mine)}")
    print(
        f"judge: {tokens['cache_read_input_tokens']:,} tokens read from cache, "
        f"{tokens['cache_creation_input_tokens']:,} written, "
        f"{tokens['input_tokens']:,} other in, {tokens['output_tokens']:,} out, "
        f"${cost_usd(tokens):.4f}"
    )
