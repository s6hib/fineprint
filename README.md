# fineprint

Ask a question about a health insurer's medical policy and get an answer with the file and page it came from.

Insurance policies are long PDFs with the useful part buried under headings, tables and billing codes. fineprint pulls a handful of public UnitedHealthcare commercial policies, cuts them into chunks that keep their heading and page number, and gives Claude one tool: `search_policies`. The model searches, reads what comes back, and answers with a `[file.pdf p.N]` tag on every claim. The system prompt tells it to say so when the chunks don't answer the question.

```sh
uv sync
./fetch_data.sh                       # five policy PDFs into data/
export ANTHROPIC_API_KEY=... VOYAGE_API_KEY=...
uv run fineprint.py "Does UHC cover bilateral cochlear implants for adults?"
```

The first run parses and embeds the PDFs and saves the result to `index.json`. Later runs read that file and rebuild it only when a PDF, `fineprint.py` or `uv.lock` changes.

## How a PDF becomes chunks

- Page 1 of each policy goes to the model once, which returns the payer, title, policy number and effective date as a typed `PolicyMeta` object. That line is prepended to every chunk so a chunk still says which policy it belongs to.
- Headings are found by font size: the most common size on the page is body text, anything 15% larger and under 120 characters starts a new chunk. A chunk runs to the next heading and closes at every page break so its page number stays right.
- Tables come out as their own chunks, one row per line, cells joined with `|`.
- Pages with no text layer are OCR'd (PyMuPDF, 300 dpi).
- Chunks over 1,500 characters are split at paragraph breaks, then at sentence ends, and keep their heading.

## Search

Two searches run on every query and their results are fused:

1. BM25 over an in-memory SQLite FTS5 table. Query terms are quoted so billing codes like `A0431` stay whole.
2. Voyage `voyage-4` embeddings, scored by dot product against every chunk.

The top 10 from each are merged with reciprocal rank fusion, then Voyage `rerank-3` picks the final 3 chunks the model sees. The agent loop allows up to 6 tool calls before it gives up.

## Eval

`gold.jsonl` holds 50 questions with the answer and the file and pages it lives on: coverage criteria, exclusions, billing codes, definitions, policy metadata, and three questions the corpus can't answer.

```sh
uv run evals.py
```

It prints recall@10 after fusion and recall@3 after reranking, and lists every question the reranker missed with whether fusion had found it. Out-of-corpus questions are skipped since there is no page to find.

`answers.py` runs the agent on every gold question and saves what it said to `answers.jsonl`. `labels.jsonl` marks each answer pass or fail with a one-line reason.

```sh
uv run judge.py
```

`judge.py` grades the answers with a second model. Six labeled answers go to the judge as examples, and it is scored on the other 44: how many passing answers it keeps and how many failing answers it catches. For each answer the judge sees the question, the reference answer, and the text of the pages the reference comes from and the pages the answer cites. Its verdicts go to `judgments.jsonl`.

```sh
uv run citations.py
```

`citations.py` checks the citations in the saved answers with plain code and no model. It counts how many answers cite anything, how many citations point to a file and page that exist, how many answers cite a page the gold answer lives on, and how many quoted passages appear on a page the answer cites.

## Files

| File | What it is |
|---|---|
| `fineprint.py` | chunking, index, search tool, agent loop |
| `evals.py` | recall over the gold set |
| `gold.jsonl` | the 50 questions |
| `answers.py` | runs the agent on every gold question |
| `answers.jsonl` | the agent's 50 answers |
| `labels.jsonl` | pass or fail for each answer, with a reason |
| `citations.py` | checks the citations and quotes in the answers |
| `judge.py` | grades answers with a second model and compares it with the labels |
| `judgments.jsonl` | the judge's verdicts on the 44 answers it was scored on |
| `fetch_data.sh` | downloads the policy PDFs into `data/` (git-ignored) |
