# fineprint

Ask a question about a health insurer's medical policy and get an answer with the file and page it came from.

Live demo: https://fineprint.fly.dev

Insurance policies are long PDFs with the useful part buried under headings, tables and billing codes. fineprint pulls a handful of public UnitedHealthcare commercial policies, cuts them into chunks that keep their heading and page number, and gives Claude three tools for reading them: `search_policies`, `lookup_code` for billing codes, and `get_policy` to read a whole section. The model searches, reads what comes back, and answers with a `[file.pdf p.N]` tag on every claim. A fourth tool, `draft_prior_auth`, drafts a prior-authorization request and saves it only if you approve it. The system prompt tells it to say so when the chunks don't answer the question.

```sh
uv sync
./fetch_data.sh                       # five policy PDFs into data/
export ANTHROPIC_API_KEY=... VOYAGE_API_KEY=...
uv run fineprint.py "Does UHC cover bilateral cochlear implants for adults?"
```

The first run parses and embeds the PDFs and saves the result to `index.json`. Later runs read that file. When a PDF, `fineprint.py` or `uv.lock` changes, the chunks are cut again, but only chunks whose text is new are embedded.

## Web app

```sh
make web      # build the page into web/dist (needs Node)
make serve    # the API and the page on http://127.0.0.1:8000
```

`api.py` puts the agent behind one endpoint, `POST /ask`, and serves the page. The page (`web/`, React and TypeScript) sends the question, shows the answer, and draws each citation the API confirmed as a tag. A citation is confirmed only if a chunk in the index carries that file and page. Clicking a tag opens a panel with that page's text as the index holds it and a link to the policy on uhcprovider.com. The PDFs themselves are not served. When the agent drafts a prior-authorization request, the API holds the draft in memory for 15 minutes and the page shows it with Approve and Decline buttons. The draft is written to `requests/` only after an approve. Three limits are checked before the model is called: a daily spending cap (`FINEPRINT_DAILY_CAP_USD`, default $0.10), 5 questions a minute per address, and a request size of 8,000 bytes.

### Deploy

The `Dockerfile` builds the page, installs the locked Python packages and copies in the code, the PDFs and the saved `index.json`. The build stops if that index does not match the files, so a deployed app never re-embeds on its own. `fly.toml` runs the image on Fly.io as one always-on machine with 512 MB of memory and a 1 GB disk for the day's spending total. Approved requests are written next to the code, which Fly replaces at every deploy, so they do not outlive one.

Two settings matter on any host. `FINEPRINT_STATE_DIR` says where `spend.json` is kept, and `FINEPRINT_CLIENT_IP_HEADER` names the header the host's proxy puts the caller's address in (`fly-client-ip` on Fly). Without the second one every caller shares the proxy's address and the rate limit counts them as one.

## How a PDF becomes chunks

- Page 1 of each policy goes to the model once (the result is saved in `meta.json`), which returns the payer, title, policy number and effective date as a typed `PolicyMeta` object. That line is prepended to every chunk so a chunk still says which policy it belongs to.
- Headings are found by font size: the most common size on the page is body text, anything 15% larger and under 120 characters starts a new chunk. A chunk runs to the next heading and closes at every page break so its page number stays right. A chunk under a sub-heading keeps its section name as well, as in `Clinical Evidence > Revisional Surgery`.
- Lines printed on most pages of a document, its headers and footers, are left out. Page numbers are masked first so "Page 3 of 7" and "Page 4 of 7" count as the same line.
- Tables come out as their own chunks, one row per line, cells joined with `|`. A box counts as a table only if its first row is two or more short column names, which leaves out sidebars and boxed paragraphs. Text inside a table is kept only in its table chunk, which is headed by the section it sits in and then its column names, as in `Applicable Codes > CPT Code | Description`. Long tables are split between rows.
- Pages with no text layer are OCR'd (PyMuPDF, 300 dpi).
- Chunks over 1,500 characters are split at paragraph breaks, then at sentence ends, and keep their heading.

## Search

Two searches run on every query and their results are fused:

1. BM25 over an in-memory SQLite FTS5 table. Query terms are quoted so billing codes like `A0431` stay whole.
2. Voyage `voyage-4` embeddings, scored by dot product against every chunk.

The top 20 from each are merged with reciprocal rank fusion, then Voyage `rerank-3` picks the final 3 chunks the model sees. Each chunk comes with its policy's title, number and effective date. `lookup_code` finds every chunk that mentions a billing code or modifier by plain text matching and returns them with code tables first. It calls no API.

`get_policy` returns one whole section of one policy in page order, such as its Coverage Rationale, up to 10,000 characters. Search hands the model three chunks, which can be only part of a list of criteria, so the prompt tells it to read the full section when an answer depends on every condition. With no section named it lists the policy's sections.

`draft_prior_auth` is the one tool that changes anything. Asked to draft a prior-authorization request, the model reads the policy's requirements and passes the tool the policy, the service, any billing codes, and each requirement with the page it is on. Code checks the draft before anyone sees it: the policy has to exist, each code has to appear in that policy, and each requirement has to cite a real page of it. A draft that fails goes back to the model with the reason. A draft that passes is printed, and you are asked whether to save it. Only `y` or `yes` writes it to `requests/`. In the eval runs there is no one to ask, so nothing is saved. The draft has no field for patient details, and the prompt tells the model to leave them out.

```sh
uv run fineprint.py "Draft a prior authorization request for sleeve gastrectomy"
```

The agent loop allows 6 model calls. On the sixth the model can no longer search and has to answer from what it has found. API calls time out after 60 seconds, and a failure that looks temporary, such as a rate limit, is retried up to 3 times. A search that still fails is reported to the model as an error, so it can try again or say it could not look. Every model call is marked for prompt caching in two places: the system prompt and tools, which are the same for every question, and the conversation so far, which the next call in the same question resends. A cached part is read at a tenth of the input price.

## Eval

`make eval` runs the three checks below in order: search recall, citations on the latest run, and the judge scored against the labeled run. `make answers` runs the agent on all 50 questions again, and `make grade` has the judge mark that run.

`gold.jsonl` holds 50 questions with the answer and the file and pages it lives on: coverage criteria, exclusions, billing codes, definitions, policy metadata, and three questions the corpus can't answer.

```sh
uv run evals.py
```

It prints recall@20 after fusion and recall@3 after reranking, and lists every question the reranker missed with whether fusion had found it. Out-of-corpus questions are skipped since there is no page to find.

`answers.py` runs the agent on every gold question and saves what it said to `answers.jsonl`, with the model calls, tokens, cost and time each one took.

An earlier run is kept as `labeled_answers.jsonl`. `labels.jsonl` marks each of its answers pass or fail with a one-line reason, and that run is what the judge is checked against.

```sh
uv run judge.py
```

`judge.py` grades answers with a separate model call. Run like this it scores the judge itself: six labeled answers go to the judge as examples, and it is scored on the other 44: how many passing answers it keeps and how many failing answers it catches. For each answer the judge sees the question, the reference answer, the text of the pages the reference comes from and the pages the answer cites, and the line naming each policy that the agent is shown with every passage. Its verdicts go to `judgments.jsonl`. The rubric and examples are the same for every case, so they are cached: after the first call the rest read them at a tenth of the input price, and the run prints what it cost.

```sh
uv run judge.py --cli
```

With `--cli` the same rubric and cases go to a bigger model, Sonnet 5.5, through the Claude Code CLI (`claude -p`). The CLI uses whatever account it is signed in to, so with a Claude subscription login that run is not billed to the API key. The script removes the key from the CLI's environment and stops if the CLI reports it is using one. Each answer is judged three times and the label most verdicts give is kept, since a single verdict on a borderline answer can go either way from run to run. It needs `claude` on your PATH, or `CLAUDE_BIN` set to it.

```sh
uv run judge.py --grade --cli
```

With `--grade` the judge marks the latest run in `answers.jsonl`, which has no labels, and writes its verdicts to `grades.jsonl`. The same six labeled answers are its examples.

```sh
uv run citations.py
```

`citations.py` checks the citations in the latest run, `answers.jsonl`, with plain code and no model. It counts how many answers cite anything, how many citations point to a file and page that exist, how many answers cite a page the gold answer lives on, and how many quoted passages appear on a page the answer cites.

## Files

| File | What it is |
|---|---|
| `fineprint.py` | chunking, index, the four tools, agent loop |
| `evals.py` | recall over the gold set |
| `gold.jsonl` | the 50 questions |
| `answers.py` | runs the agent on every gold question |
| `answers.jsonl` | the agent's 50 answers from the latest run |
| `grades.jsonl` | the judge's verdict on each of those |
| `labeled_answers.jsonl` | the 50 answers from an earlier run |
| `labels.jsonl` | pass or fail for each of those, with a reason |
| `citations.py` | checks the citations and quotes in the answers |
| `judge.py` | grades answers with a separate model call, and scores that judge against the labels |
| `judgments.jsonl` | the judge's verdicts on the 44 labeled answers it was scored on |
| `api.py` | the HTTP API: `POST /ask`, approve and decline for drafts, and the built web page at `/` |
| `limits.py` | the daily spending cap and the per-address rate limit |
| `drafts.py` | drafted requests that wait for an approve or decline |
| `Dockerfile`, `fly.toml` | the image and the Fly.io settings for the deployed demo |
| `web/` | the web page (React, TypeScript, Vite) |
| `Makefile` | `make eval`, `make answers`, `make grade`, `make serve` and `make web` |
| `fetch_data.sh` | downloads the policy PDFs into `data/` (git-ignored) |
