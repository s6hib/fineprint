import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, StringConstraints, field_validator

from fineprint import SOURCE, load_index, run_agent
from limits import Budget, RateLimit

MAX_QUESTION_CHARS = 500  # a question is a sentence or two. Longer input only costs tokens.
STOPPED = "Stopped:"  # how run_agent starts an answer when it gave up
MAX_BODY_BYTES = 8000  # room for 500 characters at 12 bytes each, the most JSON escaping needs
RATE_LIMIT = 5  # questions one address may ask ...
RATE_WINDOW = 60  # ... in this many seconds

# The most the model calls may cost in one UTC day. Set FINEPRINT_DAILY_CAP_USD to change it.
DAILY_CAP_USD = float(os.environ.get("FINEPRINT_DAILY_CAP_USD", "0.10"))

budget = Budget(Path(__file__).parent / "spend.json", DAILY_CAP_USD)
rate = RateLimit(RATE_LIMIT, RATE_WINDOW)


class Question(BaseModel):
    question: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS),
    ]

    @field_validator("question")
    @classmethod
    def has_a_word(cls, value: str) -> str:
        """Turn away input with no letter or digit, such as a lone invisible character."""
        if not any(ch.isalnum() for ch in value):
            raise ValueError("no letters or digits")
        return value


class Citation(BaseModel):
    file: str
    page: int


class Answer(BaseModel):
    answer: str
    citations: list[Citation]
    model_calls: int
    cost_usd: float
    seconds: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the index once at startup. Every request reads the same copy."""
    app.state.corpus, app.state.vectors = load_index()
    yield


# The generated docs pages are off: a public demo has no need to publish its schema.
app = FastAPI(title="fineprint", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.exception_handler(RequestValidationError)
async def bad_request(request: Request, exc: RequestValidationError) -> JSONResponse:
    """One short message for any request that fails validation.

    FastAPI's own reply repeats the rejected input, so a huge request would get a huge reply.
    """
    detail = f"Send a question of 1 to {MAX_QUESTION_CHARS} characters."
    return JSONResponse({"detail": detail}, status_code=422)


@app.middleware("http")
async def guard_posts(request: Request, call_next):
    """Check every POST before any of its body is read: how often, then how big.

    This runs ahead of FastAPI's own handling, which reads and parses the whole body
    before anything else gets a say.
    """
    if request.method == "POST":
        address = request.client.host if request.client else "unknown"
        if not rate.allow(address):
            return JSONResponse(
                {"detail": "Too many questions. Wait a minute and try again."},
                status_code=429,
                headers={"Retry-After": str(RATE_WINDOW)},
            )
        length = request.headers.get("content-length", "")
        # A chunked body has no stated size, and the server would follow the chunks even
        # when a Content-Length is sent with them.
        if "transfer-encoding" in request.headers or not (length.isascii() and length.isdigit()):
            return JSONResponse({"detail": "Send a Content-Length header."}, status_code=411)
        if int(length) > MAX_BODY_BYTES:
            return JSONResponse({"detail": "The request is too large."}, status_code=413)
    return await call_next(request)


def cited_pages(answer: str, corpus: list[dict]) -> list[Citation]:
    """Each page the answer cites, once, in the order it first appears.

    A tag is kept only if a chunk in the index carries it, so every citation sent to the
    browser is a page the agent could have read.
    """
    known = {(c["file"], c["page"]) for c in corpus}
    found = {}  # a dict keeps first-seen order and drops repeats
    for m in SOURCE.finditer(answer):
        key = (m[1], int(m[2]))
        if key in known:
            found[key] = Citation(file=key[0], page=key[1])
    return list(found.values())


@app.get("/health")
def health() -> dict:
    return {"ok": True}


# A plain def, not async: run_agent blocks while it waits on the model, so FastAPI runs
# each request on a worker thread and the server keeps answering others.
@app.post("/ask")
def ask(body: Question, request: Request) -> Answer:
    """Answer one question. No approve function is passed, so a draft is never saved."""
    if not budget.has_room():
        raise HTTPException(
            status_code=503, detail="Today's budget for answers is used up. Try again tomorrow."
        )
    state = request.app.state
    run = run_agent(body.question, state.corpus, state.vectors)
    budget.add(run["cost_usd"])  # a run that gave up still used tokens
    if run["answer"].startswith(STOPPED):  # the model was unreachable or the run hit a limit
        raise HTTPException(status_code=503, detail=run["answer"])
    return Answer(
        answer=run["answer"],
        citations=cited_pages(run["answer"], state.corpus),
        model_calls=run["model_calls"],
        cost_usd=run["cost_usd"],
        seconds=run["seconds"],
    )


# The built web page, when there is one (`make web`). Mounted last, so the routes above
# are matched first and everything else is looked up as a file in web/dist.
WEB_DIR = Path(__file__).parent / "web" / "dist"
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
