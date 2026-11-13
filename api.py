from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, StringConstraints, field_validator

from fineprint import SOURCE, load_index, run_agent

MAX_QUESTION_CHARS = 500  # a question is a sentence or two. Longer input only costs tokens.
STOPPED = "Stopped:"  # how run_agent starts an answer when it gave up


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


app = FastAPI(title="fineprint", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def bad_request(request: Request, exc: RequestValidationError) -> JSONResponse:
    """One short message for any request that fails validation.

    FastAPI's own reply repeats the rejected input, so a huge request would get a huge reply.
    """
    detail = f"Send a question of 1 to {MAX_QUESTION_CHARS} characters."
    return JSONResponse({"detail": detail}, status_code=422)


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
    state = request.app.state
    run = run_agent(body.question, state.corpus, state.vectors)
    if run["answer"].startswith(STOPPED):  # the model was unreachable or the run hit a limit
        raise HTTPException(status_code=503, detail=run["answer"])
    return Answer(
        answer=run["answer"],
        citations=cited_pages(run["answer"], state.corpus),
        model_calls=run["model_calls"],
        cost_usd=run["cost_usd"],
        seconds=run["seconds"],
    )
