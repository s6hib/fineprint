import secrets
import threading
import time

KEEP_SECONDS = 15 * 60  # how long a draft waits for its answer
MAX_WAITING = 100  # drafts held at once. Past this the oldest is dropped.


class Waiting:
    """Drafted requests that wait for a yes or no from the person who asked.

    The agent has finished by the time a draft is on the screen, so the draft is kept
    here under a random id until a second request approves or declines it. It is held
    in memory, so a restart forgets every draft that was waiting.
    """

    def __init__(self, keep: float = KEEP_SECONDS, most: int = MAX_WAITING, clock=time.monotonic):
        self.keep = keep
        self.most = most
        self.clock = clock
        self.held = {}  # id -> (text, when it stops waiting), oldest first
        self.lock = threading.Lock()  # requests run on several threads

    def add(self, text: str) -> str:
        """Hold a draft and return the id that approves or declines it."""
        with self.lock:
            now = self.clock()
            for old in [key for key, (_, until) in self.held.items() if until <= now]:
                del self.held[old]
            while len(self.held) >= self.most:
                del self.held[next(iter(self.held))]
            draft_id = secrets.token_urlsafe(16)  # too long to guess
            self.held[draft_id] = (text, now + self.keep)
            return draft_id

    def take(self, draft_id: str) -> str | None:
        """Hand a draft over and forget it. None if the id is unknown or it waited too long.

        A draft can be taken once, so two clicks on Approve save one file.
        """
        with self.lock:
            text, until = self.held.pop(draft_id, (None, 0))
            return text if text is not None and self.clock() < until else None
