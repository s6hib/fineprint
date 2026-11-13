import json
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path


def today() -> str:
    """The current UTC date, the day a budget is counted over."""
    return datetime.now(timezone.utc).date().isoformat()


class Budget:
    """Dollars spent on model calls today, with a cap.

    The total is written to a small file after every run, so restarting the server does
    not hand out a fresh day's budget. If the file can't be written the total in memory
    still holds, and only a restart would lose it. A run is checked before it starts and counted when
    it ends, so runs already in flight when the cap is reached still finish. The overshoot
    is at most a few runs.
    """

    def __init__(self, path: Path, cap_usd: float, day=today):
        self.path, self.cap, self.clock = path, cap_usd, day
        self.lock = threading.Lock()
        self.day, self.usd = self.clock(), 0.0
        try:
            saved = json.loads(path.read_text())
            if saved["day"] == self.day:
                self.usd = float(saved["usd"])
        except FileNotFoundError:
            pass
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            print(f"  {path.name} could not be read, counting from zero: {type(e).__name__}")

    def roll(self) -> None:
        """Start from zero when the date has changed. Call with the lock held."""
        if self.clock() != self.day:
            self.day, self.usd = self.clock(), 0.0

    def has_room(self) -> bool:
        with self.lock:
            self.roll()
            return self.usd < self.cap

    def add(self, usd: float) -> None:
        with self.lock:
            self.roll()
            self.usd += usd
            tmp = self.path.with_suffix(".tmp")
            try:
                tmp.write_text(json.dumps({"day": self.day, "usd": round(self.usd, 6)}))
                tmp.replace(self.path)  # swapped in whole, so a stopped write can't leave half a file
            except OSError as e:
                print(f"  {self.path.name} could not be written: {type(e).__name__}")


MAX_KEYS = 10_000  # callers remembered at once, so the table can't grow without end


class RateLimit:
    """At most `limit` requests per caller in any `window` seconds. Kept in memory."""

    def __init__(self, limit: int, window: float, clock=time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self.lock = threading.Lock()
        self.seen: dict[str, deque] = {}  # caller -> times of its recent requests

    def allow(self, key: str) -> bool:
        """Count one request from this caller. False if it is over the limit."""
        with self.lock:
            now = self.clock()
            if len(self.seen) >= MAX_KEYS:  # forget callers with nothing inside the window
                self.seen = {k: q for k, q in self.seen.items() if q and q[-1] > now - self.window}
            if key not in self.seen and len(self.seen) >= MAX_KEYS:
                return False  # the table is full of active callers
            times = self.seen.setdefault(key, deque())
            while times and times[0] <= now - self.window:
                times.popleft()
            if len(times) >= self.limit:
                return False
            times.append(now)
            return True
