"""Utility routines."""
import time as _time

class Timer:
    """Context manager to measure times."""

    def __init__(self):
        """Constructor."""
        self.start = 0
        self.end = 0
        self.interval = 0

    def __enter__(self):
        self.start = _time.time()
        return self

    def __exit__(self, *args):
        self.end = _time.time()
        self.interval = self.end - self.start

