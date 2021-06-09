"""
Time utilitiles
"""
import time


class Timer:
    """Context manager to measure times."""

    def __init__(self):
        """Constructor."""
        self.start = 0
        self.end = 0
        self.interval = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start

def seconds_to_minutes(seconds):
    """
    Convert seconds to minutes and seconds.
    Parameters:
    -----------
    seconds : float
    Returns:
    --------
    (float, float)
        Minutes and seconds left as a tuple.
    """
    minutes = seconds // 60
    seconds = seconds % 60
    return minutes, seconds