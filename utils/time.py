"""
Time utilitiles
"""
from functools import wraps
import time

def timeit(verbose=False):
	def _timeit(f):
		"""Timer"""
		@wraps(f)
		def wrapper(*args, **kwargs):
			if verbose:
				start = time.time()
				res = f(*args, **kwargs)
				runtime = time.time() - start
				print(f'{f.__name__!r} in {runtime:.4f} s')
			else:
				res = f(*args, **kwargs)
			return res

		return wrapper
	return _timeit


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