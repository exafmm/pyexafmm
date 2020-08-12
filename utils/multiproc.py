from ctypes import c_int
from multiprocessing import Pool, Value, Lock


class SingletonDecorator:
    """
    Decorate instance to enforce Singleton
    """
    def __init__(self, klass):
        self.klass = klass
        self.instance = None

    def __call__(self, *args, **kwargs):
        if self.instance == None:
            self.instance = self.klass(*args, **kwargs)
        return self.instance


def setup_pool(processes, initializer=None, initargs=None, maxtasksperchild=None):
    """
    Singleton Pool wrapper.

    Parameters:
    -----------
    processes : int
        Number of worker processes to launch.
    initializer : None/function
        If initializer is not None then each worker process will call
        initializer(*initargs) when it starts.
    initargs : None/any
        Arguments of initializer.
    maxtasksperchild : None/int
        Maximum number of tasks a worker will complete before destroying
        itself. Default of None reuslts in process staying alive as long as
        the pool is alive.

    Returns:
    --------
    Pool
    """
    singleton = SingletonDecorator(Pool)

    pool = singleton(
            processes=processes,
            initializer=initializer,
            initargs=initargs,
            maxtasksperchild=maxtasksperchild
            )

    return pool


def setup_counter(default):
    """
    Setup a global counter.
    """
    counter = Value(c_int)
    counter.value = default
    return counter


def setup_lock():
    """
    Setup a lock.
    """
    return Lock()
