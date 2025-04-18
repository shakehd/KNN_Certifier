"""
This type stub file was generated by pyright.
"""

import threading
from typing import Callable, Optional

_waitforthreads_lock = ...
def waitforqueues(queues: list, timeout: float = ...) -> filter:
    """Waits for one or more *Queue* to be ready or until *timeout* expires.

    *queues* is a list containing one or more *Queue.Queue* objects.
    If *timeout* is not None the function will block
    for the specified amount of seconds.

    The function returns a list containing the ready *Queues*.

    """
    ...

def prepare_queues(queues: list, lock: threading.Condition): # -> None:
    """Replaces queue._put() method in order to notify the waiting Condition."""
    ...

def wait_queues(queues: list, lock: threading.Condition, timeout: Optional[float]): # -> None:
    ...

def reset_queues(queues: list): # -> None:
    """Resets original queue._put() method."""
    ...

def waitforthreads(threads: list, timeout: float = ...) -> filter:
    """Waits for one or more *Thread* to exit or until *timeout* expires.

    .. note::

       Expired *Threads* are not joined by *waitforthreads*.

    *threads* is a list containing one or more *threading.Thread* objects.
    If *timeout* is not None the function will block
    for the specified amount of seconds.

    The function returns a list containing the ready *Threads*.

    """
    ...

def prepare_threads(new_function: Callable) -> Callable:
    """Replaces threading._get_ident() function in order to notify
    the waiting Condition."""
    ...

def wait_threads(threads: list, lock: threading.Condition, timeout: Optional[float]): # -> None:
    ...

def reset_threads(old_function: Callable): # -> None:
    """Resets original threading.get_ident() function."""
    ...

def new_method(self, *args): # -> None:
    ...

