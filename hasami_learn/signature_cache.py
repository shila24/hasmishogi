"""Cached ``inspect.signature(...).bind`` checks for callables invoked once per decision (~30 us each uncached).

Entries are keyed weakly: plain functions by the function object, bound methods by owner class and function, plus a
caller-chosen call pattern (one function may be called with different argument shapes). Closures created per game
(e.g. an AlphaZero ``actor_fn()``) are therefore never kept alive by the cache. Other callables (partials, instances
with ``__call__``) carry per-object signatures and are inspected on every call.
"""

from __future__ import annotations

import inspect
import weakref

_FUNCTIONS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()  # function -> {pattern: bool}
_METHODS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()  # class -> WeakKeyDictionary(function -> {...})


def _slot(target) -> dict | None:
    if inspect.ismethod(target):
        functions = _METHODS.get(type(target.__self__))
        if functions is None:
            functions = _METHODS[type(target.__self__)] = weakref.WeakKeyDictionary()
        return functions.setdefault(target.__func__, {})
    if inspect.isfunction(target):
        return _FUNCTIONS.setdefault(target, {})
    return None


def accepts(target, pattern: str, args: tuple, kwargs: dict) -> bool:
    """Whether ``target(*args, **kwargs)`` binds; cached per (function, owner class, pattern) where possible."""
    slot = _slot(target)
    if slot is not None and pattern in slot:
        return slot[pattern]
    try:
        inspect.signature(target).bind(*args, **kwargs)
        accepted = True
    except TypeError:
        accepted = False
    if slot is not None:
        slot[pattern] = accepted
    return accepted


def cached_entries() -> int:
    """Number of cached (function, owner, pattern) answers; for tests."""
    return (sum(len(slot) for slot in _FUNCTIONS.values())
            + sum(len(slot) for functions in _METHODS.values() for slot in functions.values()))
