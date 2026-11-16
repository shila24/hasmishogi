"""Named registries so every agent-related component can be swapped by config."""

from __future__ import annotations

from typing import Callable, Generic, Iterator, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A small name -> object registry with decorator registration."""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, T] = {}

    def register(self, name: str, item: T | None = None):
        if not isinstance(name, str) or not name:
            raise ValueError(f"{self.kind} name must be a nonempty string")

        def decorator(obj: T) -> T:
            existing = self._items.get(name)
            if existing is not None and existing is not obj:
                raise ValueError(f"duplicate {self.kind} name: {name}")
            self._items[name] = obj
            return obj

        return decorator(item) if item is not None else decorator

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError as exc:
            raise KeyError(f"unknown {self.kind} '{name}'; known: {', '.join(self.names()) or 'none'}") from exc

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(self._items)


GAMES: Registry[Callable] = Registry("game")

ALGORITHMS: Registry = Registry("algorithm")
NETWORKS: Registry[Callable] = Registry("network")
ENCODERS: Registry[Callable] = Registry("encoder")
OPPONENT_STRATEGIES: Registry[Callable] = Registry("opponent_strategy")


# Load only the built-in game package; external registrations must be explicit.
from . import games as _builtin_games  # noqa: E402, F401
