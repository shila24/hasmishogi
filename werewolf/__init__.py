"""Standard-library-only five-player werewolf village."""

from .game import Outcome, Rules, WerewolfGame, create_game, replay

__all__ = ['Outcome', 'Rules', 'WerewolfGame', 'create_game', 'replay']
