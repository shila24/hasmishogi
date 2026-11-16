"""Small explicit positions and snapshot helpers for public API tests."""
from shogi import create_game
from shogi.model import Position
from shogi.notation import usi_to_square, usi_to_move


def setup(pieces, turn=0, hands=((0,) * 7, (0,) * 7), rules=None, number=1):
    board = [0] * 81
    for square, cell in pieces.items():
        board[usi_to_square(square)] = cell
    return create_game(rules=rules, sfen=Position(tuple(board), tuple(tuple(h) for h in hands), turn, number).sfen())


def play_text(game, moves):
    return [game.play(usi_to_move(game, text)) for text in moves]


def snapshot(game):
    return (game.initial, game.position, game.outcome, game.moves.copy(),
            game.history_keys(), game.legal_actions(), game.can_declare(), game.checkers())
