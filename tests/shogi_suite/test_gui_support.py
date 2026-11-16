"""Display-free shogi input, presentation and optional-model contract tests."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from shogi import create_game
from shogi.model import DECLARE, Position, encode_move
from shogi.notation import usi_to_move, usi_to_square
from shogi.gui_support import (
    InputState, click_square, click_hand, choose_promotion, declaration,
    square_to_display, display_to_square, move_text, piece_name, outcome_text,
    advice_rows, level_text, discover_models, analyze, History, MoveCodeAgent,
)


class InputTests(unittest.TestCase):
    def test_selection_deselection_reselection_and_illegal_target(self):
        game = create_game()
        state = InputState()
        for sq in (0, 40, -1, 81):
            self.assertEqual(click_square(state, game, sq), (state, None, ()))
        state, action, targets = click_square(state, game, 56)
        self.assertIsNone(action)
        self.assertEqual(targets, (47,))
        self.assertEqual(click_square(state, game, 40), (state, None, targets))
        other, _, _ = click_square(state, game, 57)
        self.assertEqual(other.square, 57)
        self.assertEqual(click_square(state, game, 56), (InputState(), None, ()))
        cleared, action, targets = click_square(state, game, 47)
        self.assertEqual(action, usi_to_move(game, '7g7f'))
        self.assertEqual(cleared, InputState())
        self.assertFalse(game.moves)

    def test_promotion_optional_forced_and_cancel(self):
        game = create_game(sfen='4k4/9/4P4/9/9/9/9/9/4K4 b - 1')
        state, _, _ = click_square(InputState(), game, usi_to_square('5c'))
        pending, action, _ = click_square(state, game, usi_to_square('5b'))
        self.assertEqual(pending.mode, 'promotion')
        self.assertIsNone(action)
        self.assertEqual(click_square(pending, game, 40)[0], pending)
        for promote in (True, False):
            _, action, _ = choose_promotion(pending, promote)
            self.assertEqual(action, usi_to_move(game, '5c5b' + ('+' if promote else '')))
        self.assertEqual(choose_promotion(pending, None), (InputState(), None, ()))
        self.assertIsNone(choose_promotion(InputState(), True)[1])
        game = create_game(sfen='k8/4P4/9/9/9/9/9/9/4K4 b - 1')
        state, _, _ = click_square(InputState(), game, 13)
        state, action, _ = click_square(state, game, 4)
        self.assertEqual(action, usi_to_move(game, '5b5a+'))
        self.assertEqual(state, InputState())

    def test_drops_turn_and_terminal(self):
        game = create_game(sfen='4k4/9/9/9/9/9/4P4/9/4K4 b P 1')
        state, _, targets = click_hand(InputState(), game, 0, 0)
        self.assertEqual(state.mode, 'hand')
        self.assertNotIn(40, targets)
        self.assertNotIn(0, targets)
        self.assertEqual(click_hand(state, game, 0, 0), (InputState(), None, ()))
        for player, hand in ((1, 0), (0, 1), (0, -1), (2, 0)):
            self.assertEqual(click_hand(state, game, player, hand)[0], state)
        self.assertEqual(click_square(state, game, 41)[1], usi_to_move(game, 'P*4e'))
        self.assertIsNone(declaration(game)[1])
        game.resign()
        self.assertEqual(click_square(state, game, 41), (InputState(), None, ()))
        self.assertEqual(click_hand(state, game, 0, 0), (InputState(), None, ()))

    def test_declaration(self):
        game = create_game(sfen='4K4/PPPP1PPPP/4P+P3/9/9/9/9/9/4k4 b 2R2B 1')
        self.assertEqual(declaration(game), (InputState(), DECLARE, ()))

    def test_coordinates(self):
        for flipped in (False, True):
            for sq in range(81):
                self.assertEqual(display_to_square(*square_to_display(sq, flipped), flipped), sq)
        self.assertEqual(square_to_display(0, True), (8, 8))
        self.assertIsNone(display_to_square(-1, 0))
        self.assertIsNone(display_to_square(0, 9))


class PresentationTests(unittest.TestCase):
    def test_notation_and_names(self):
        game = create_game()
        for text, expected in [('7g7f', '▲７六歩'), ('3c3d', '△３四歩'), ('8h2b+', '▲２二角成'),
                               ('3a2b', '△同　銀')]:
            code = usi_to_move(game, text)
            self.assertEqual(move_text(game, code), expected)
            game.play(code)
        game = create_game(sfen='k8/9/4P4/9/9/9/9/9/4K4 b P 1')
        self.assertIn('不成', move_text(game, usi_to_move(game, '5c5b')))
        self.assertIn('打', move_text(game, usi_to_move(game, 'P*4e')))
        game = create_game(sfen='4k4/9/9/9/9/9/3G1G3/9/4K4 b - 1')
        self.assertIn('右', move_text(game, usi_to_move(game, '4g5f')))
        self.assertIn('左', move_text(game, usi_to_move(game, '6g5f')))
        self.assertEqual(piece_name(30), '竜')
        self.assertEqual(piece_name(24), '王')
        self.assertEqual(piece_name(10), '成香')
        game.resign()
        self.assertIn('投了', outcome_text(game.outcome))

    def test_all_disambiguation_qualifiers_in_both_orientations(self):
        cases = [((48, 50), 50, '右'), ((48, 50), 48, '左'),
                 ((48, 49), 49, '直'), ((39, 48), 48, '上'),
                 ((31, 49), 31, '引'), ((39, 48), 39, '寄')]
        for origins, source, word in cases:
            for side in (0, 1):
                board = [0] * 81
                board[4], board[76] = 24, 8
                for sq in origins:
                    board[sq] = 5
                if side:
                    board = [0 if not cell else ((cell & 15) + (0 if cell >> 4 else 16))
                             for cell in reversed(board)]
                game = create_game(sfen=Position(tuple(board), ((0,)*7, (0,)*7), side, 1).sfen())
                code = encode_move(80-source if side else source, 40)
                self.assertTrue(move_text(game, code).endswith(word), (side, origins, word, move_text(game, code)))

    def test_advice_uses_move_not_policy_label_and_preserves_history(self):
        game = create_game()
        before = game.history_keys()
        class Advisor:
            def advise(self, snapshot, level, **kwargs):
                self.snapshot = snapshot
                return [SimpleNamespace(action=0, move=snapshot.legal_actions()[0],
                                        practical_score=.625, value_loss=.1, difficulty=None)]
            def engine_best(self, snapshot, cancel=None):
                return self.advise(snapshot, 1)[0]
        advisor = Advisor()
        hints, best = analyze(advisor, game, 1)
        self.assertEqual(advice_rows(game, hints)[0][1:], ('62.5%', '0.100', '未計測'))
        self.assertEqual(game.history_keys(), before)
        self.assertIsNot(advisor.snapshot, game)
        self.assertEqual(level_text(SimpleNamespace(level=1, elo=None)), 'レベル 1（Elo 未計測）')
        self.assertIn('123', level_text(SimpleNamespace(level=2, elo=123.4)))
        advisor.advise = lambda *a, **k: [SimpleNamespace(move=True)]
        with self.assertRaises(ValueError):
            analyze(advisor, game, 1)

    def test_history_navigation_branch_and_resign(self):
        history = History(create_game())
        try:
            for text in ('7g7f', '3c3d', '2g2f'):
                history.play(usi_to_move(history.game, text))
            final = history.game.history_keys()
            history.seek(1)
            self.assertEqual(history.game.ply, 1)
            self.assertEqual(history.length, 3)
            history.seek(3)
            self.assertEqual(history.game.history_keys(), final)
            history.seek(1)
            with self.assertRaises(ValueError):
                history.play(usi_to_move(history.game, '8c8d'))
            history.play(usi_to_move(history.game, '8c8d'), branch=True)
            self.assertEqual(history.length, 2)
            history.resign()
            history.seek(0)
            history.seek(2)
            self.assertEqual(history.game.outcome.reason, 'resign')
            history.undo((True, True))
            self.assertFalse(history.game.outcome.terminal)
        finally:
            history.close()

    def test_discovery_only_installed_shogi_and_no_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'ok').mkdir()
            for file in ('manifest.json', 'weights.pt'):
                (root / 'ok' / file).write_text('{}')
            (root / 'index.json').write_text(json.dumps({'models': [
                {'name': 'ok', 'game': 'shogi'}, {'name': 'missing', 'game': 'shogi'},
                {'name': '../escape', 'game': 'shogi'}, {'name': 'ok', 'game': 'hasami'}]}))
            self.assertEqual(discover_models(root), {'ok': str((root / 'ok').resolve())})
            (root / 'index.json').write_text('[]')
            with self.assertRaises(ValueError):
                discover_models(root)
        self.assertEqual(discover_models(Path(tmp)), {})

    def test_import_has_no_tk_or_learning_dependencies(self):
        result = subprocess.run([sys.executable, '-c',
            "import shogi.gui_support,sys; assert not any(x in sys.modules for x in ('tkinter','torch','numpy','hasami_learn'))"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class LearnedActionTests(unittest.TestCase):
    def test_policy_labels_converted_for_both_sides_and_declaration(self):
        from hasami.agents import Decision, SearchLimits
        from shogi.encoding import move_label
        for sfen in (None, '4K4/PPPP1PPPP/4P+P3/9/9/9/9/9/4k4 b 2R2B 1'):
            game = create_game(sfen=sfen)
            for _ in range(2 if sfen is None else 1):
                move = DECLARE if sfen else game.legal_actions()[-1]
                label = move_label(game, move)
                agent = MoveCodeAgent(SimpleNamespace(choose=lambda *a: Decision(label, score=123)))
                result = agent.choose(game, SearchLimits())
                self.assertEqual(result.action, move)
                self.assertEqual(result.score, 123)
                game.play(move)
        bad = MoveCodeAgent(SimpleNamespace(choose=lambda *a: Decision(True)))
        with self.assertRaises(ValueError):
            bad.choose(create_game(), SearchLimits())

    def test_optional_factories_are_lazy_and_preserve_public_options(self):
        from unittest.mock import Mock, patch
        from shogi.gui_support import load_strong_agent, load_advisor
        fake = Mock()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'manifest.json').write_text(json.dumps({'game_id': 'shogi', 'network': {'head': 'q'}}))
            with patch.dict(sys.modules, {'hasami_learn.inference': SimpleNamespace(create_agent=fake)}):
                wrapped = load_strong_agent(directory)
            self.assertIsInstance(wrapped, MoveCodeAgent)
            self.assertEqual(fake.call_args.kwargs['game_id'], 'shogi')
            self.assertEqual(fake.call_args.kwargs['mode'], 'q')
            with patch.dict(sys.modules, {'hasami_learn.practical': SimpleNamespace(create_advisor=fake)}):
                load_advisor(directory)
            fake.assert_called_with(checkpoint=directory, method='search')

    def test_advisor_rejects_other_game_and_methods_require_practical_metadata(self):
        from shogi.gui_support import advisor_methods, load_advisor
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'manifest.json'
            path.write_text(json.dumps({'game_id': 'shogi', 'inference_hint': {'mode': 'mcts'}}))
            self.assertEqual(advisor_methods(directory), ('search',))
            path.write_text(json.dumps({'game_id': 'shogi', 'inference_hint': {'mode': 'practical'}}))
            self.assertEqual(advisor_methods(directory), ('search', 'learned'))
            path.write_text(json.dumps({'game_id': 'hasami'}))
            with self.assertRaises(ValueError):
                load_advisor(directory)
