"""Independent score tables and winning-tile regression fixtures."""
import json
import subprocess
import sys
import unittest

from riichi.agari import Meld
from riichi.rules import RiichiRules
from riichi.tiles import parse_counts
from riichi.yaku import WinContext
from riichi.scoring import score_hand, payments, base_points, tenpai_payments, placement_points


class ScoringTests(unittest.TestCase):
    def test_classic_ron_grid(self):
        tables = {
            False: {30: (1000, 2000, 3900, 7700), 40: (1300, 2600, 5200, 8000), 50: (1600, 3200, 6400, 8000)},
            True: {30: (1500, 2900, 5800, 11600), 40: (2000, 3900, 7700, 12000), 50: (2400, 4800, 9600, 12000)},
        }
        for dealer, rows in tables.items():
            for fu, amounts in rows.items():
                for han, expected in enumerate(amounts, 1):
                    with self.subTest(dealer=dealer, fu=fu, han=han):
                        self.assertEqual(payments(base_points(han, fu), dealer=dealer).ron, expected)

    def test_classic_tsumo_grid(self):
        nondealer = {30: ((300, 500), (500, 1000), (1000, 2000), (2000, 3900)),
                     40: ((400, 700), (700, 1300), (1300, 2600), (2000, 4000)),
                     50: ((400, 800), (800, 1600), (1600, 3200), (2000, 4000))}
        dealer = {30: (500, 1000, 2000, 3900), 40: (700, 1300, 2600, 4000), 50: (800, 1600, 3200, 4000)}
        for fu, row in nondealer.items():
            for han, (low, high) in enumerate(row, 1):
                with self.subTest(fu=fu, han=han):
                    pay = payments(base_points(han, fu), tsumo=True)
                    self.assertEqual((pay.tsumo_non_dealer, pay.tsumo_dealer), (low, high))
                    self.assertEqual(payments(base_points(han, fu), tsumo=True, dealer=True).tsumo_non_dealer, dealer[fu][han - 1])

    def test_limits_and_counters(self):
        for han, fu, expected in ((3, 70, 2000), (4, 40, 2000), (5, 30, 2000), (6, 30, 3000),
                                  (8, 30, 4000), (11, 30, 6000), (13, 30, 8000), (26, 30, 8000)):
            self.assertEqual(base_points(han, fu), expected)
        self.assertEqual(base_points(4, 30, rules=RiichiRules(kiriage_mangan=True)), 2000)
        self.assertEqual(base_points(13, 30, rules=RiichiRules(kazoe_yakuman=False)), 6000)
        self.assertEqual(payments(2000, honba=2, riichi_sticks=3).total, 11600)
        pay = payments(2000, tsumo=True, honba=2, riichi_sticks=3)
        self.assertEqual((pay.tsumo_non_dealer, pay.tsumo_dealer, pay.total), (2200, 4200, 11600))

    def test_tenpai_and_placement(self):
        for flags, expected in (((True, False, False, False), (3000, -1000, -1000, -1000)),
                                ((True, True, False, False), (1500, 1500, -1500, -1500)),
                                ((True, True, True, False), (1000, 1000, 1000, -3000)),
                                ((False,) * 4, (0,) * 4), ((True,) * 4, (0,) * 4)):
            self.assertEqual(tenpai_payments(flags), expected)
        self.assertEqual(placement_points((40000, 30000, 20000, 10000)), (50.0, 10.0, -20.0, -40.0))
        self.assertEqual(placement_points((25000,) * 4), (35.0, 5.0, -15.0, -25.0))

    def test_fu_wait_assignment(self):
        hand = parse_counts('123456m23455p678s')
        for win, tsumo, fu, pinfu in ((0, True, 20, True), (0, False, 30, True), (2, True, 30, False), (2, False, 40, False)):
            result = score_hand(hand, win, context=WinContext(tsumo=tsumo, riichi=True))
            self.assertEqual(result.fu, fu)
            self.assertEqual('pinfu' in {y.name for y in result.yaku}, pinfu)

    def test_cli(self):
        result = subprocess.run([sys.executable, '-m', 'riichi.cli', 'score', '123m456p789s1122z', '--win', '2z', '--tsumo', '--riichi'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual((data['han'], data['fu']), (3, 40))
        self.assertEqual(data['payment']['total'], 5200)

    def test_fu_meld_table(self):
        cases = [
            ('222m', 'pon', True, '123m345p678s55z', False, 30),
            ('111m', 'pon', True, '234m345p678s55z', False, 30),
            ('2222m', 'kan', False, '456m345p678s55z', False, 50),
            ('2222m', 'kan', True, '456m345p678s55z', False, 30),
            ('1111m', 'kan', False, '234m345p678s55z', False, 70),
            ('1111m', 'kan', True, '234m345p678s55z', False, 40),
            ('1111m', 'kan', False, '234m345p678s55z', True, 60),
        ]
        from riichi.agari import decompose
        from riichi.scoring import calculate_fu
        for code, kind, opened, hand, tsumo, expected in cases:
            with self.subTest(meld=code, open=opened, tsumo=tsumo):
                counts = parse_counts(code)
                called = Meld(tuple(t for t, n in enumerate(counts) for _ in range(n)), kind, opened)
                d, = decompose(parse_counts(hand), (called,), 11)
                self.assertEqual(calculate_fu(d, WinContext(tsumo=tsumo)), expected)
        self.assertEqual(score_hand(parse_counts('11m22p334455s6677z'), 0).fu, 25)

    def test_double_wind_pair_and_high_fu(self):
        from riichi.yaku import pair_fu
        self.assertEqual(pair_fu(27, WinContext(seat_wind=27), RiichiRules()), 4)
        self.assertEqual(pair_fu(27, WinContext(seat_wind=27), RiichiRules(double_wind_pair_fu=2)), 2)
        called = (Meld((0,) * 4, 'kan', False), Meld((8,) * 4, 'kan', False), Meld((9,) * 4, 'kan', False))
        result = score_hand(parse_counts('999s55z'), 26, called)
        self.assertEqual((result.fu, result.base_points), (140, 4000))

    def test_settlement_conserves_points_and_external_sticks(self):
        from riichi.scoring import settle_payment
        for winner in range(4):
            for dealer in range(4):
                for tsumo in (True, False):
                    pay = payments(2000, dealer=winner == dealer, tsumo=tsumo, honba=2, riichi_sticks=1)
                    deltas = settle_payment(pay, winner=winner, dealer=dealer, discarder=None if tsumo else (winner + 1) % 4)
                    self.assertEqual(sum(deltas), 1000)
                    self.assertEqual(deltas[winner], pay.total)

    def test_cli_in_process_and_invalid_inputs(self):
        from contextlib import redirect_stderr, redirect_stdout
        import io
        from riichi.cli import main
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(main(['score', '123m456p789s1122z', '--win', '2z', '--riichi', '--tsumo']), 0)
        self.assertEqual(json.loads(output.getvalue())['fu'], 40)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(['score', '123m', '--win', '2z'])
