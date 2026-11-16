"""Seeded four-player hand and match state machine for riichi-4p-v1.

Players retain their initial seat identity; dealer advances modulo four.
A match ends after East/South (East for tonpuu), including dealer repeats;
there is no unspecified West extension. Agari-yame requires the last dealer
to win while first and at or above return points. Draws never invoke it.
All three other seats submit a claim or pass simultaneously, keeping actor
membership independent of their concealed hands. Chance resolves internally.
Nagashi is off by default for the milestone; explicit rules can enable it.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from functools import lru_cache
import random
import secrets
from typing import Iterable

from .actions import (CHI_HIGH, CHI_LOW, CHI_MID, KAN, KAN_BASE,
                      NINE_TERMINALS, PASS, PON, RIICHI, RON, TSUMO)
from .agari import Meld, decompose
from .rules import RiichiRules
from .scoring import (NoYakuError, payments, placement_points, score_hand,
                      settle_payment, tenpai_payments)
from .shanten import shanten, waits
from .tiles import ORPHANS, RED_IDS, build_wall, to_counts
from .yaku import WinContext, nagashi_eligible


@lru_cache(maxsize=65536)
def _waits(counts: tuple[int, ...], melds: tuple[Meld, ...]) -> tuple[int, ...]:
    return waits(counts, melds)


@dataclass(frozen=True)
class Outcome:
    terminal: bool = False
    winner: int = -2
    reason: str = 'ongoing'


@dataclass(frozen=True)
class Position:
    turn: int


@dataclass(frozen=True)
class Discard:
    tile: int
    tsumogiri: bool
    riichi: bool = False
    called: bool = False


@dataclass(frozen=True)
class DeclaredMeld:
    tiles: tuple[int, ...]
    kind: str
    open: bool
    from_player: int
    called_tile: int | None = None

    def analysis(self) -> Meld:
        return Meld(tuple(t // 4 for t in self.tiles), self.kind, self.open)


class RiichiGame:
    """Full state, including hidden wall, furiten history, and joint action log."""

    def __init__(self, seed: int | None = None, rules: RiichiRules | None = None):
        if seed is not None and type(seed) is not int:
            raise ValueError('seed must be an integer or None')
        self.seed = secrets.randbits(63) if seed is None else seed
        self.rules = rules if rules is not None else RiichiRules(nagashi_mangan=False)
        if not isinstance(self.rules, RiichiRules):
            raise ValueError('rules must be RiichiRules')
        # These optional core flags need historical contracts beyond this milestone.
        if self.rules.pao or not self.rules.atozuke:
            raise ValueError('the match engine requires pao=False and atozuke=True')
        self.rng = random.Random(self.seed)
        self.initial = {}
        self.moves: list[dict[int, int]] = []
        self.scores = [self.rules.start_points] * 4
        self.dealer = 0
        self.round_index = 0
        self.honba = 0
        self.riichi_sticks = 0
        self.hand_number = 0
        self.outcome = Outcome()
        self.last_hand: dict = {}
        self.hand_start = 0  # index into ``moves`` where the current hand began
        self.final_points: tuple[float, ...] | None = None
        self._legal_cache: dict[int, tuple[int, ...]] = {}
        self._start_hand()

    @property
    def position(self) -> Position:
        return Position(self.turn)

    @property
    def history(self) -> list[dict[int, int]]:
        return [dict(step) for step in self.moves]

    @property
    def remaining_wall(self) -> int:
        return len(self.live_wall)

    @property
    def dora_indicators(self) -> tuple[int, ...]:
        return tuple(self.dead_wall[4:4 + 2 * self.dora_count:2])

    def analysis_melds(self, player: int) -> tuple[Meld, ...]:
        return tuple(m.analysis() for m in self.melds[player])

    def closed(self, player: int) -> bool:
        return not any(m.open for m in self.melds[player])

    def _start_hand(self) -> None:
        wall = build_wall(rng=self.rng)
        self.live_wall = list(wall.live)
        self.dead_wall = list(wall.dead)
        self.hands = [[] for _ in range(4)]
        for _ in range(13):
            for offset in range(4):
                self.hands[(self.dealer + offset) % 4].append(self.live_wall.pop(0))
        self.melds: list[list[DeclaredMeld]] = [[] for _ in range(4)]
        self.discards: list[list[Discard]] = [[] for _ in range(4)]
        self.riichi = [False] * 4
        self.double_riichi = [False] * 4
        self.ippatsu = [False] * 4
        self.temporary_furiten = [False] * 4
        self.riichi_furiten = [False] * 4
        self.drawn_tiles: list[int | None] = [None] * 4
        self.draw_counts = [0] * 4
        self.uninterrupted = True
        self.kan_count = 0
        self.dora_count = 1
        self.pending_dora = 0
        self.pending_riichi: int | None = None
        self.pending_kan: tuple[int, str, int] | None = None
        self.pending_four_kans = False
        self.forbidden_discards: set[int] = set()
        self.last_discard: tuple[int, int] | None = None
        self.draw_kind = 'live'
        self.hand_number += 1
        self._draw(self.dealer)

    def _draw(self, player: int, *, rinshan: bool = False) -> None:
        if not self.live_wall:
            self._exhaustive_draw()
            return
        if rinshan:
            index = self.kan_count - 1
            tile = self.dead_wall[index]
            self.dead_wall[index] = self.live_wall.pop()
        else:
            tile = self.live_wall.pop(0)
        self.hands[player].append(tile)
        self.hands[player].sort()
        self.drawn_tiles[player] = tile
        self.draw_counts[player] += 1
        self.temporary_furiten[player] = False
        self.turn = player
        self.phase = 'draw'
        self.draw_kind = 'rinshan' if rinshan else 'live'
        self.forbidden_discards = set()
        self._legal_cache.clear()

    @staticmethod
    def _validate_player(player: int) -> None:
        if type(player) is not int or not 0 <= player < 4:
            raise ValueError('player must be an integer in 0..3')

    def actors(self) -> list[int]:
        if self.outcome.terminal:
            return []
        if self.phase in ('claim', 'kan_claim'):
            return [(self.turn + offset) % 4 for offset in (1, 2, 3)]
        return [self.turn]

    def _before_draw_counts(self, player: int) -> tuple[int, ...]:
        hand = list(self.hands[player])
        if len(hand) == 14 - 3 * len(self.melds[player]):
            drawn = self.drawn_tiles[player]
            if drawn is None:
                return ()
            hand.remove(drawn)
        return to_counts(hand)

    def player_waits(self, player: int) -> tuple[int, ...]:
        counts = self._before_draw_counts(player)
        return _waits(counts, self.analysis_melds(player)) if counts else ()

    def furiten(self, player: int) -> tuple[bool, bool, bool]:
        discarded = {d.tile // 4 for d in self.discards[player]}
        own = self.rules.discard_furiten and bool(discarded.intersection(self.player_waits(player)))
        return (own, self.rules.temporary_furiten and self.temporary_furiten[player],
                self.rules.riichi_furiten and self.riichi_furiten[player])

    def _score(self, player: int, tile: int, *, tsumo: bool = False,
               chankan: bool = False, honba: int | None = None):
        hand = list(self.hands[player])
        if not tsumo:
            if sum(t // 4 == tile // 4 for t in hand) == 4:
                return None
            hand.append(tile)
        counts = to_counts(hand)
        melds = self.analysis_melds(player)
        if not decompose(counts, melds, tile // 4):
            return None
        if not tsumo and any(self.furiten(player)):
            return None
        first = self.uninterrupted and not self.discards[player] and not self.riichi[player]
        rinshan = tsumo and self.draw_kind == 'rinshan'
        last = not self.live_wall and not rinshan and not chankan and self.draw_kind == 'live'
        context = WinContext(
            tsumo=tsumo, seat_wind=27 + (player - self.dealer) % 4,
            round_wind=27 + self.round_index // 4,
            riichi=self.riichi[player], double_riichi=self.double_riichi[player],
            ippatsu=self.ippatsu[player], rinshan=rinshan, chankan=chankan,
            haitei=tsumo and last, houtei=not tsumo and last,
            tenhou=first and tsumo and player == self.dealer,
            chiihou=first and tsumo and player != self.dealer,
            renhou=first and not tsumo and player != self.dealer and self.draw_counts[player] == 0,
            dora_indicators=tuple(t // 4 for t in self.dora_indicators),
            ura_indicators=tuple(t // 4 for t in self.dead_wall[5:5 + 2 * self.dora_count:2]),
        )
        ids = hand + [t for m in self.melds[player] for t in m.tiles]
        try:
            return score_hand(counts, tile // 4, melds, context=context,
                              rules=self.rules, honba=self.honba if honba is None else honba,
                              tile_ids=ids)
        except NoYakuError:
            return None

    def _kan_options(self, player: int) -> dict[int, str]:
        if self.kan_count >= 4 or not self.live_wall or self.phase != 'draw':
            return {}
        counts = to_counts(self.hands[player])
        options = {t: 'closed' for t, n in enumerate(counts) if n == 4}
        if not self.riichi[player]:
            for meld in self.melds[player]:
                t = meld.tiles[0] // 4
                if meld.kind == 'pon' and counts[t]:
                    options[t] = 'added'
            return options
        drawn = self.drawn_tiles[player]
        before = self._before_draw_counts(player)
        old_melds = self.analysis_melds(player)
        old_waits = _waits(before, old_melds)
        allowed = {}
        for t in options:
            if drawn is None or drawn // 4 != t:
                continue
            after = list(counts)
            after[t] -= 4
            new_melds = old_melds + (Meld((t,) * 4, 'kan', False),)
            if not old_waits or _waits(tuple(after), new_melds) != old_waits:
                continue
            allowed[t] = 'closed'
        return allowed

    def _chi_tiles(self, action: int, tile: int) -> tuple[int, int, int]:
        low = tile - (action - CHI_LOW)
        return (low, low + 1, low + 2)

    def _call_forbidden(self, action: int, tile: int) -> set[int]:
        forbidden = {tile}
        if action == CHI_LOW and tile % 9 < 6:
            forbidden.add(tile + 3)
        if action == CHI_HIGH and tile % 9 > 2:
            forbidden.add(tile - 3)
        return forbidden

    def legal_actions(self, player: int | None = None) -> list[int]:
        if player is None:
            actors = self.actors()
            if len(actors) != 1:
                raise ValueError('specify player for simultaneous or terminal decisions')
            player = actors[0]
        self._validate_player(player)
        if player not in self.actors():
            return []
        if player in self._legal_cache:
            return list(self._legal_cache[player])
        counts = to_counts(self.hands[player])
        melds = self.analysis_melds(player)
        if self.phase in ('claim', 'kan_claim'):
            actions = [PASS]
            tile_id = (self.last_discard[1] if self.phase == 'claim' else
                       self._kan_tile(self.turn, self.pending_kan[2]))
            tile = tile_id // 4
            score = self._score(player, tile_id, chankan=self.phase == 'kan_claim')
            if score is not None:
                if self.phase != 'kan_claim' or self.pending_kan[1] != 'closed' or score.decomposition.kind == 'kokushi':
                    actions.append(RON)
            if self.phase == 'claim' and not self.riichi[player] and self.live_wall:
                if counts[tile] >= 2 and any(n and t != tile for t, n in enumerate(counts)):
                    actions.append(PON)
                if counts[tile] == 3 and self.kan_count < 4:
                    actions.append(KAN)
                if player == (self.turn + 1) % 4 and tile < 27:
                    for action in (CHI_LOW, CHI_MID, CHI_HIGH):
                        sequence = self._chi_tiles(action, tile)
                        if sequence[0] < 0 or sequence[-1] >= 27 or sequence[0] // 9 != sequence[-1] // 9:
                            continue
                        required = [t for t in sequence if t != tile]
                        if all(counts[t] for t in required):
                            after = list(counts)
                            for t in required:
                                after[t] -= 1
                            forbidden = self._call_forbidden(action, tile)
                            if any(n and t not in forbidden for t, n in enumerate(after)):
                                actions.append(action)
        else:
            drawn = self.drawn_tiles[player]
            actions = ([drawn // 4] if self.riichi[player] else
                       [t for t, n in enumerate(counts) if n and t not in self.forbidden_discards])
            if self.phase == 'draw':
                if self._score(player, drawn, tsumo=True) is not None:
                    actions.append(TSUMO)
                if (not self.riichi[player] and self.closed(player) and self.remaining_wall >= 4
                        and self.scores[player] >= self.rules.riichi_stick and shanten(counts, melds) <= 0):
                    for t in tuple(actions):
                        if t >= 34:
                            continue
                        after = list(counts)
                        after[t] -= 1
                        if _waits(tuple(after), melds):
                            actions.append(RIICHI + t)
                kans = self._kan_options(player)
                actions.extend([KAN] if len(kans) == 1 else [KAN_BASE + t for t in kans])
                if (self.rules.kyuushu_kyuuhai and self.uninterrupted and not self.discards[player]
                        and len({t for t, n in enumerate(counts) if n and t in ORPHANS}) >= 9):
                    actions.append(NINE_TERMINALS)
        result = tuple(sorted(actions))
        self._legal_cache[player] = result
        return list(result)

    def _take(self, player: int, tile: int, count: int = 1) -> tuple[int, ...]:
        choices = sorted((t for t in self.hands[player] if t // 4 == tile),
                         key=lambda t: (self.rules.red_fives and t in RED_IDS, t))[:count]
        if len(choices) != count:
            raise ValueError('insufficient tiles')
        for t in choices:
            self.hands[player].remove(t)
        return tuple(choices)

    def _kan_tile(self, player: int, tile: int) -> int:
        drawn = self.drawn_tiles[player]
        if drawn is not None and drawn // 4 == tile:
            return drawn
        return min(t for t in self.hands[player] if t // 4 == tile)

    def play_joint(self, actions: dict[int, int]) -> Outcome:
        if (not isinstance(actions, dict) or self.outcome.terminal
                or any(type(p) is not int for p in actions) or set(actions) != set(self.actors())):
            raise ValueError('joint actions must match the current actors')
        for player, action in actions.items():
            if type(action) is not int or action not in self.legal_actions(player):
                raise ValueError(f'illegal action for player {player}: {action!r}')
        actions = dict(actions)
        hand_before = self.hand_number
        if self.phase in ('claim', 'kan_claim'):
            self._resolve_claims(actions)
        else:
            player = self.turn
            action = actions[player]
            if action < 68:
                self._discard(player, action % 34, riichi=action >= RIICHI)
            elif action == TSUMO:
                self._win([player], self.drawn_tiles[player], tsumo=True)
            elif action == NINE_TERMINALS:
                self._finish_hand('kyuushu_kyuuhai', repeat=True, abort=True)
            else:
                options = self._kan_options(player)
                tile = next(iter(options)) if action == KAN else action - KAN_BASE
                self.pending_kan = (player, options[tile], tile)
                self.phase = 'kan_claim'
                self.uninterrupted = False
        self.moves.append(actions)
        if self.hand_number != hand_before:
            self.hand_start = len(self.moves)
        self._legal_cache.clear()
        return self.outcome

    def play(self, action: int) -> Outcome:
        actors = self.actors()
        if len(actors) != 1:
            raise ValueError('play_joint is required for simultaneous decisions')
        return self.play_joint({actors[0]: action})

    def _discard(self, player: int, tile: int, *, riichi: bool) -> None:
        drawn = self.drawn_tiles[player]
        if drawn is not None and drawn // 4 == tile:
            physical = drawn
            self.hands[player].remove(physical)
        else:
            physical = self._take(player, tile)[0]
        if self.riichi[player]:
            self.ippatsu[player] = False
        self.discards[player].append(Discard(physical, self.rules.track_tsumogiri and physical == drawn, riichi))
        self.drawn_tiles[player] = None
        self.last_discard = (player, physical)
        self.pending_riichi = player if riichi else None
        self.phase = 'claim'
        # Open-kan dora enters the scoring context only after this discard.
        self.dora_count += self.pending_dora
        self.pending_dora = 0

    def _accept_riichi(self) -> None:
        player = self.pending_riichi
        if player is not None:
            self.scores[player] -= self.rules.riichi_stick
            self.riichi_sticks += 1
            self.riichi[player] = True
            self.double_riichi[player] = self.uninterrupted and len(self.discards[player]) == 1
            self.ippatsu[player] = self.rules.ippatsu
            self.pending_riichi = None

    def _resolve_claims(self, actions: dict[int, int]) -> None:
        source = self.turn
        winners = [p for p in self.actors() if actions[p] == RON]
        for p, action in actions.items():
            # Passed discards cause furiten even when the complete shape has no yaku.
            if action != RON and (
                self.last_discard[1] // 4 in self.player_waits(p) if self.phase == 'claim'
                else RON in self.legal_actions(p)
            ):
                self.temporary_furiten[p] = True
                if self.riichi[p]:
                    self.riichi_furiten[p] = True
        if winners:
            if len(winners) == 3 and self.rules.triple_ron_abort:
                self._accept_riichi()
                self._finish_hand('triple_ron', repeat=True, abort=True)
                return
            if self.rules.atama_hane or not self.rules.double_ron:
                winners = winners[:1]
            chankan = self.phase == 'kan_claim'
            tile = self._kan_tile(source, self.pending_kan[2]) if chankan else self.last_discard[1]
            self._win(winners, tile, chankan=chankan)
            return
        if self.phase == 'kan_claim':
            player, kind, tile = self.pending_kan
            self._complete_kan(player, kind, tile)
            return
        self._accept_riichi()
        if self.rules.suucha_riichi and all(self.riichi):
            self._finish_hand('suucha_riichi', repeat=True, abort=True)
            return
        if self.rules.suufon_renda and self.uninterrupted and all(len(d) == 1 for d in self.discards):
            first = [d[0].tile // 4 for d in self.discards]
            if len(set(first)) == 1 and 27 <= first[0] <= 30:
                self._finish_hand('suufon_renda', repeat=True, abort=True)
                return
        if self.pending_four_kans:
            self._finish_hand('suukaikan', repeat=True, abort=True)
            return
        callers = [p for p in self.actors() if actions[p] in (PON, KAN)]
        if not callers:
            callers = [p for p in self.actors() if actions[p] in (CHI_LOW, CHI_MID, CHI_HIGH)]
        if callers:
            self._call(callers[0], actions[callers[0]])
        else:
            self._draw((source + 1) % 4)

    def _call(self, player: int, action: int) -> None:
        source, physical = self.last_discard
        tile = physical // 4
        self.discards[source][-1] = replace(self.discards[source][-1], called=True)
        self.ippatsu = [False] * 4
        self.uninterrupted = False
        if action == KAN:
            owned = self._take(player, tile, 3)
            self.melds[player].append(DeclaredMeld(owned + (physical,), 'kan', True, source, physical))
            self._after_kan(player, opened=True)
            return
        if action == PON:
            owned = self._take(player, tile, 2)
        else:
            owned = tuple(self._take(player, t)[0] for t in self._chi_tiles(action, tile) if t != tile)
        self.melds[player].append(DeclaredMeld(owned + (physical,), 'pon' if action == PON else 'chi', True, source, physical))
        self.turn = player
        self.phase = 'discard'
        self.drawn_tiles[player] = None
        self.forbidden_discards = self._call_forbidden(action, tile)

    def _complete_kan(self, player: int, kind: str, tile: int) -> None:
        if kind == 'closed':
            owned = self._take(player, tile, 4)
            self.melds[player].append(DeclaredMeld(owned, 'kan', False, player))
        else:
            physical = self._kan_tile(player, tile)
            self.hands[player].remove(physical)
            index = next(i for i, m in enumerate(self.melds[player]) if m.kind == 'pon' and m.tiles[0] // 4 == tile)
            meld = self.melds[player][index]
            self.melds[player][index] = replace(meld, tiles=meld.tiles + (physical,), kind='kan')
        self.pending_kan = None
        self._after_kan(player, opened=kind == 'added')

    def _after_kan(self, player: int, *, opened: bool) -> None:
        # Robbery can combine with ippatsu; a surviving kan cancels it before rinshan.
        self.ippatsu = [False] * 4
        self.kan_count += 1
        if self.rules.kan_dora:
            immediate = not self.rules.open_kan_dora_after_discard if opened else self.rules.closed_kan_dora_immediate
            if immediate:
                self.dora_count += 1
            else:
                self.pending_dora += 1
        owners = {p for p in range(4) if any(m.kind == 'kan' for m in self.melds[p])}
        self.pending_four_kans = self.rules.suukaikan and self.kan_count == 4 and len(owners) > 1
        self._draw(player, rinshan=True)

    def _win(self, winners: list[int], tile: int, *, tsumo: bool = False, chankan: bool = False) -> None:
        # Winners are in claim order; only the nearest receives honba and table sticks.
        results = [self._score(p, tile, tsumo=tsumo, chankan=chankan,
                               honba=self.honba if index == 0 else 0)
                   for index, p in enumerate(winners)]
        if any(score is None for score in results):
            raise ValueError('win no longer eligible')
        for player, score in zip(winners, results):
            delta = settle_payment(score.payment, winner=player, dealer=self.dealer,
                                   discarder=None if tsumo else self.turn)
            self.scores = [s + d for s, d in zip(self.scores, delta)]
        self.scores[winners[0]] += self.riichi_sticks * self.rules.riichi_stick
        self.riichi_sticks = 0
        self.pending_riichi = None
        self._finish_hand('tsumo' if tsumo else 'chankan' if chankan else 'ron',
                          repeat=self.dealer in winners, winners=winners,
                          winning_tile=tile, discarder=None if tsumo else self.turn)

    def _exhaustive_draw(self) -> None:
        if self.rules.nagashi_mangan:
            winners = [p for p in range(4) if nagashi_eligible(
                [d.tile // 4 for d in self.discards[p]],
                any_discard_called=any(d.called for d in self.discards[p]),
                has_open_meld=not self.closed(p), rules=self.rules)]
            if winners:
                for p in winners:
                    pay = payments(2000, dealer=p == self.dealer, tsumo=True, honba=self.honba, rules=self.rules)
                    delta = settle_payment(pay, winner=p, dealer=self.dealer)
                    self.scores = [s + d for s, d in zip(self.scores, delta)]
                self.scores[winners[0]] += self.riichi_sticks * self.rules.riichi_stick
                self.riichi_sticks = 0
                self._finish_hand('nagashi_mangan', repeat=self.dealer in winners, winners=winners)
                return
        tenpai = [bool(_waits(to_counts(self.hands[p]), self.analysis_melds(p))) for p in range(4)]
        self.scores = [s + d for s, d in zip(self.scores, tenpai_payments(tenpai, self.rules))]
        self._finish_hand('exhaustive_draw', repeat=tenpai[self.dealer], tenpai=tenpai)

    def ranking(self) -> list[int]:
        return sorted(range(4), key=lambda p: (-self.scores[p], p))

    def _reveal(self) -> dict:
        """Per-seat hidden state at hand end (fixed seat order), the supervision for belief heads.

        ``waits`` and ``tenpai`` describe the 13-tile hand before the last draw (structural waits,
        ignoring furiten and yaku); ``hand_counts`` and ``hands`` are the concealed tiles as held.
        """
        waits = []
        for p in range(4):
            counts = self._before_draw_counts(p)
            tiles = _waits(counts, self.analysis_melds(p)) if counts else ()
            waits.append([int(t in tiles) for t in range(34)])
        return dict(
            tenpai=[any(row) for row in waits], waits=waits,
            hand_counts=[list(to_counts(self.hands[p])) for p in range(4)],
            hands=[sorted(self.hands[p]) for p in range(4)],
            melds=[[dict(tiles=list(m.tiles), kind=m.kind, open=m.open, from_player=m.from_player)
                    for m in self.melds[p]] for p in range(4)],
            riichi=list(self.riichi), dora_indicators=list(self.dora_indicators),
        )

    def _finish_hand(self, reason: str, *, repeat: bool, abort: bool = False,
                     winners: Iterable[int] = (), tenpai: Iterable[bool] = (),
                     winning_tile: int | None = None, discarder: int | None = None) -> None:
        winners = list(winners)
        self.last_hand = dict(reason=reason, winners=winners, scores=list(self.scores),
                              dealer=self.dealer, round_index=self.round_index,
                              honba=self.honba, riichi_sticks=self.riichi_sticks,
                              hand_number=self.hand_number, winning_tile=winning_tile,
                              discarder=discarder, **self._reveal())
        if tenpai:
            # Exhaustive draws report the tenpai declarations that were paid.
            self.last_hand['tenpai'] = list(tenpai)
        limit = 4 if self.rules.rounds == 'tonpuu' else 8
        end = None
        if self.rules.tobi and min(self.scores) < 0:
            end = 'tobi'
        elif (self.rules.agari_yame and self.round_index == limit - 1 and self.dealer in winners
              and self.ranking()[0] == self.dealer and self.scores[self.dealer] >= self.rules.return_points):
            end = 'agari_yame'
        self.honba = self.honba + 1 if repeat or not winners else 0
        if not repeat and not abort:
            self.round_index += 1
            self.dealer = (self.dealer + 1) % 4
        if self.round_index >= limit:
            end = end or 'rounds_complete'
        if end:
            self.scores[self.ranking()[0]] += self.riichi_sticks * self.rules.riichi_stick
            self.riichi_sticks = 0
            self.final_points = placement_points(self.scores, self.rules)
            self.outcome = Outcome(True, self.ranking()[0], end)
            self.phase = 'terminal'
            self.pending_riichi = None
            self.pending_kan = None
            self._legal_cache.clear()
        # Standings after this hand's payments (and the final stick award at game end).
        self.last_hand['placement'] = [self.ranking().index(p) + 1 for p in range(4)]
        self.last_hand['scores'] = list(self.scores)
        if not end:
            self._start_hand()

    def observation(self, player: int) -> dict:
        """Return fresh public information plus only this player's private tiles."""
        self._validate_player(player)
        order = [(player + offset) % 4 for offset in range(4)]
        counts = to_counts(self.hands[player])
        return dict(
            player=player, hand=list(self.hands[player]), hand_counts=list(counts),
            drawn_tile=self.drawn_tiles[player],
            discards=[[dict(tile=d.tile, tsumogiri=d.tsumogiri, riichi=d.riichi, called=d.called)
                       for d in self.discards[p]] for p in order],
            melds=[[dict(tiles=list(m.tiles), kind=m.kind, open=m.open,
                         from_player=(m.from_player - player) % 4, called_tile=m.called_tile)
                    for m in self.melds[p]] for p in order],
            riichi=[self.riichi[p] for p in order],
            double_riichi=[self.double_riichi[p] for p in order],
            ippatsu=[self.ippatsu[p] for p in order],
            dora_indicators=list(self.dora_indicators), red_fives=self.rules.red_fives,
            remaining_wall=self.remaining_wall, scores=[self.scores[p] for p in order],
            placement=[self.ranking().index(p) + 1 for p in order],
            round_wind=27 + min(self.round_index // 4, 3),
            seat_winds=[27 + (p - self.dealer) % 4 for p in order],
            round_index=self.round_index, dealer=(self.dealer - player) % 4,
            honba=self.honba, riichi_sticks=self.riichi_sticks,
            furiten=list(self.furiten(player)),
            shanten=shanten(counts, self.analysis_melds(player)), waits=list(self.player_waits(player)),
            turn=(self.turn - player) % 4, phase=self.phase,
            last_discard=None if self.last_discard is None else
                ((self.last_discard[0] - player) % 4, self.last_discard[1]),
            pending_kan=None if self.pending_kan is None else
                ((self.pending_kan[0] - player) % 4, self.pending_kan[1], self.pending_kan[2]),
            legal_actions=self.legal_actions(player), terminal=self.outcome.terminal,
        )

    def redeal(self, hands: Iterable[Iterable[int]], live_wall: Iterable[int], dead_wall: Iterable[int],
               drawn_tiles: Iterable[int | None], temporary_furiten: Iterable[bool],
               riichi_furiten: Iterable[bool], seed: int) -> None:
        """Replace the hidden state with another arrangement consistent with the public state.

        For determinised search on an owned clone. The hidden tiles (all hands, live wall and dead
        wall) may only be permuted among those locations: hand sizes, live wall length, revealed dora
        indicators, and which seats hold a drawn tile are public and must not change. A drawn tile
        must be in its seat's hand. Furiten flags are replaced for every seat and the wall RNG is
        reseeded for later hands, so ``replay(seed, moves)`` no longer reproduces the game.
        Raises ValueError and leaves the game unchanged on any inconsistency.
        """
        if self.outcome.terminal:
            raise ValueError('cannot redeal a terminal game')
        if type(seed) is not int or not 0 <= seed < 2 ** 64:
            raise ValueError('redeal seed must be an integer in [0, 2**64)')
        hands = [list(hand) for hand in hands]
        live, dead = list(live_wall), list(dead_wall)
        drawn = list(drawn_tiles)
        flags = [list(temporary_furiten), list(riichi_furiten)]
        if len(hands) != 4 or len(drawn) != 4 or any(len(f) != 4 for f in flags):
            raise ValueError('redeal needs four hands, drawn tiles and furiten flags')
        if any(type(v) is not bool for f in flags for v in f):
            raise ValueError('furiten flags must be booleans')
        tiles = [t for hand in hands for t in hand] + live + dead
        if any(type(t) is not int for t in tiles):
            raise ValueError('tile ids must be integers')
        old = [t for hand in self.hands for t in hand] + self.live_wall + self.dead_wall
        if sorted(tiles) != sorted(old):
            raise ValueError('redeal must permute exactly the hidden tiles')
        if [len(h) for h in hands] != [len(h) for h in self.hands] or len(live) != len(self.live_wall) \
                or len(dead) != len(self.dead_wall):
            raise ValueError('hand sizes and wall lengths are public and must not change')
        if dead[4:4 + 2 * self.dora_count:2] != self.dead_wall[4:4 + 2 * self.dora_count:2]:
            raise ValueError('revealed dora indicators must not change')
        for p in range(4):
            if (drawn[p] is None) != (self.drawn_tiles[p] is None):
                raise ValueError('which seats hold a drawn tile is public')
            if drawn[p] is not None and (type(drawn[p]) is not int or drawn[p] not in hands[p]):
                raise ValueError('a drawn tile must be in its seat hand')
        self.hands = [sorted(hand) for hand in hands]
        self.live_wall, self.dead_wall, self.drawn_tiles = live, dead, drawn
        self.temporary_furiten, self.riichi_furiten = flags
        self.seed = seed
        self.rng = random.Random(seed)
        self._legal_cache.clear()

    def clone(self) -> RiichiGame:
        return copy.deepcopy(self)

    def close(self) -> None:
        """No external handles are owned by this engine."""

    @classmethod
    def replay(cls, seed: int, actions: Iterable[dict[int, int]], rules: RiichiRules | None = None) -> RiichiGame:
        game = cls(seed=seed, rules=rules)
        for joint in actions:
            game.play_joint(joint)
        return game


Game = RiichiGame
