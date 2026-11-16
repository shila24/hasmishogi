"""Small command-line score calculator for a concealed hand before its win."""
import argparse
from dataclasses import asdict
import json

from .scoring import score_hand
from .tiles import parse_tiles, to_counts
from .yaku import WinContext


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    score = commands.add_parser('score')
    score.add_argument('hand', help='13 concealed tiles BEFORE the winning tile')
    score.add_argument('--win', required=True, help='winning tile appended to the hand')
    score.add_argument('--tsumo', action='store_true')
    score.add_argument('--riichi', action='store_true')
    score.add_argument('--seat', choices=('east', 'south', 'west', 'north'), default='south')
    score.add_argument('--round', choices=('east', 'south', 'west', 'north'), default='east')
    score.add_argument('--dora', default='', help='visible indicator tile codes')
    score.add_argument('--ura', default='', help='ura indicator tile codes')
    score.add_argument('--honba', type=int, default=0)
    score.add_argument('--sticks', type=int, default=0)
    args = parser.parse_args(argv)
    try:
        before = parse_tiles(args.hand)
        win = parse_tiles(args.win)
        if len(before) != 13 or len(win) != 1:
            raise ValueError('provide 13 hand tiles and exactly one winning tile')
        complete = parse_tiles(args.hand + args.win)
        winds = ('east', 'south', 'west', 'north')
        context = WinContext(tsumo=args.tsumo, riichi=args.riichi,
                             seat_wind=27 + winds.index(args.seat), round_wind=27 + winds.index(args.round),
                             dora_indicators=tuple(t // 4 for t in parse_tiles(args.dora)),
                             ura_indicators=tuple(t // 4 for t in parse_tiles(args.ura)))
        result = score_hand(to_counts(complete), win[0] // 4, context=context, tile_ids=complete,
                            honba=args.honba, riichi_sticks=args.sticks)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(asdict(result), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
