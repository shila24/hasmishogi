"""Command-line inspection, perft and seeded random-play tools."""
import argparse
import random

from . import create_game
from .model import Rules
from .notation import move_to_usi, usi_to_move
from .perft import perft, divide


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    count = commands.add_parser("perft")
    count.add_argument("--depth", required=True, type=int)
    count.add_argument("--sfen")
    count.add_argument("--divide", action="store_true")
    inspect = commands.add_parser("sfen")
    inspect.add_argument("--sfen")
    inspect.add_argument("--moves", nargs="*", default=[])
    play = commands.add_parser("play-random")
    play.add_argument("--sfen")
    play.add_argument("--seed", type=int, default=0)
    play.add_argument("--max-moves", type=int, default=320)
    args = parser.parse_args(argv)
    try:
        rules = Rules(max_moves=args.max_moves) if args.command == "play-random" else Rules()
        game = create_game(rules=rules, sfen=args.sfen)
        if args.command == "perft":
            if args.divide:
                counts = divide(game, args.depth)
                for code, value in counts.items():
                    print(f"{move_to_usi(code)}: {value}")
                print(f"total: {sum(counts.values())}")
            else:
                print(perft(game, args.depth))
        elif args.command == "sfen":
            for text in args.moves:
                game.play(usi_to_move(game, text))
            print(game.sfen())
        else:
            rng = random.Random(args.seed)
            while not game.outcome.terminal:
                code = rng.choice(game.legal_actions())
                print(move_to_usi(code))
                game.play(code)
            print(game.sfen())
            print(f"winner={game.outcome.winner} reason={game.outcome.reason}")
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
