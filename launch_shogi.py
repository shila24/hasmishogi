"""Launch the Shogi GUI when the runtime module is available."""
import importlib.util


def main():
    try:
        available = importlib.util.find_spec("shogi.gui") is not None
    except ModuleNotFoundError:
        available = False
    if not available:
        raise SystemExit("将棋 GUI はこの版には含まれていません。")
    from shogi.gui import main as launch
    launch()


if __name__ == "__main__":
    main()
