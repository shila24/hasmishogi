"""Standard-library validation of the shared release evidence contracts."""
from datetime import datetime, timezone
import hashlib
import math
from pathlib import Path
import re

from .state import LoopError, read_json, validate_schema


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_document(data, name):
    schema = Path(__file__).resolve().parents[1] / "schemas" / f"{name}.schema.json"
    validate_schema(data, read_json(schema))
    try:
        date = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
        if date.utcoffset() != timezone.utc.utcoffset(None):
            raise ValueError("not UTC")
    except (ValueError, TypeError) as exc:
        raise LoopError("created_at must be an ISO 8601 UTC date") from exc


def wilson_interval(successes, games):
    p = successes / games
    z2 = 1.96**2
    denominator = 1 + z2 / games
    center = (p + z2 / (2 * games)) / denominator
    margin = 1.96 * math.sqrt(p * (1 - p) / games + z2 / (4 * games**2)) / denominator
    return max(0, center - margin), min(1, center + margin)


def effective_sample(row, *, legacy_raw=False):
    """Return the sum of opening/seating group means and the group count.

    Old evidence remains readable with its historical raw interval. A gate
    cannot recover its groups, so it credits at most one sample for each of
    the two legacy seatings, using the recorded score rate.
    """
    extra = row.get("extra", {})
    if not isinstance(extra, dict):
        raise LoopError("evaluation extra must be an object")
    keys = {"effective_games", "effective_successes", "distinct_games", "distinct_openings", "sample_unit"}
    if not keys.intersection(extra):
        games = row["games"] if legacy_raw else min(2, row["games"])
        return (row["wins"] + 0.5 * row["draws"]) / row["games"] * games, games
    if not keys <= extra.keys() or extra["sample_unit"] != "opening_actions_and_seating":
        raise LoopError("incomplete evaluation effective sample")
    games, successes = extra["effective_games"], extra["effective_successes"]
    for name in ("effective_games", "distinct_games", "distinct_openings"):
        if type(extra[name]) is not int or not 1 <= extra[name] <= row["games"]:
            raise LoopError(f"invalid evaluation {name}")
    if not extra["distinct_openings"] <= games <= extra["distinct_games"]:
        raise LoopError("inconsistent evaluation distinct counts")
    if type(successes) not in (int, float) or not math.isfinite(successes) or not 0 <= successes <= games:
        raise LoopError("invalid evaluation effective_successes")
    return successes, games


def validate_evaluation(data):
    validate_document(data, "evaluation-evidence")
    if not re.fullmatch(r"cpu|mps|cuda(?::[0-9]+)?", data["device"]):
        raise LoopError("evaluation must record an actual device")
    for row in data["results"].values():
        if row["wins"] + row["losses"] + row["draws"] != row["games"] or row["truncated"] > row["draws"]:
            raise LoopError("invalid evaluation counts")
        low, high = wilson_interval(*effective_sample(row, legacy_raw=True))
        if not math.isclose(row["ci_low"], low, abs_tol=1e-12) or not math.isclose(row["ci_high"], high, abs_tol=1e-12):
            raise LoopError("evaluation Wilson intervals do not match counts")


def model_name(loop_id, cycle):
    """Retain a readable name and hash long identities to avoid truncation collisions."""
    name = f"{loop_id.lower()}-c{cycle}"
    if len(name) > 64:
        digest = hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]
        name = f"{name[:47]}-{digest}"
    return name
