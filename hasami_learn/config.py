"""Training configuration dataclasses validated against hasami_learn/schemas/training-config.schema.json."""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from .games import get_spec
from .search.draw import SEARCH_SEMANTICS_VERSION
from .schema_validation import SCHEMA_DIR, load_schema as _load_packaged, validate as _validate

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_NAME = "training-config.schema.json"
SCHEMA_PATH = SCHEMA_DIR / SCHEMA_NAME  # packaged copy of hasami_learn/schemas/training-config.schema.json


class ConfigError(ValueError):
    pass


class _UseDefault:
    """Sentinel for with_overrides: reset a nullable field (e.g. max_plies) to the game default."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "USE_DEFAULT"


USE_DEFAULT = _UseDefault()

GAME_LIST_DEFAULTS = (
    ("selfplay", "draw_reasons", "draw_reasons", ()),
    ("opponents", "classical", "default_baselines", ("random",)),
    ("evaluation", "opponents", "default_baselines", ("random",)),
)


@dataclass
class RunConfig:
    run_id: str = "run"
    output_dir: str = "runs"
    iterations: int = 10
    seed: int = 0
    backend: str = "auto"
    checkpoint_every: int = 1
    eval_every: int = 0
    keep_checkpoints: int = 5
    checkpoint_every_seconds: int = 600  # time-based checkpoint in addition to the iteration cadence
    notes: str = ""
    init_checkpoint: str | None = None


@dataclass
class DeviceConfig:
    device: str = "auto"
    amp: str = "auto"
    threads: int | None = None
    deterministic: bool = False
    # Rows per batched self-play forward; None = per-device default (devices.DEFAULT_INFERENCE_BATCH).
    inference_batch: int | None = None
    # Pad every batched forward to inference_batch rows: outputs independent of batch composition.
    fixed_batch_shape: bool = True

    def __post_init__(self) -> None:
        # JSON (and the schema's "integer") does not distinguish 8 from 8.0; keep an int so collection never sees a
        # float, and reject anything else at load time instead of at the first collection.
        value = self.inference_batch
        if type(value) is float and value.is_integer() and value >= 1:
            self.inference_batch = int(value)
        elif value is not None and (type(value) is not int or value < 1):
            raise ConfigError("device.inference_batch must be a positive integer or null")


@dataclass
class NamedConfig:
    name: str = ""
    params: dict[str, Any] = field(default_factory=dict)


class _TracksDefaults:
    """Track explicit assignments to game-derived caps and lists."""

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        if name in ("max_plies", "draw_reasons", "classical", "opponents"):
            object.__setattr__(self, f"_{name}_explicit", value is not None)

    def list_is_default(self, name):
        return (not getattr(self, f"_{name}_explicit", True) and
                getattr(self, name) == getattr(self, f"_{name}_default", None))

    def mark_max_plies_defaulted(self) -> None:
        object.__setattr__(self, "_max_plies_explicit", False)

    @property
    def max_plies_explicit(self) -> bool:
        return bool(getattr(self, "_max_plies_explicit", getattr(self, "max_plies", None) is not None))


@dataclass
class SelfplayConfig(_TracksDefaults):
    games_per_iteration: int = 8
    max_plies: int | None = None
    workers: int = 1
    temperature: float = 1.0
    temperature_plies: int = 20
    symmetry_augmentation: bool = True
    truncation_value: float = 0.0
    # Draw contempt: annealed penalty for listed draw reasons so early self-play does not settle on repetition.
    draw_value_start: float = -0.2
    draw_value_end: float = 0.0
    draw_value_iterations: int = 100
    draw_reasons: list[str] | None = None


@dataclass
class ReplayConfig:
    capacity: int = 50000
    batch_size: int = 128
    min_size: int = 256
    persist: bool = False


@dataclass
class OptimizerConfig:
    name: str = "adam"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    momentum: float = 0.9
    grad_clip: float = 1.0
    updates_per_iteration: int = 50
    lr_schedule: str = "constant"


@dataclass
class OpponentsConfig(_TracksDefaults):
    strategy: str = "self"
    pool_size: int = 8
    snapshot_every: int = 0
    classical: list[str] | None = None
    classical_weight: float = 0.0
    exploiter_every: int = 0
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatastoreConfig:
    """Self-play game store with rolling caps (docs/TRAINING_ja.md#resume)."""

    enabled: bool = False
    max_games: int = 10000
    max_bytes: int = 30 * 1024 ** 3
    path: str | None = None  # default runs/<run_id>/selfplay


@dataclass
class EvaluationConfig(_TracksDefaults):
    games: int = 10
    opponents: list[str] | None = None
    swap_sides: bool = True
    max_plies: int | None = None
    limits: dict[str, int] = field(default_factory=lambda: {"max_depth": 1, "time_ms": 200, "node_limit": 20000})
    simulations: int = 0
    save_records: bool = False
    opening_plies: int = 0


@dataclass
class TrainingConfig:
    schema_version: int = 1
    # Kept outside algorithm.params so explicit CLI parameter overrides round-trip unchanged.
    search_semantics_version: int = SEARCH_SEMANTICS_VERSION
    game: NamedConfig = field(default_factory=lambda: NamedConfig("hasami"))
    run: RunConfig = field(default_factory=RunConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    algorithm: NamedConfig = field(default_factory=lambda: NamedConfig("reinforce"))
    network: NamedConfig = field(default_factory=lambda: NamedConfig("mlp"))
    encoder: NamedConfig = field(default_factory=lambda: NamedConfig(""))
    selfplay: SelfplayConfig = field(default_factory=SelfplayConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    opponents: OpponentsConfig = field(default_factory=OpponentsConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    datastore: DatastoreConfig = field(default_factory=DatastoreConfig)

    def __post_init__(self) -> None:
        # Omitted move caps are resolved from the game's default and flagged as defaulted; the flag
        # flips to explicit on any later assignment (even to the same number), see _TracksDefaults.
        self.resolve_defaults()

    def resolve_defaults(self) -> "TrainingConfig":
        """Resolve game-derived move caps against the *current* game selection (idempotent).

        Called at construction and again by the Trainer, so a configuration whose ``game`` was
        reassigned after construction uses the selected game's default before any collection.
        """
        pending = [section for section in ("selfplay", "evaluation")
                   if getattr(self, section).max_plies is None or not getattr(self, section).max_plies_explicit]
        spec = get_spec(self.game.name, self.game.params)
        if pending:
            default = spec.default_max_plies
            for section in pending:
                getattr(self, section).max_plies = default
                getattr(self, section).mark_max_plies_defaulted()
        for section, name, attribute, fallback in GAME_LIST_DEFAULTS:
            config = getattr(self, section)
            if getattr(config, name) is None or config.list_is_default(name):
                setattr(config, name, list(getattr(spec, attribute, fallback)))
                object.__setattr__(config, f"_{name}_explicit", False)
                object.__setattr__(config, f"_{name}_default", list(getattr(config, name)))
        return self

    def defaulted_max_plies(self) -> set[str]:
        """Sections whose max_plies is game-derived (never explicitly assigned)."""
        return {section for section in ("selfplay", "evaluation") if not getattr(self, section).max_plies_explicit}

    # ------------------------------------------------------------ conversion
    @classmethod
    def from_dict(cls, data: dict[str, Any], validate: bool = True) -> "TrainingConfig":
        if validate:
            errors = validate_config_dict(data)
            if errors:
                raise ConfigError("invalid training config:\n" + "\n".join(errors))
        return _build(cls, data)

    def to_dict(self, preserve_defaults: bool = True) -> dict[str, Any]:
        """Serialise the configuration.

        Move caps that were resolved from the game's default are written as ``null`` (when
        ``preserve_defaults`` is true) so every round trip (``with_overrides`` chains, saved run
        configs, CLI reconstruction) keeps them game-derived instead of freezing them as explicit.
        """
        data = asdict(self)
        if preserve_defaults:
            for section in self.defaulted_max_plies():
                data[section]["max_plies"] = None
            for section, name, _, _ in GAME_LIST_DEFAULTS:
                if getattr(self, section).list_is_default(name):
                    data[section][name] = None
        return data

    @classmethod
    def load(cls, path: str | Path, validate: bool = True) -> "TrainingConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data, validate=validate)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def with_overrides(self, **dotted: Any) -> "TrainingConfig":
        """Return a copy with ``section.key=value`` overrides applied (e.g. run__iterations=3)."""
        data = copy.deepcopy(self.to_dict())
        # to_dict() keeps omitted caps as null, so a game override re-resolves exactly those;
        # explicitly supplied values (even ones equal to the old default) are preserved.
        # ``None`` means "argument absent"; pass USE_DEFAULT to reset a nullable field to its default.
        for key, value in dotted.items():
            if value is None:
                continue
            section, _, name = key.partition("__")
            if not name or section not in data:
                raise ConfigError(f"unknown override {key}")
            data[section][name] = None if value is USE_DEFAULT else value
        return TrainingConfig.from_dict(data)


def _build(cls, data: dict[str, Any]):
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        target = f.type
        if isinstance(target, str):
            target = _TYPES.get(target, None)
        if target is not None and is_dataclass(target) and isinstance(value, dict):
            kwargs[f.name] = _build(target, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


_TYPES = {
    "RunConfig": RunConfig,
    "DeviceConfig": DeviceConfig,
    "NamedConfig": NamedConfig,
    "SelfplayConfig": SelfplayConfig,
    "ReplayConfig": ReplayConfig,
    "OptimizerConfig": OptimizerConfig,
    "OpponentsConfig": OpponentsConfig,
    "EvaluationConfig": EvaluationConfig,
    "DatastoreConfig": DatastoreConfig,
}


def load_schema() -> dict[str, Any]:
    return _load_packaged(SCHEMA_NAME)


def validate_config_dict(data: dict[str, Any]) -> list[str]:
    """Validate against the packaged training-config schema (see hasami_learn.schema_validation)."""
    return _validate(load_schema(), data)


def tiny_config(algorithm: str, run_id: str | None = None, **algorithm_params: Any) -> TrainingConfig:
    """A CPU-friendly smoke configuration used by tests and quick checks."""
    return TrainingConfig(
        run=RunConfig(run_id=run_id or f"tiny-{algorithm}", iterations=1, backend="python", checkpoint_every=1, eval_every=0, keep_checkpoints=2),
        device=DeviceConfig(device="cpu", amp="off", threads=2),
        algorithm=NamedConfig(algorithm, dict(algorithm_params)),
        network=NamedConfig("mlp", {"hidden": 64, "layers": 1}),
        selfplay=SelfplayConfig(games_per_iteration=2, max_plies=40, temperature=1.0, temperature_plies=10, symmetry_augmentation=False),
        replay=ReplayConfig(capacity=2048, batch_size=32, min_size=16),
        optimizer=OptimizerConfig(lr=1e-3, updates_per_iteration=2),
        opponents=OpponentsConfig(strategy="self", pool_size=2, snapshot_every=0, classical=["random"], classical_weight=0.0),
        evaluation=EvaluationConfig(games=0, opponents=["random"], max_plies=40),
    )
