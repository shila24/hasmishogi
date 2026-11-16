"""Versioned rule configuration; game-phase flags are consumed by the later engine."""
from dataclasses import asdict, dataclass, fields
import json


@dataclass(frozen=True)
class RiichiRules:
    ruleset_id: str = 'riichi-4p-v1'
    rounds: str = 'hanchan'
    start_points: int = 25000
    return_points: int = 30000
    uma: tuple[int, int, int, int] = (20, 10, -10, -20)
    red_fives: bool = True
    kuitan: bool = True
    atozuke: bool = True
    ippatsu: bool = True
    ura_dora: bool = True
    kan_dora: bool = True
    open_kan_dora_after_discard: bool = True
    closed_kan_dora_immediate: bool = True
    double_ron: bool = True
    atama_hane: bool = False
    triple_ron_abort: bool = True
    kyuushu_kyuuhai: bool = True
    suufon_renda: bool = True
    suucha_riichi: bool = True
    suukaikan: bool = True
    tenpai_payment: int = 3000
    riichi_stick: int = 1000
    honba: int = 300
    tobi: bool = True
    agari_yame: bool = True
    kazoe_yakuman: bool = True
    double_yakuman: bool = False
    multiple_yakuman: bool = True
    pao: bool = False
    renhou: str = 'none'
    kiriage_mangan: bool = False
    double_wind_pair_fu: int = 4
    nagashi_mangan: bool = True
    discard_furiten: bool = True
    temporary_furiten: bool = True
    riichi_furiten: bool = True
    track_tsumogiri: bool = True

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if type(field.default) is bool and type(value) is not bool:
                raise ValueError(f'{field.name} must be boolean')
        if self.ruleset_id != 'riichi-4p-v1':
            raise ValueError('unsupported ruleset_id')
        if self.rounds not in ('hanchan', 'tonpuu') or self.renhou not in ('none', 'mangan', 'yakuman'):
            raise ValueError('unsupported rounds or renhou option')
        for name in ('start_points', 'return_points', 'tenpai_payment', 'riichi_stick', 'honba'):
            n = getattr(self, name)
            if type(n) is not int or n < 0 or n % 100:
                raise ValueError(f'{name} must be a nonnegative multiple of 100')
        if self.return_points < self.start_points or self.start_points == 0:
            raise ValueError('return points must be at least positive starting points')
        if self.honba % 300 or self.tenpai_payment % 300:
            raise ValueError('honba and tenpai payment must be divisible by 300')
        if not isinstance(self.uma, (tuple, list)) or len(self.uma) != 4 or any(type(n) is not int for n in self.uma):
            raise ValueError('uma must contain four integers')
        object.__setattr__(self, 'uma', tuple(self.uma))
        if sum(self.uma) != 0 or tuple(sorted(self.uma, reverse=True)) != self.uma:
            raise ValueError('uma must be descending and sum to zero')
        if type(self.double_wind_pair_fu) is not int or self.double_wind_pair_fu not in (2, 4):
            raise ValueError('double wind pair must score 2 or 4 fu')

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, value: str) -> 'RiichiRules':
        data = json.loads(value)
        if not isinstance(data, dict) or set(data) - {f.name for f in fields(cls)}:
            raise ValueError('expected a rule object with known fields')
        return cls(**data)
