from enum import Enum

from state.money import Money


class Stake(Enum):
    HIGH = Money.from_euro(5)
    MID = Money.from_euro(3)
    STANDARD = Money.from_euro(1)
