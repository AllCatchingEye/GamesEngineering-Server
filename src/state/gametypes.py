from enum import Enum

from state.money import Money


class Gametype(Enum):
    # The value of the enum is the stake
    SOLO = Money.from_euro(5)
    WENZ = Money.from_euro(5)
    GEIER = Money.from_euro(5)
    FARBWENZ = Money.from_euro(3)
    FARBGEIER = Money.from_euro(3)
    SAUSPIEL = Money.from_euro(1)
    RAMSCH = Money.from_euro(1)
