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


class GameGroup(Enum):
    # Group acutal gametypes to have less detailed options for play decision discussion
    HIGH_SOLO = 1  # Farbsolo
    MID_SOLO = 2  # Wenz und Geier
    LOW_SOLO = 3  # Farbwenz und Farbgeier
    SAUSPIEL = 4
