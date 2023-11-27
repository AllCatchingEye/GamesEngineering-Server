from dataclasses import dataclass
from enum import Enum

from state.stakes import Stake
from state.suits import Suit


class Gametype(Enum):
    # The value of the enum is the stake
    SOLO = 1
    WENZ = 2
    GEIER = 3
    FARBWENZ = 4
    FARBGEIER = 5
    SAUSPIEL = 6
    RAMSCH = 7

    def __repr__(self) -> str:
        return self.name


@dataclass
class GametypeWithSuit:
    gametype: Gametype
    suit: Suit | None

    def __hash__(self) -> int:
        return hash((self.gametype, self.suit))

    def __str__(self) -> str:
        if self.suit is None:
            return str(self.gametype.name)
        return f"{self.suit.name} {self.gametype.name}"


stake_for_gametype = {
    Gametype.SOLO: Stake.HIGH,
    Gametype.WENZ: Stake.HIGH,
    Gametype.GEIER: Stake.HIGH,
    Gametype.FARBWENZ: Stake.MID,
    Gametype.FARBGEIER: Stake.MID,
    Gametype.SAUSPIEL: Stake.STANDARD,
    Gametype.RAMSCH: Stake.STANDARD,
}


class GameGroup(Enum):
    # Group acutal gametypes to have less detailed options for play decision discussion
    HIGH_SOLO = 1  # Farbsolo
    MID_SOLO = 2  # Wenz und Geier
    LOW_SOLO = 3  # Farbwenz und Farbgeier
    SAUSPIEL = 4

    def __repr__(self) -> str:
        return self.name
