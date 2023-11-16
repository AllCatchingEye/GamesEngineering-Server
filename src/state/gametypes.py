from dataclasses import dataclass
from enum import Enum

from state.stakes import Stake


@dataclass
class Gametype(Enum):
    # The value of the enum is the stake
    SOLO = 1
    WENZ = 2
    GEIER = 3
    FARBWENZ = 4
    FARBGEIER = 5
    SAUSPIEL = 6
    RAMSCH = 7

    def __hash__(self):
        return hash(self.name)


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
