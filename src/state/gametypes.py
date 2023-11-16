from enum import Enum

from state.stakes import Stake


class Gametype(Enum):
    # The value of the enum is the stake
    SOLO = Stake.HIGH
    WENZ = Stake.HIGH
    GEIER = Stake.HIGH
    FARBWENZ = Stake.MID
    FARBGEIER = Stake.MID
    SAUSPIEL = Stake.STANDARD
    RAMSCH = Stake.STANDARD


class GameGroup(Enum):
    # Group acutal gametypes to have less detailed options for play decision discussion
    HIGH_SOLO = 1  # Farbsolo
    MID_SOLO = 2  # Wenz und Geier
    LOW_SOLO = 3  # Farbwenz und Farbgeier
    SAUSPIEL = 4
