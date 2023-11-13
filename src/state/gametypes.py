from enum import Enum


class Gametype(Enum):
    # The value of the enum is the stake
    SOLO = 1
    WENZ = 2
    GEIER = 3
    FARBWENZ = 4
    FARBGEIER = 5
    SAUSPIEL = 6
    RAMSCH = 7


class GameGroup(Enum):
    # Group acutal gametypes to have less detailed options for play decision discussion
    HIGH_SOLO = 1  # Farbsolo
    MID_SOLO = 2  # Wenz und Geier
    LOW_SOLO = 3  # Farbwenz und Farbgeier
    ALL = 4
