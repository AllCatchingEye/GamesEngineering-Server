from dataclasses import dataclass
from enum import Enum


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
