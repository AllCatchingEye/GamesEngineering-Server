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
