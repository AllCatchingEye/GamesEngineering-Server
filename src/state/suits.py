from enum import Enum

class Suit(Enum):
    EICHEL = 0
    GRAS = 1
    HERZ = 2
    SCHELLEN = 3

    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return self.name


def get_all_suits() -> list[Suit]:
    """Returns a list of all Suit enum values."""
    return list(Suit)
