from enum import Enum

class Suit(Enum):
    Eichel = 4
    Gras = 3
    Herz = 2
    Schellen = 1

def get_all_suits() -> list[Suit]:
    suits: list[Suit] = [suit for suit in Suit]
    return suits
