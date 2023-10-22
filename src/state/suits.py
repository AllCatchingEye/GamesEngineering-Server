from enum import Enum

class Suit(Enum):
    Eichel = 4
    Gras = 3
    Herz = 2
    Schellen = 1

    def __str__(self) -> str:
        return self.name

    def __hash__(self):
        return hash((self.name))

    def __eq__(self, object: object) -> bool:
        if isinstance(object, Suit):
            return self.value == object.value
        return False

    def __lt__(self, object: object) -> bool:
        if isinstance(object, Suit):
            return self.value < object.value
        return False

def get_all_suits() -> list[Suit]:
    suits: list[Suit] = [suit for suit in Suit]
    return suits
