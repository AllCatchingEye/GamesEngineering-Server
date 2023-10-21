from enum import Enum

class Rank(Enum):
    Ass = 11
    Zehn = 10
    Koenig = 4
    Ober = 3
    Unter = 2
    Neun = 0
    Acht = 0
    Sieben = 0

def get_all_ranks() -> list[Rank]:
    ranks: list[Rank] = [rank for rank in Rank]
    return ranks
