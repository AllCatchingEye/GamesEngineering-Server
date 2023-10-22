from enum import Enum

class Rank(Enum):
    Ass = 8
    Zehn = 7
    Koenig = 6
    Ober = 5
    Unter = 4
    Neun = 3
    Acht = 2
    Sieben = 1

    def __eq__(self, object: object) -> bool:
        if isinstance(object, Rank):
            return self.name == object.name
        return False

    def __hash__(self):
        return hash((self.name, self.value))

    def __lt__(self, object: object) -> bool:
        if isinstance(object, Rank):
            return get_value_of(self) < get_value_of(object)
        return False

    def __str__(self) -> str:
        return self.name

def get_all_ranks() -> list[Rank]:
    ranks: list[Rank] = [rank for rank in Rank]
    return ranks

def get_value_of(rank: Rank) -> int:
    values = {
        Rank.Ass : 11,
        Rank.Zehn : 10,
        Rank.Koenig : 4,
        Rank.Ober : 3,
        Rank.Unter : 2,
        Rank.Neun : 0,
        Rank.Acht : 0,
        Rank.Sieben : 0,
    }
    value: int | None = values.get(rank)

    if value is None:
        return 0

    return value
