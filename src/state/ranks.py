from enum import Enum


class Rank(Enum):
    ASS = 8
    ZEHN = 7
    KOENIG = 6
    OBER = 5
    UNTER = 4
    NEUN = 3
    ACHT = 2
    SIEBEN = 1

    def __hash__(self) -> int:
        return hash((self.name, self.value))

    def __str__(self) -> str:
        return self.name


class TrumpRank(Enum):
    OBER = 8
    UNTER = 7
    ASS = 6
    ZEHN = 5
    KOENIG = 4
    NEUN = 3
    ACHT = 2
    SIEBEN = 1

    def __hash__(self) -> int:
        return hash((self.name, self.value))

    def __str__(self) -> str:
        return self.name


def get_all_ranks() -> list[Rank]:
    """Returns a list of all Rank enum values."""
    return list(Rank)


def get_value_of(rank: Rank) -> int:
    """Returns the value associated with the given rank."""
    values = {
        Rank.ASS: 11,
        Rank.ZEHN: 10,
        Rank.KOENIG: 4,
        Rank.OBER: 3,
        Rank.UNTER: 2,
        Rank.NEUN: 0,
        Rank.ACHT: 0,
        Rank.SIEBEN: 0,
    }
    return values.get(rank, 0)


def get_trump_value_of(rank: TrumpRank) -> int:
    """Returns the value associated with the given rank."""
    values = {
        TrumpRank.ASS: 11,
        TrumpRank.ZEHN: 10,
        TrumpRank.KOENIG: 4,
        TrumpRank.OBER: 3,
        TrumpRank.UNTER: 2,
        TrumpRank.NEUN: 0,
        TrumpRank.ACHT: 0,
        TrumpRank.SIEBEN: 0,
    }
    return values.get(rank, 0)
