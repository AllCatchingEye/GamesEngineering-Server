from state.ranks import Rank, get_value_of
from state.suits import Suit


class Card:
    def __init__(self, rank: Rank, suit: Suit) -> None:
        self.suit = suit
        self.rank = rank
        self.value = get_value_of(self.rank)

    def get_suit(self) -> Suit:
        return self.suit

    def get_rank(self) -> Rank:
        return self.rank

    def get_value(self) -> int:
        return self.value

    def __hash__(self) -> int:
        return hash((self.suit, self.rank))

    def __str__(self) -> str:
        return f"{self.suit} {self.rank}"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank == other.rank and self.suit == other.suit
