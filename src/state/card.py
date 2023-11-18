from dataclasses import dataclass

from state.ranks import Rank
from state.suits import Suit


@dataclass
class Card:
    suit: Suit
    rank: Rank

    def __init__(self, suit: Suit, rank: Rank) -> None:
        self.suit = suit
        self.rank = rank

    def get_suit(self) -> Suit:
        return self.suit

    def get_rank(self) -> Rank:
        return self.rank

    def __hash__(self) -> int:
        return hash((self.suit, self.rank))

    def __str__(self) -> str:
        return f"{self.suit} {self.rank}"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit
