from ranks import Rank
from ranks import get_value_of
from suits import Suit

class Card:
    def __init__(self, rank: Rank, suit: Suit):
        self.suit = suit
        self.rank = rank

    def get_value(self):
        return get_value_of(self.rank)

    def get_suit(self) -> Suit:
        return self.suit

    def get_rank(self) -> Rank:
        return self.rank

    def __eq__(self, object: object) -> bool:
        if isinstance(object, Card):
            return self.suit == object.suit and self.rank == object.rank
        return False

    def __hash__(self):
        return hash((self.suit, self.rank))

    def __str__(self) -> str:
        return f"{self.suit} {self.rank}"
