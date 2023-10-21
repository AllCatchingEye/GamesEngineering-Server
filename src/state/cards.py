from state.ranks import Rank
from state.suits import Suit

class Card:
    def __init__(self, rank: Rank, suit: Suit):
        self.suit = suit
        self.rank = rank

    def get_value(self) -> int:
        return self.rank.value

    # def __str__(self):
    #     return self.rank + " of " + self.suit
