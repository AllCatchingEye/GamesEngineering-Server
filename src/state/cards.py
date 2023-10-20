class Card:
    def __init__(self, rank: str, suit: str):
        self.suit = suit
        self.rank = rank

    def __str__(self):
        return self.rank + " of " + self.suit
