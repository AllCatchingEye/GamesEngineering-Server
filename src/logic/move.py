from state.cards import Card


class Move:
    def __init__(self, card: Card) -> None:
        self.card = card

    def __str__(self) -> str:
        return self.card.__str__()
