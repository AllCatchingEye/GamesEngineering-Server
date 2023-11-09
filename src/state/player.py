from state.card import Card
from state.hand import Hand


class Player:
    def __init__(self, player_id: int) -> None:
        self.id = player_id
        self.points = 0
        self.hand: Hand
        self.played_cards: list[Card] = []

    def lay_card(self, card: Card) -> None:
        self.hand.remove_card(card)
        self.played_cards.append(card)

    def __repr__(self) -> str:
        return f"Player {self.id}"
