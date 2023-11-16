from dataclasses import dataclass
from state.card import Card
from state.hand import Hand
from state.money import Money

@dataclass
class Player:
    id: int
    money: Money
    points: int
    hand: Hand
    played_cards: list[Card]

    def __init__(self, player_id: int) -> None:
        self.id = player_id
        self.points = 0
        self.money = Money(0)
        self.hand: Hand = Hand([])
        self.played_cards = []

    def lay_card(self, card: Card) -> None:
        self.hand.remove_card(card)
        self.played_cards.append(card)

    def __repr__(self) -> str:
        return f"Player {self.id}"
