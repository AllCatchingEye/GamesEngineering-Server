import uuid
from dataclasses import dataclass

from state.card import Card
from state.hand import Hand
from state.money import Money


class PlayerId(str):
    pass


@dataclass
class Player:
    id: PlayerId
    slot_id: int
    # turn order during each game round
    turn_order: int
    money: Money
    points: int
    hand: Hand
    played_cards: list[Card]
    stitches: list[Card]

    def __init__(self, slot_id: int, turn_order: int) -> None:
        self.id = PlayerId(str(uuid.uuid4()))
        self.slot_id = slot_id
        self.turn_order = turn_order
        self.points = 0
        self.money = Money(0)
        self.hand: Hand = Hand([])
        self.played_cards = []
        self.stitches = []

    def lay_card(self, card: Card) -> None:
        self.hand.remove_card(card)
        self.played_cards.append(card)

    def reset(self) -> None:
        self.played_cards = []
        self.stitches = []
        self.points = 0

    def get_amount_stitches(self) -> int:
        return round(len(self.stitches) / 4)

    def __repr__(self) -> str:
        return f"Player {self.id}"
