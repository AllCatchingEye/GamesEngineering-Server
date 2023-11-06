from abc import ABC, abstractmethod

from state.card import Card
from state.hand import Hand
from state.player import Player
from state.stack import Stack
from state.suits import Suit


class GameMode(ABC):
    suit: Suit | None
    trumps: [Card]

    def __init__(self, suit: Suit | None):
        self.suit = suit

    @abstractmethod
    def get_playable_cards(self, stack: Stack, hand: Hand) -> list[Card]:
        """Determine which cards can be played"""

    def get_trump_cards(self) -> list[Card]:
        """Returns a list of all trump cards"""
        return self.trumps

    def determine_stitch_winner(self, stack: Stack) -> Player:
        strongest_played_card = stack.get_played_cards()[0]
        for played_card in stack.get_played_cards()[1:]:
            if self.__card_is_stronger_than(played_card.get_card(), strongest_played_card.get_card()):
                strongest_played_card = played_card
        return strongest_played_card.get_player()

    def __card_is_stronger_than(self, card_one: Card, card_two: Card) -> bool:
        if card_one in self.trumps:
            if card_two in self.trumps:
                # Trumpf-Vergleich
                return self.trumps.index(card_one) < self.trumps.index(card_two)
            else:
                return True
        elif card_one.get_suit == card_two.get_suit:
            return card_one.get_value() > card_two.get_value()
        else:
            return False
