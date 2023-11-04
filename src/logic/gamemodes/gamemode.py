from abc import ABC, abstractmethod

from state.card import Card
from state.hand import Hand
from state.player import Player
from state.stack import Stack
from state.suits import Suit


class GameMode(ABC):
    suit: Suit | None

    def __init__(self, suit: Suit | None):
        self.suit = suit

    @abstractmethod
    def get_trump_cards(self) -> list[Card]:
        """Returns a list of all trump cards"""

    @abstractmethod
    def get_playable_cards(self, stack: Stack, hand: Hand) -> list[Card]:
        """Determine which cards can be played"""

    @abstractmethod
    def determine_stitch_winner(self, stack: Stack) -> Player:
        """Determine the winner of the current stitch"""
        # TODO: What about teams?
