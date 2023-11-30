from abc import ABC, abstractmethod

from state.card import Card
from state.event import Event
from state.player import PlayerId
from state.stack import Stack


class ISelectCardAgent(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        """Invoked to reset the agent's internal state"""

    @abstractmethod
    def select_card(self, stack: Stack, playable_cards: list[Card]) -> Card:
        """Select card via drl algorithm based on stack and playable cards"""

    @abstractmethod
    def on_game_event(self, event: Event, player_id: PlayerId) -> None:
        """Handle game events"""

    @abstractmethod
    def set_hand_cards(self, hand_cards: list[Card]):
        """Update hand cards"""
