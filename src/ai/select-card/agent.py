from abc import ABC, abstractmethod

from state.card import Card
from state.stack import Stack
from state.event import Event

class ISelectCardAgent(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Invoked to initialize the agent"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Invoked to reset the agent's internal state"""
        pass

    @abstractmethod
    def play_card(self, stack: Stack, playable_cards: list[Card]) -> Card:
        """Select card via drl algorithm based on stack and playable cards"""
        pass

    @abstractmethod
    def on_game_event(self, event: Event) -> None:
        """Handle game events"""
