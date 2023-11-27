from abc import ABC, abstractmethod

from state.card import Card
from state.event import Event
from state.player import PlayerId
from state.stack import Stack


class ISelectCardAgent(ABC):
    player_id: PlayerId
    
    def __init__(self) -> None:
        pass

    @abstractmethod
    def reset(self) -> None:
        """Invoked to reset the agent's internal state"""
        pass

    @abstractmethod
    def select_card(self, stack: Stack, playable_cards: list[Card]) -> Card:
        """Select card via drl algorithm based on stack and playable cards"""
        pass

    @abstractmethod
    def on_game_event(self, event: Event) -> None:
        """Handle game events"""
