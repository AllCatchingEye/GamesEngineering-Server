from abc import ABC, abstractmethod

from state.card import Card
from state.gametypes import Gametype
from state.suits import Suit


class ISelectGameAgent(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Invoked to initialize the agent"""

    @abstractmethod
    def reset(self) -> None:
        """Invoked to reset the agent's internal state"""

    @abstractmethod
    def should_play(self, hand_cards: list[Card], decisions: list[bool | None]) -> bool:
        """Invoked to receive a decision if the agent would play"""

    @abstractmethod
    def select_game_type(
        self,
        hand_cards: list[Card],
        choosable_game_types: list[tuple[Gametype, Suit | None]],
    ) -> tuple[Gametype, Suit | None]:
        """Invoked to receive a decision which game type the agent would play"""
