from abc import ABC, abstractmethod

from state.card import Card
from state.event import Event
from state.gametypes import GameGroup, Gametype
from state.stack import Stack
from state.suits import Suit


class PlayerController(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    async def on_game_event(self, event: Event) -> None:
        """Called when a game event occurs"""

    @abstractmethod
    async def wants_to_play(self, current_lowest_gamegroup: GameGroup) -> bool:
        """Decide if the player wants to play or pass on."""

    @abstractmethod
    async def select_gametype(
        self, choosable_gametypes: list[tuple[Gametype, Suit | None]]
    ) -> tuple[Gametype, Suit | None]:
        """
        Ask the player what game type to be played.
        This is only called if the player wants to play.
        """

    @abstractmethod
    async def play_card(self, stack: Stack, playable_cards: list[Card]) -> Card:
        """Determine which card to play"""

    @abstractmethod
    async def choose_game_group(self, available_groups: list[GameGroup]) -> GameGroup:
        """Choose the highest game group you would play"""
