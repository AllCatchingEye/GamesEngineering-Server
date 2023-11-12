from abc import ABC, abstractmethod

from state.card import Card
from state.event import Event
from state.gametypes import Gametype
from state.player import Player
from state.stack import Stack
from state.suits import Suit


class PlayerController(ABC):
    def __init__(self, player: Player) -> None:
        self.player = player

    @abstractmethod
    async def on_game_event(self, event: Event) -> None:
        """Called when a game event occurs"""

    @abstractmethod
    def wants_to_play(self, decisions: list[bool | None]) -> bool:
        """Decide if the player wants to play or pass on."""

    @abstractmethod
    def select_gametype(
        self, choosable_gametypes: list[tuple[Gametype, Suit | None]]
    ) -> tuple[Gametype, Suit | None]:
        """
        Ask the player what game type to be played.
        This is only called if the player wants to play.
        """
        # TODO: What have the players before said chosen?

    @abstractmethod
    def play_card(self, stack: Stack, playable_cards: list[Card]) -> Card:
        """Determine which card to play"""
