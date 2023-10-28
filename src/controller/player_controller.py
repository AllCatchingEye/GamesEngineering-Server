from abc import ABC, abstractmethod

from state.card import Card
from state.gametypes import Gametype
from state.player import Player
from state.stack import Stack


class PlayerController(ABC):
    def __init__(self, player: Player) -> None:
        self.player = player

    @abstractmethod
    def wants_to_play(self, decisions: list[bool | None]) -> bool:
        """Decide if the player wants to play or pass on."""

    @abstractmethod
    def select_gametype(self, choosable_gametypes: list[Gametype]) -> Gametype:
        """Ask the player what game type to be played. This is only called if the player wants to play."""
        # TODO: What have the players before said chosen?

    @abstractmethod
    def announce_gametype(self, gametype: Gametype) -> None:
        """The gametype for this game has been decided"""

    @abstractmethod
    def play_card(self, stack: Stack, playable_cards: list[Card]) -> Card:
        """Determine which card to play"""

    @abstractmethod
    def announce_round_result(self) -> None:
        """The round has ended and the result is announced"""
        # TODO: Winner, points, next player, ...

    @abstractmethod
    def announce_game_result(self) -> None:
        """The game has ended and the result is announced"""
        # TODO: Winner, points, money, ...
