from abc import ABC
from dataclasses import dataclass

from state.card import Card
from state.gametypes import Gametype
from state.hand import Hand
from state.money import Money
from state.player import Player
from state.stack import Stack
from state.suits import Suit


@dataclass
class Event(ABC):
    pass


@dataclass
class GameStartEvent(Event):
    hand: Hand


@dataclass
class PlayDecisionEvent(Event):
    player: Player
    wants_to_play: bool


@dataclass
class GametypeWishedEvent(Event):
    player: Player
    gametype: tuple[Gametype, Suit | None]


@dataclass
class GametypeDeterminedEvent(Event):
    player: Player | None
    gametype: Gametype
    suit: Suit | None
    parties: list[list[Player]] | None


@dataclass
class CardPlayedEvent(Event):
    player: Player
    card: Card
    stack: Stack


@dataclass
class RoundResultEvent(Event):
    round_winner: Player  # TODO: Teams?
    points: int
    stack: Stack


@dataclass
class GameEndEvent(Event):
    winner: list[Player]
    play_party: list[list[Player]]
    points: list[int]


@dataclass
class AnnouncePlayPartyEvent(Event):
    parties: list[list[Player]]


@dataclass
class MoneyUpdateEvent(Event):
    money: Money
