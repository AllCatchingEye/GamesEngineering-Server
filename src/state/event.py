from abc import ABC
from dataclasses import dataclass

from state.card import Card
from state.gametypes import Gametype
from state.hand import Hand
from state.player import Player
from state.stack import Stack


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
    gametype: Gametype


@dataclass
class GametypeDeterminedEvent(Event):
    player: Player | None
    gametype: Gametype


@dataclass
class CardPlayedEvent(Event):
    player: Player
    card: Card
    stack: Stack


@dataclass
class RoundResultEvent(Event):
    round_winner: Player  # TODO: Teams?
    points: int  # TODO: Scoreboard?
    stack: Stack


@dataclass
class GameEndEvent(Event):
    winner: Player  # TODO: Teams?
    points: int  # TODO: Scoreboard?
