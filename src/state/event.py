from abc import ABC
from dataclasses import dataclass
from enum import Enum
from state import player

from state.card import Card
from state.gametypes import Gametype
from state.hand import Hand
from state.player import Player
from state.stack import Stack


# class EventType(Enum):
#     # Phase 1: Game start
#     GAME_START = 0
#     PLAY_DECISION = 1
#     GAMETYPE_DECISION = 2
#     GAMETYPE_DETERMINED = 3
#     # Phase 2: Play phase
#     CARD_PLAYED = 4
#     ROUND_RESULT = 5
#     # Phase 3: Game end
#     GAME_END = 6


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
    player: Player
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
