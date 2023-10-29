from abc import ABC
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
from typing import override

from state.card import Card
from state.gametypes import Gametype
from state.hand import Hand
from state.player import Player
from state.stack import Stack

class EventType(Enum):
    # Phase 1: Game start
    GAME_START = 0
    PLAY_DECISION = 1
    GAMETYPE_WISH = 2
    GAMETYPE_DETERMINED = 3
    # Phase 2: Play phase
    CARD_PLAYED = 4
    ROUND_RESULT = 5
    # Phase 3: Game end
    GAME_END = 6


@dataclass
class Event(ABC):
    def name(self) -> str:
        return self.__class__.__name__
    
    @abstractmethod
    def event_type(self) -> EventType:
        """Return the type of the event"""

    def splurge(self) -> dict[str, object]:
        return self.__dict__

@dataclass
class GameStartEvent(Event):
    hand: Hand

    @override
    def event_type(self) -> EventType:
        return EventType.GAME_START

@dataclass
class PlayDecisionEvent(Event):
    player: Player
    wants_to_play: bool

    @override
    def event_type(self) -> EventType:
        return EventType.PLAY_DECISION

@dataclass
class GametypeWishedEvent(Event):
    player: Player
    gametype: Gametype

    @override
    def event_type(self) -> EventType:
        return EventType.GAMETYPE_WISH
    
@dataclass
class GametypeDeterminedEvent(Event):
    player: Player | None
    gametype: Gametype

    @override
    def event_type(self) -> EventType:
        return EventType.GAMETYPE_DETERMINED

@dataclass
class CardPlayedEvent(Event):
    player: Player
    card: Card
    stack: Stack

    @override
    def event_type(self) -> EventType:
        return EventType.CARD_PLAYED

@dataclass
class RoundResultEvent(Event):
    round_winner: Player  # TODO: Teams?
    points: int  # TODO: Scoreboard?
    stack: Stack

    @override
    def event_type(self) -> EventType:
        return EventType.ROUND_RESULT

@dataclass
class GameEndEvent(Event):
    winner: Player  # TODO: Teams?
    points: int  # TODO: Scoreboard?
    
    @override
    def event_type(self) -> EventType:
        return EventType.GAME_END