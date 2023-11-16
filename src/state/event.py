import json
from abc import ABC
from dataclasses import dataclass, is_dataclass, asdict
from enum import Enum
import json

from state.card import Card
from state.gametypes import GameGroup, Gametype
from state.hand import Hand
from state.player import Player
from state.stack import Stack
from state.suits import Suit



@dataclass
class Event(ABC):
    def to_json(self) -> str:
        return json.dumps(self, cls=EnhancedJSONEncoder)
    pass

@dataclass
class GameStart(Event):
    player_order: list[int]
    player: Player
    hand: Hand
    gametypes: list[tuple[Gametype, Suit | None]]

@dataclass
class PlayDecisionEvent(Event):
    player: Player
    wants_to_play: bool


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
class GameGroupChosenEvent(Event):
    player: Player
    game_groups: list[GameGroup]

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o: object):
        if is_dataclass(o):
            result = asdict(o)
            result["id"] = getattr(o, "__name__", o.__class__.__name__)
            return result
        if isinstance(o, Enum):
            return o.name
        return super().default(o)
