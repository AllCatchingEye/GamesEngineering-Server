import json
from abc import ABC
from dataclasses import dataclass, is_dataclass, asdict
from typing import Type, TypeVar
from enum import Enum

from state.card import Card
from state.gametypes import GameGroup, Gametype
from state.hand import Hand
from state.money import Money
from state.player import Player
from state.stack import Stack
from state.suits import Suit


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o: object):
        if isinstance(o, Player):
            return str(o.player_id)
        if is_dataclass(o):
            result = asdict(o)
            result["id"] = getattr(o, "__name__", o.__class__.__name__)
            return result
        if isinstance(o, Enum):
            return o.name
        return super().default(o)


@dataclass
class Event(ABC):
    def to_json(self) -> str:
        return json.dumps(self, cls=EnhancedJSONEncoder)


E = TypeVar("E", bound=Event)


def parse_as(message: str, event_type: Type[E]) -> E:
    dct = json.loads(message)
    return event_type(**dct)


@dataclass
class GameStartUpdate(Event):
    hand: Hand


@dataclass
class PlayDecisionUpdate(Event):
    player: Player
    wants_to_play: bool


@dataclass
class GametypeDeterminedUpdate(Event):
    player: Player | None
    gametype: Gametype
    suit: Suit | None
    parties: list[list[Player]] | None


@dataclass
class CardPlayedUpdate(Event):
    player: Player
    card: Card
    stack: Stack


@dataclass
class RoundResultUpdate(Event):
    round_winner: Player
    points: int
    stack: Stack


@dataclass
class GameEndUpdate(Event):
    winner: list[Player]
    play_party: list[list[Player]]
    points: list[int]


@dataclass
class AnnouncePlayPartyUpdate(Event):
    parties: list[list[Player]]


@dataclass
class GameGroupChosenUpdate(Event):
    player: Player
    game_groups: list[GameGroup]


@dataclass
class MoneyUpdate(Event):
    player: Player
    money: Money


@dataclass
class LobbyInformationUpdate(Event):
    lobby_id: str


@dataclass
class LobbyInformationPlayerJoinedUpdate(Event):
    lobby_id: str
    player: Player
    slot_id: int


@dataclass
class LobbyInformationPlayerLeftUpdate(Event):
    lobby_id: str
    player: Player


@dataclass
class LobbyInformationPlayerReadyUpdate(Event):
    player: Player
    lobby_id: str
    player_is_ready: bool


@dataclass
class PlayerWantsToPlayQuery(Event):
    current_lowest_gamegroup: GameGroup


@dataclass
class PlayerSelectGameTypeQuery(Event):
    choosable_gametypes: list[tuple[Gametype, Suit | None]]


@dataclass
class PlayerChooseGameGroupQuery(Event):
    available_groups: list[GameGroup]


@dataclass
class PlayerPlayCardQuery(Event):
    stack: Stack
    playable_cards: list[Card]


@dataclass
class PlayerWantsToPlayAnswer(Event):
    decision: bool


@dataclass
class ChooseGameGroupAnswer(Event):
    gamegroup_index: int


@dataclass
class PlayerSelectGameTypeAnswer(Event):
    gametype_index: int


@dataclass
class PlayerPlayCardAnswer(Event):
    card_index: int
