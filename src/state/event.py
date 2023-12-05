import json
from abc import ABC
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Type, TypeVar

from websockets import Data

from state.card import Card
from state.gametypes import GameGroup, Gametype, GametypeWithSuit
from state.money import Money
from state.player import PlayerId
from state.suits import Suit
from state.player import PlayPartiesStruct


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o: object):
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


def parse_as(message: str | Data, event_type: Type[E]) -> E:
    text = message if isinstance(message, str) else message.decode()
    dct = json.loads(text)
    # if has id, delete
    if "id" in dct:
        del dct["id"]
    return event_type(**dct)


@dataclass
class GameStartUpdate(Event):
    player: PlayerId
    hand: list[Card]


@dataclass
class PlayDecisionUpdate(Event):
    player: PlayerId
    wants_to_play: bool


@dataclass
class GametypeDeterminedUpdate(Event):
    player: PlayerId | None
    gametype: Gametype
    suit: Suit | None
    parties: PlayPartiesStruct | None


@dataclass
class CardPlayedUpdate(Event):
    player: PlayerId
    card: Card


@dataclass
class RoundResultUpdate(Event):
    round_winner: PlayerId
    points: int


@dataclass
class GameEndUpdate(Event):
    winner: list[PlayerId]
    play_party: PlayPartiesStruct
    points: list[int]


@dataclass
class AnnouncePlayPartyUpdate(Event):
    parties: PlayPartiesStruct


@dataclass
class GameGroupChosenUpdate(Event):
    player: PlayerId
    game_groups: list[GameGroup]


@dataclass
class MoneyUpdate(Event):
    player: PlayerId
    money: Money


@dataclass
class PlayOrderUpdate(Event):
    order: list[PlayerId]


@dataclass
class LobbyInformationUpdate(Event):
    lobby_id: str


@dataclass
class LobbyInformationPlayerJoinedUpdate(Event):
    lobby_id: str
    player: PlayerId
    slot_id: int


@dataclass
class LobbyInformationPlayerLeftUpdate(Event):
    lobby_id: str
    player: PlayerId


@dataclass
class LobbyInformationPlayerReadyUpdate(Event):
    player: PlayerId
    lobby_id: str
    player_is_ready: bool


@dataclass
class PlayerWantsToPlayQuery(Event):
    current_lowest_gamegroup: GameGroup


@dataclass
class PlayerSelectGameTypeQuery(Event):
    choosable_gametypes: list[GametypeWithSuit]


@dataclass
class PlayerChooseGameGroupQuery(Event):
    available_groups: list[GameGroup]


@dataclass
class PlayerPlayCardQuery(Event):
    playable_cards: list[Card]


@dataclass
class PlayerWantsToPlayAnswer(Event):
    decision: bool


@dataclass
class PlayerChooseGameGroupAnswer(Event):
    gamegroup_index: int


@dataclass
class PlayerSelectGameTypeAnswer(Event):
    gametype_index: int


@dataclass
class PlayerPlayCardAnswer(Event):
    card_index: int
