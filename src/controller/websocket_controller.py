import json
import logging
from typing import Type, TypeVar

from websockets.server import WebSocketServerProtocol

from controller.player_controller import PlayerController
from state.card import Card
from state.event import (
    Event,
    PlayerChooseGameGroupAnswer,
    PlayerChooseGameGroupQuery,
    PlayerPlayCardAnswer,
    PlayerPlayCardQuery,
    PlayerSelectGameTypeAnswer,
    PlayerSelectGameTypeQuery,
    PlayerWantsToPlayAnswer,
    PlayerWantsToPlayQuery,
    parse_as,
)
from state.gametypes import GameGroup, Gametype
from state.player import Player
from state.stack import Stack
from state.suits import Suit

E = TypeVar("E", bound=Event)


class WebSocketController(PlayerController):
    def __init__(self, player: Player, ws: WebSocketServerProtocol) -> None:
        self.ws = ws
        super().__init__(player)

    async def wants_to_play(self, current_lowest_gamegroup: GameGroup) -> bool:
        request = PlayerWantsToPlayQuery(current_lowest_gamegroup)
        await self.ws.send(request.to_json())

        response = await self.get_answer(PlayerWantsToPlayAnswer)
        return response.decision

    async def select_gametype(
        self, choosable_gametypes: list[tuple[Gametype, Suit | None]]
    ) -> tuple[Gametype, Suit | None]:
        request = PlayerSelectGameTypeQuery(choosable_gametypes)
        await self.ws.send(request.to_json())

        response = await self.get_answer(PlayerSelectGameTypeAnswer)
        return choosable_gametypes[response.gametype_index]

    async def play_card(self, stack: Stack, playable_cards: list[Card]) -> Card:
        request = PlayerPlayCardQuery(playable_cards)
        await self.ws.send(request.to_json())

        response = await self.get_answer(PlayerPlayCardAnswer)
        return playable_cards[response.card_index]

    async def choose_game_group(self, available_groups: list[GameGroup]) -> GameGroup:
        request = PlayerChooseGameGroupQuery(available_groups)
        await self.ws.send(request.to_json())

        response = await self.get_answer(PlayerChooseGameGroupAnswer)
        return available_groups[response.gamegroup_index]

    async def get_answer(self, event_type: Type[E]) -> E:
        response = await self.ws.recv()
        logging.info(f"Received {response}")
        data = json.loads(response)
        return parse_as(data, event_type)

    async def on_game_event(self, event: Event) -> None:
        message = event.to_json()
        logging.info(f"Sending {message}")
        await self.ws.send(message)
