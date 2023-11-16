import json
from websockets.server import WebSocketServerProtocol
from controller.player_controller import PlayerController
from state.player import Player
from state.card import Card
from state.gametypes import GameGroup, Gametype
from state.stack import Stack
from state.suits import Suit
from state.event import (
    Event,
    PlayerChooseGameGroupQuery,
    PlayerPlayCardQuery,
    PlayerSelectGameTypeQuery,
    PlayerWantsToPlayQuery,
)

class WebSocketController(PlayerController):
    def __init__(self, player: Player, ws: WebSocketServerProtocol) -> None:
        self.ws = ws
        super().__init__(player)

    async def wants_to_play(self, current_lowest_gamegroup: GameGroup) -> bool:
        request = PlayerWantsToPlayQuery(current_lowest_gamegroup)
        await self.ws.send(request.to_json())

        decision = await self.get_int_answer("decision")
        return decision == 1

    async def select_gametype(
        self, choosable_gametypes: list[tuple[Gametype, Suit | None]]
    ) -> tuple[Gametype, Suit | None]:
        request = PlayerSelectGameTypeQuery(choosable_gametypes)
        await self.ws.send(request.to_json())

        gametype_index = await self.get_int_answer("gametype_index")
        return choosable_gametypes[gametype_index]

    async def play_card(self, stack: Stack, playable_cards: list[Card]) -> Card:
        request = PlayerPlayCardQuery(stack, playable_cards)
        await self.ws.send(request.to_json())

        card_index = await self.get_int_answer("card_index")
        return playable_cards[card_index]

    async def choose_game_group(self, available_groups: list[GameGroup]) -> GameGroup:
        request = PlayerChooseGameGroupQuery(available_groups)
        await self.ws.send(request.to_json())

        gamegroup_index = await self.get_int_answer("gamegroup_index")
        return available_groups[gamegroup_index]

    async def get_int_answer(self, field_name: str) -> int:
        answer = await self.get_answer(field_name)
        if isinstance(answer, int):
            return answer

        raise TypeError(f"Expected int, got {type(answer)}")

    async def get_answer(self, field_name: str) -> int | str:
        response = await self.ws.recv()
        data = json.loads(response)
        answer = data[field_name]
        return answer

    async def on_game_event(self, event: Event) -> None:
        message = event.to_json()
        await self.ws.send(message)
