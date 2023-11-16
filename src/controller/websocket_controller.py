from dataclasses import asdict
import json
from controller.player_controller import PlayerController
from state.player import Player
from state.card import Card
from state.event import Event, EnhancedJSONEncoder
from state.gametypes import GameGroup, Gametype
from state.stack import Stack
from state.suits import Suit

from websockets.server import WebSocketServerProtocol


class WebsocketController(PlayerController):
    def __init__(self, player: Player, ws: WebSocketServerProtocol) -> None:
        self.ws = ws
        super().__init__(player)

    async def wants_to_play(self, current_lowest_gamegroup: GameGroup) -> bool:
        message = "Your hand:\n"
        message += f"{self.player.hand}\n"
        message += f"You have to play atleast {current_lowest_gamegroup}"
        data = {
            "id": "wants_to_play",
            "message": message
        }
        await self.ws.send(json.dumps(data))

        decision = await self.get_answer("decision")
        return decision == "y"

    async def select_gametype(
        self, choosable_gametypes: list[tuple[Gametype, Suit | None]]
    ) -> tuple[Gametype, Suit | None]:
        message = "Choose a gamemode:"
        message += self.list_to_string(choosable_gametypes)
        request = {
            "id": "select_gametype",
            "message": message,
        }
        await self.ws.send(json.dumps(request))

        gametype_index: int = await self.get_answer("gametype_index")
        return choosable_gametypes[gametype_index]

    async def play_card(self, stack: Stack, playable_cards: list[Card]) -> Card:
        message: str = "The stack is:\n"
        message += f"{stack}\n"
        message += "Choose a card to play:\n"
        message += f"{self.list_to_string(playable_cards)}\n"
        data = {
            "id": "play_card",
            "message": message
        }
        await self.ws.send(json.dumps(data, cls=EnhancedJSONEncoder))

        card_index: int = await self.get_answer("card_index")
        return playable_cards[card_index]

    async def choose_game_group(self, available_groups: list[GameGroup]) -> GameGroup:
        message: str = "Choose a gamegroup:\n"
        message += f"{self.list_to_string(available_groups)}\n"
        request = {
            "id": "choose_game_group",
            "message": message
        }
        await self.ws.send(json.dumps(request))

        gamegroup_index = await self.get_answer("gamegroup")
        return available_groups[gamegroup_index]

    def list_to_string(self, o: list[object]) -> str:
        message = ""
        for index, val in enumerate(o):
            message += f"{index}: {val}\n"
        return message

    async def get_answer(self, index_name: str) -> int | str:
        response = await self.ws.recv()
        data = json.loads(response)
        print(data)
        answer = data[index_name]
        return answer

    async def on_game_event(self, event: Event) -> None:
        message = event.to_json()
        await self.ws.send(message)
