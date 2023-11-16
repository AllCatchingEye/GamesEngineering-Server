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
        data = asdict(self.player.hand)
        data["decisions"] = str(current_lowest_gamegroup)
        data["id"] = "wants_to_play"
        print(data)
        message = json.dumps(data, cls=EnhancedJSONEncoder)
        await self.ws.send(message)

        response = await self.ws.recv()
        data = json.loads(response)
        decision = data["decision"]
        return decision == "y"

    async def select_gametype(
        self, choosable_gametypes: list[tuple[Gametype, Suit | None]]
    ) -> tuple[Gametype, Suit | None]:
        data = choosable_gametypes.__dict__
        data["id"] = "select_gametype"
        message = json.dumps(data, cls=EnhancedJSONEncoder)
        await self.ws.send(message)

        response = await self.ws.recv()
        data = json.loads(response)
        gametype_index: int = data["gametype_index"]

        return choosable_gametypes[gametype_index]

    async def play_card(self, stack: Stack, playable_cards: list[Card]) -> Card:
        data = stack.__dict__
        data["playable_cards"] = playable_cards.__dict__
        data["id"] = "play_card"
        message = json.dumps(data, cls=EnhancedJSONEncoder)
        await self.ws.send(message)

        response = await self.ws.recv()
        data = json.loads(response)
        card_index: int = data["card_index"]
        return playable_cards[card_index]

    async def choose_game_group(self, available_groups: list[GameGroup]) -> GameGroup:
        message = "Choose a gamegroup:\n"
        message += self.list_to_string(available_groups)

        request = {
            id: "choose_game_group",
            message: message
        }
        await self.ws.send(json.dumps(request))

        answer = await 
        return available_groups[gamegroup_index]

    def list_to_string(self, o: list[object]) -> str:
        message = ""
        for index, val in enumerate(o):
            message += f"{index}: {val}\n"

    async def get_answer(self, index_name: str) -> int:
        response = await self.ws.recv()
        data = json.loads(response)
        answer = data[index_name]
        return answer

    async def on_game_event(self, event: Event) -> None:
        message = event.to_json()
        await self.ws.send(message)
