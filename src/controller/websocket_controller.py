from dataclasses import asdict
import json
from controller.player_controller import PlayerController
from state.player import Player
from state.card import Card
from state.event import Event, EnhancedJSONEncoder
from state.gametypes import Gametype
from state.stack import Stack
from state.suits import Suit

from websockets.server import WebSocketServerProtocol


class WebsocketController(PlayerController):
    def __init__(self, player: Player, ws: WebSocketServerProtocol) -> None:
        self.ws = ws
        super().__init__(player)

    async def wants_to_play(self, decisions: list[bool | None]) -> bool:
        data = asdict(self.player.hand)
        data["decisions"] = str(decisions)
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
        data: dict[str, object] = {
            "id": "select_gametype",
            "choosable_gametypes": self.to_str(choosable_gametypes)
        }
        message = json.dumps(data, cls=EnhancedJSONEncoder)
        await self.ws.send(message)

        response = await self.ws.recv()
        response_data = json.loads(response)
        gametype_index: int = response_data["gametype_index"]

        return choosable_gametypes[gametype_index]

    async def play_card(self, stack: Stack, playable_cards: list[Card]) -> Card:
        data: dict[str, object] = {
            "id": "play_card",
            "stack": str(stack),
            "playable_cards": self.to_str(playable_cards)
        }
        message = json.dumps(data, cls=EnhancedJSONEncoder)
        await self.ws.send(message)

        response = await self.ws.recv()
        response_data = json.loads(response)
        card_index: int = response_data["card_index"]
        return playable_cards[card_index]

    async def on_game_event(self, event: Event) -> None:
        message: str = await self.to_message(event)
        await self.ws.send(message)

    def to_str(self, o: object) -> str:
        string: str = ""
        for index, content in enumerate(o):
            string += str(index )+ ': ' + str(content )+ '\n'
        return string

    async def to_message(self, event: Event) -> str:
        message: str = event.to_json()
        return message
