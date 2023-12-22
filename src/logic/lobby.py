import random
import string
from websockets import WebSocketServerProtocol
from controller.delayed_controller import DelayedController
from controller.player_controller import PlayerController
from controller.random_controller import RandomController
from controller.websocket_controller import WebSocketController

from logic.game import Game
from state.event import LobbyInformationUpdate


class Lobby:
    clients: list[WebSocketServerProtocol]
    id: str

    def __init__(self) -> None:
        self.clients = []
        self.id = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))

    async def add_player(self, ws: WebSocketServerProtocol) -> None:
        self.clients.append(ws)
        for client in self.clients:
            await client.send(
                LobbyInformationUpdate(self.id, len(self.clients)).to_json()
            )

    async def run(self):
        game = Game()
        controllers: list[PlayerController] = []
        for client in self.clients:
            controllers.append(WebSocketController(client))

        while len(controllers) < 4:
            controllers.append(DelayedController(RandomController()))
        game.controllers = controllers
        await game.run()
