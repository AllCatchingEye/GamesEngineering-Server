from dataclasses import dataclass
import random
import string
from websockets import WebSocketServerProtocol
from controller.delayed_controller import DelayedController
from controller.player_controller import PlayerController
from controller.random_controller import RandomController
from controller.websocket_controller import WebSocketController

from logic.game import Game
from state.bot_types import BotType, bot_type_to_controller
from state.event import LobbyInformationUpdate


@dataclass
class RunConfig:
    bots: list[BotType]


class Lobby:
    clients: list[WebSocketServerProtocol]
    id: str
    available_bots: list[BotType]

    def __init__(self) -> None:
        self.clients = []
        self.id = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
        self.available_bots = list(BotType)

    async def add_player(self, ws: WebSocketServerProtocol) -> None:
        self.clients.append(ws)
        for client in self.clients:
            await client.send(
                LobbyInformationUpdate(
                    self.id, len(self.clients), self.available_bots
                ).to_json()
            )

    async def run(self, run_config: RunConfig) -> None:
        game = Game()
        controllers: list[PlayerController] = []
        for client in self.clients:
            controllers.append(WebSocketController(client))

        while len(controllers) < 4 and len(run_config.bots) > 0:
            c = bot_type_to_controller(run_config.bots.pop())
            controllers.append(DelayedController(c))

        while len(controllers) < 4:
            controllers.append(DelayedController(RandomController()))

        random.shuffle(controllers)

        game.controllers = controllers
        await game.run(games_to_play=20)
