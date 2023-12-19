import asyncio
import json
import logging
import os
import random
import string
from numpy import append

from websockets import Data, WebSocketServerProtocol, serve
from controller import websocket_controller
from controller.delayed_controller import DelayedController

from controller.delayed_controller import DelayedController
from controller.passive_controller import PassiveController
from controller.random_controller import RandomController
from controller.websocket_controller import WebSocketController
from logic.game import Game

TIMEOUT_SECS = 5 * 60  # 5 Minutes timeout
CLIENTS: set[WebSocketServerProtocol] = set()

LOBBIES: dict[str, set[WebSocketController]] = {}

async def main() -> None:
    host = os.environ.get("HOST", "localhost")
    port = int(os.environ.get("PORT", 8765))
    print("Server WebSocket started")
    async with serve(handler, host, port, open_timeout=TIMEOUT_SECS):
        print("Websocket runs")
        await asyncio.Future()  # run forever


async def handler(ws: WebSocketServerProtocol) -> None:
    data: Data = await ws.recv()
    message: dict[str, str] = json.loads(data)

    CLIENTS.add(ws)

    key = "iD" if message.get("iD") else "id"
    match message[key]:
        case "lobby_host":
            lobby_type = message["lobby_type"]

            if lobby_type == "single":
                ## create single player lobby with 3 bots
                await single_player_lobby(ws)
            elif lobby_type == "multi":
                await start_multi_player_lobby(ws)
            else:
                await ws.send("Unknown lobby type")
        case "lobby_join":
            lobby_id = message["lobby_id"]
            LOBBIES[lobby_id].add(WebSocketController(ws))
            await ws.send("Joined lobby")

            if len(LOBBIES[lobby_id]) == 4:
                await multi_player_lobby(lobby_id)
        case _:
            msg = {key: "input_error", "message": "Unknown message"}
            await ws.send(msg)

async def single_player_lobby(ws: WebSocketServerProtocol) -> None:
    game: Game = create_single_player_game(ws)

    await game.run(games_to_play=1)
    await ws.wait_closed()

async def start_multi_player_lobby(ws: WebSocketServerProtocol) -> None:
    lobby_id = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    LOBBIES[lobby_id] = set()
    await ws.send(lobby_id)  # TODO

    LOBBIES[lobby_id].add(WebSocketController(ws))

async def multi_player_lobby(lobby_id: str) -> None:
    game: Game = create_multi_player_game(lobby_id)

    await game.run(games_to_play=1)
    # TODO do we need ws.wait_closed() here?

def create_multi_player_game(lobby_id: str) -> Game:
    game: Game = Game()
    s = LOBBIES[lobby_id]
    game.controllers = list(map(lambda x: x, s))

    return game

def create_single_player_game(ws: WebSocketServerProtocol) -> Game:
    game: Game = Game()
    game.controllers = [
        WebSocketController(ws),
        DelayedController(RandomController()),
        DelayedController(PassiveController()),
        DelayedController(PassiveController()),
    ]
    return game

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    asyncio.run(main())
    print("Server WebSocket stopped")
