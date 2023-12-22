import asyncio
import json
import logging
import os
import random
import string

from websockets import Data, WebSocketServerProtocol, serve

from controller.delayed_controller import DelayedController
from controller.random_controller import RandomController
from controller.websocket_controller import WebSocketController
from logic.game import Game
from state.event import LobbyInformationPlayerUpdate

TIMEOUT_SECS = 5 * 60  # 5 Minutes timeout
CLIENTS: set[WebSocketServerProtocol] = set()

LOBBIES: dict[str, Game] = {}


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
                await single_player(ws)
            elif lobby_type == "multi":
                await create_lobby(ws)
            else:
                await ws.send("Unknown lobby type")
        case "lobby_join":
            lobby_id = message["lobby_id"]
            await add_player_to_lobby(lobby_id, ws)

            if len(LOBBIES[lobby_id].controllers) == 4:
                await start_lobby(lobby_id)
        case _:
            msg = {key: "input_error", "message": "Unknown message"}
            await ws.send(msg)


async def single_player(ws: WebSocketServerProtocol) -> None:
    game: Game = Game()
    game.controllers = [
        WebSocketController(ws),
        DelayedController(RandomController()),
        DelayedController(PassiveController()),
        DelayedController(PassiveController()),
    ]

    await game.run(games_to_play=1)
    await ws.wait_closed()


async def add_player_to_lobby(lobby_id: str, ws: WebSocketServerProtocol) -> None:
    game: Game = LOBBIES[lobby_id]
    player = game.add_controller(WebSocketController(ws))
    for c in game.controllers:
        await c.on_game_event(
            LobbyInformationPlayerUpdate(lobby_id, player.id, player.slot_id)
        )


async def create_lobby(ws: WebSocketServerProtocol) -> None:
    lobby_id = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
    LOBBIES[lobby_id] = Game()
    await add_player_to_lobby(lobby_id, ws)


async def start_lobby(lobby_id: str) -> None:
    game: Game = LOBBIES[lobby_id]
    await game.run(games_to_play=1)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    asyncio.run(main())
    print("Server WebSocket stopped")
