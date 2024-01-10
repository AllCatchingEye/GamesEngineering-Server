import asyncio
import json
import logging
import os

from websockets import Data, WebSocketServerProtocol, serve

from controller.delayed_controller import DelayedController
from controller.passive_controller import PassiveController
from controller.random_controller import RandomController
from controller.websocket_controller import WebSocketController
from logic.game import Game

TIMEOUT_SECS = 5 * 60  # 5 Minutes timeout
CLIENTS: set[WebSocketServerProtocol] = set()


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

    key = "iD" if message["iD"] else "id"
    match message[key]:
        case "lobby_host":
            lobby_type = message["lobby_type"]

            if lobby_type == "single":
                ## create single player lobby with 3 bots
                await single_player_lobby(ws)

                ## add player to game

                ## send back single player lobby information
            elif lobby_type == "multi":
                raise NotImplementedError("Multiplayer not implemented")
            else:
                await ws.send("Unknown lobby type")
        case "lobby_join":
            raise NotImplementedError("Multiplayer not implemented")
        case _:
            msg = {key: "input_error", "message": "Unknown message"}
            await ws.send(msg)


async def single_player_lobby(ws: WebSocketServerProtocol) -> None:
    game: Game = create_single_player_game(ws)

    await game.run(games_to_play=1)
    await ws.wait_closed()


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
