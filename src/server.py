import asyncio
import json

from websockets import Data, WebSocketServerProtocol, serve

from controller.random_controller import RandomController
from controller.websocket_controller import WebSocketController
from logic.game import Game

TIMEOUT_SECS = 5 * 60  # 5 Minutes timeout
CLIENTS: set[WebSocketServerProtocol] = set()

async def main() -> None:
    print("Server WebSocket started")
    async with serve(handler, "localhost", 8765, open_timeout=TIMEOUT_SECS):
        print("Websocket runs")
        await asyncio.Future()  # run forever


async def handler(ws: WebSocketServerProtocol) -> None:
    data: Data = await ws.recv()
    message: dict[str, str] = json.loads(data)

    CLIENTS.add(ws)

    match message["id"]:
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
            msg = {"id": "input_error", "message": "Unknown message"}
            await ws.send(msg)


async def single_player_lobby(ws: WebSocketServerProtocol):
    game: Game = create_single_player_game(ws)

    await game.run()
    await ws.wait_closed()


def create_single_player_game(ws: WebSocketServerProtocol) -> Game:
    game: Game = Game()
    game.controllers = [
        WebSocketController(game.players[0], ws),
        RandomController(game.players[1]),
        RandomController(game.players[2]),
        RandomController(game.players[3]),
    ]
    return game


if __name__ == "__main__":
    asyncio.run(main())
    print("Server WebSocket stopped")
