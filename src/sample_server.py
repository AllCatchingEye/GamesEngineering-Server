import asyncio
import json
import random
from secrets import token_urlsafe

from websockets import broadcast

# import logging
from websockets.server import WebSocketServerProtocol, serve
from websockets.typing import Data

from controller.random_controller import RandomController
from controller.websocket_controller import WebsocketController
from logic.game import Game

CLIENT_COUNT = 4
free_id = 0
active_clients = 0

CLIENTS: set[WebSocketServerProtocol] = set()
RUNNING_GAMES: set[Game] = set()
LOBBIES: dict[str, tuple[Game, set[WebSocketServerProtocol]]] = {}

TIMEOUT_SECS = 5 * 60  # 5 Minutes timeout


async def main() -> None:
    print("Websocket starts")
    async with serve(handler, "localhost", 8765, open_timeout=TIMEOUT_SECS):
        print("Websocket runs")
        await asyncio.Future()  # run forever


async def handler(ws: WebSocketServerProtocol) -> None:
    data: Data = await ws.recv()
    message: dict[str, str] = json.loads(data)

    CLIENTS.add(ws)

    match message["id"]:
        case "connect":
            await connect(ws, message)
        case "player_info":
            pass
        case _:
            await ws.send("Unknown message")


async def connect(ws: WebSocketServerProtocol, message: dict[str, any]):
    game_mode: str | Unknown = message["game_mode"]
    if game_mode == "single":
        await open_game(ws)
    elif game_mode == "multi":
        await open_game(ws, multiplayer=True)
    else:
        print("Unknown option")


async def open_game(ws: WebSocketServerProtocol, multiplayer: bool = False):
    game = create_game(ws)
    RUNNING_GAMES.add(game)

    await send_lobby_info(ws)

    if not multiplayer:
        await start_game(ws, game)


def create_game(ws: WebSocketServerProtocol) -> Game:
    rng = random.Random()
    game: Game = Game(rng)
    game.controllers = [
        WebsocketController(game.players[0], ws),
        RandomController(game.players[1], rng),
        RandomController(game.players[2], rng),
        RandomController(game.players[3], rng),
    ]
    return game


async def start_game(ws: WebSocketServerProtocol, game: Game):
    message = json.dumps({"id": "game_start"})
    await ws.send(message)
    print("Game starts")
    await game.run()

    if await wants_new_game(ws):
        await game.run()


async def wants_new_game(ws: WebSocketServerProtocol) -> bool:
    new_game_message = json.dumps({"id": "new_game"})
    await ws.send(new_game_message)

    response: Data = await ws.recv()
    print(response)
    wants_new_game_res: str = response
    return wants_new_game_res == "y"


async def send_lobby_info(ws: WebSocketServerProtocol) -> None:
    lobby_id: str = token_urlsafe()
    message = json.dumps({"id": "info_lobby", "id_lobby": lobby_id})
    await ws.send(message)

async def start(ws: WebSocketServerProtocol) -> None:
    global free_id, active_clients
    connected: set[WebSocketServerProtocol] = {ws}

    rng = random.Random()
    game: Game = Game(rng)
    game.controllers = [
        WebsocketController(game.players[0], ws),
        RandomController(game.players[1], rng),
        RandomController(game.players[2], rng),
        RandomController(game.players[3], rng),
    ]

    lobby_id: str = token_urlsafe(12)
    LOBBIES[lobby_id] = (game, connected)

    print(f"Starting new game with key: {lobby_id}")
    try:
        event = {"lobby_id": lobby_id, "id": 0}
        free_id += 1
        active_clients += 1

        await ws.send(json.dumps(event))
        await play(ws, connected)
    finally:
        del LOBBIES[lobby_id]

async def play(
    ws: WebSocketServerProtocol, connected: set[WebSocketServerProtocol]
) -> None:
    turn_id = 0
    await start_round(connected)

    async for message in ws:
        action = json.loads(message)

        gamestate = execute_action(action, turn_id)
        await update_clients(gamestate, connected)


async def start_round(connected: set[WebSocketServerProtocol]) -> None:
    gamestate: dict[str, str | int] = {
        "turn_id": 0,
        "gamestate": "dummy",
    }
    message = json.dumps(gamestate)
    broadcast(connected, message)

def execute_action(event: dict[str, str | int], turn_id: int) -> dict[str, str | int]:
    action: object = event.get("action")
    print(f"Executed: {action}")
    return get_new_gamestate(turn_id)


def get_new_gamestate(turn_id: int) -> dict[str, str | int]:
    gamestate: dict[str, str | int] = {"turn_id": turn_id, "gamestate": "dummy"}
    return gamestate


async def update_clients(
    gamestate: dict[str, str | int], connected: set[WebSocketServerProtocol]
) -> None:
    message: str = json.dumps(gamestate)
    broadcast(connected, message)


# logger = logging.getLogger('websockets')
# logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler)
asyncio.run(main())
