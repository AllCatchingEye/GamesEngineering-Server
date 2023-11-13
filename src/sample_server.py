import asyncio

import json
import random
from websockets.typing import Data

from controller.random_controller import RandomController
from controller.websocket_controller import WebsocketController

# import logging
from websockets.server import WebSocketServerProtocol, serve
from websockets import broadcast
from logic.game import Game
from secrets import token_urlsafe

CLIENT_COUNT = 4
free_id = 0
active_clients = 0

CLIENTS: set[WebSocketServerProtocol] = set()
RUNNING_GAMES: set[Game] = set()
LOBBIES: dict[str, tuple[Game, set[WebSocketServerProtocol]] ]= dict()

async def main() -> None:
    print("Websocket starts")
    async with serve(handler, "localhost", 8765):
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
        case "info_lobby_join":
            await join_lobby(ws, message)
        case _:
            await ws.send("Unknown message")

async def connect(ws: WebSocketServerProtocol, message: dict[str, any]):
    game_mode: str = message["game_mode"]
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
    message = json.dumps({
        "id": "game_start"
    })
    await ws.send(message)
    print("Game starts")
    await game.run()

    if await wants_new_game(ws):
        await game.run()


async def wants_new_game(ws: WebSocketServerProtocol) -> bool:
    new_game_message = json.dumps({
        "id": "new_game"
    })
    await ws.send(new_game_message)

    response: Data = await ws.recv()
    print(response)
    wants_new_game: str = response
    return wants_new_game == "y"

async def send_lobby_info(ws: WebSocketServerProtocol):
    lobby_id: str = token_urlsafe()
    message = json.dumps({
        "id": "info_lobby",
        "id_lobby": lobby_id
    })
    await ws.send(message)

async def join_lobby(ws: WebSocketServerProtocol, message: dict[str, any]):
    pass

async def start(websocket: WebSocketServerProtocol) -> None:
    global free_id, active_clients
    connected: set[WebSocketServerProtocol] = {websocket}

    rng = random.Random()
    game: Game = Game(rng)
    game.controllers = [
        WebsocketController(game.players[0], websocket),
        RandomController(game.players[1], rng),
        RandomController(game.players[2], rng),
        RandomController(game.players[3], rng),
    ]
    
    lobby_id: str = token_urlsafe(12)
    LOBBIES[lobby_id] = (game, connected)

    print(f"Starting new game with key: {lobby_id}")
    try:
        event = {
            "lobby_id": lobby_id,
            "id": 0
        }
        free_id += 1
        active_clients += 1

        await websocket.send(json.dumps(event))
        await play(websocket, connected)
    finally:
        del LOBBIES[lobby_id]

async def join(websocket: WebSocketServerProtocol, lobby_id: str) -> None:
    global free_id, active_clients
    print(f"Client joined lobby with key: {lobby_id}")
    lobby: tuple[Game, set[WebSocketServerProtocol]] = LOBBIES[lobby_id]
    connected: set[WebSocketServerProtocol] = lobby[1]
    connected.add(websocket)

    id_assignment: dict[str, int] = {
        "id": free_id
    }
    free_id += 1
    active_clients += 1
    message: str = json.dumps(id_assignment)
    await websocket.send(message)

    await play(websocket, connected)
    try:
        await websocket.wait_closed()
    finally:
        connected.remove(websocket)

async def play(websocket: WebSocketServerProtocol, connected: set[WebSocketServerProtocol]) -> None:
    turn_id = 0
    await start_round(connected)

    async for message in websocket:
        action = json.loads(message)

        gamestate = execute_action(action, turn_id)
        await update_clients(gamestate, connected)

async def start_round(connected: set[WebSocketServerProtocol]) -> None:
    gamestate: dict[str, str | int] = {
        "turn_id": 0,
        "gamestate": "dummy",
    }
    message = json.dumps(gamestate)
    broadcast(connected, message )

def parse_message(message: Data) -> dict[str, str | int]:
    action: dict[str, str | int] = json.loads(message)
    return action

def execute_action(event: dict[str, str | int], turn_id: int) -> dict[str, str | int]:
    action: object = event.get("action")
    print(f"Executed: {action}")
    return get_new_gamestate(turn_id)

def get_new_gamestate(turn_id: int) -> dict[str, str | int]:
    gamestate: dict[str, str | int] = {
        "turn_id": turn_id,
        "gamestate": "dummy"
    }
    return gamestate

async def update_clients(gamestate: dict[str, str | int], connected: set[WebSocketServerProtocol]) -> None:
    message: str = json.dumps(gamestate)
    broadcast(connected, message)

# logger = logging.getLogger('websockets')
# logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler)
asyncio.run(main())
