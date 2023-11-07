import asyncio

import json
from websockets.typing import Data

# import logging
from websockets.server import WebSocketServerProtocol, serve
from websockets import broadcast
from logic.game import Game
from secrets import token_urlsafe

CLIENT_COUNT = 4
free_id = 0
active_clients = 0

JOIN: dict[str, tuple[Game, set[WebSocketServerProtocol]] ]= dict()

async def main() -> None:
    print("Websocket starts")
    async with serve(handler, "localhost", 8765):
        print("Websocket runs")
        await asyncio.Future()  # run forever

async def handler(websocket: WebSocketServerProtocol) -> None:
    message: Data = await websocket.recv()
    event: dict[str, str] = json.loads(message)

    if event["type"] == "start":
        await start(websocket)
    else:
        join_key: str = event["join_key"]
        await join(websocket, join_key)

async def start(websocket: WebSocketServerProtocol) -> None:
    global free_id, active_clients
    connected: set[WebSocketServerProtocol] = {websocket}

    game: Game = Game()
    
    join_key: str = token_urlsafe(12)
    JOIN[join_key] = (game, connected)

    print(f"Starting new game with key: {join_key}")
    try:
        event = {
            "join_key": join_key,
            "id": 0
        }
        free_id += 1
        active_clients += 1

        await websocket.send(json.dumps(event))
        await play(websocket, connected)
    finally:
        del JOIN[join_key]

async def join(websocket: WebSocketServerProtocol, join_key: str) -> None:
    global free_id, active_clients
    print(f"Client joined lobby with key: {join_key}")
    lobby: tuple[Game, set[WebSocketServerProtocol]] = JOIN[join_key]
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
