import asyncio
import json
import sys
import websockets
from websockets.client import WebSocketClientProtocol
from websockets.typing import Data

# Just for testing global
join_key_g = ""

async def start_client(type: str) -> None:
    uri = "ws://localhost:8765"

    async with websockets.connect(uri) as websocket:
        print("Starting client...")
        while True:
            await run_client(websocket, type)

async def run_client(websocket: WebSocketClientProtocol, type: str) -> None:
    if type == "start":
        await start(websocket, type)
    else:
        await join(websocket)


async def start(websocket: WebSocketClientProtocol, type: str) -> None:
    print("Asking server to start new game...")
    event: dict[str, str] = {
        "type": type
    }

    message: str = json.dumps(event)
    await websocket.send(message)

    response: dict[str, str | int] = json.loads(await websocket.recv())
    join_key_g: str = response["join_key"]
    id: int = response["id"]
    print(f"Server created new game with key {join_key_g}")

    await play(websocket, id)
    await websocket.wait_closed()

async def join(websocket: WebSocketClientProtocol) -> None:
    print(f"Asking server to join game with key {join_key_g}")
    event: dict[str, str] = {
        "type": "join",
        "join_key": join_key_g
    }

    message = json.dumps(event)
    await websocket.send(message)

    response: Data = await websocket.recv()
    id_assignment: dict[str, int] = json.loads(response)
    id: int = id_assignment["id"]

    await play(websocket, id)

    await websocket.wait_closed()


async def play(websocket: WebSocketClientProtocol, id: int) -> None:
    async for message in websocket:
        gamestate = parse_message(message)
        turn_id = gamestate["turn_id"]
        print(f"Player {id} received gamestate: {gamestate}")

        # Just for testing
        if turn_id == id:
            input("Enter a action")
            await send_action(websocket)

def parse_message(message: Data) -> dict[str, str | int]:
    gamestate: dict[str, str | int] = json.loads(message)
    return gamestate

async def send_action(websocket: WebSocketClientProtocol) -> None:
    action = {
        "action": "dummy_action"
    }
    message = json.dumps(action)
    await websocket.send(message)

type: str = sys.argv[1]
if len(sys.argv) > 2:
    join_key_g = sys.argv[2]
asyncio.run(start_client(type))
