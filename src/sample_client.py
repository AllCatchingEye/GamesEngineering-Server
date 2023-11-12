import asyncio
import json
from logging import error
import websockets
from websockets.client import WebSocketClientProtocol


# Just for testing global
join_key_g = ""

async def start_client(game_mode: str) -> None:
    uri = "ws://localhost:8765"

    async with websockets.connect(uri) as websocket:
        print("Starting client...")
        while True:
            await run_client(websocket, game_mode)

async def run_client(websocket: WebSocketClientProtocol, game_mode: str) -> None:
    if game_mode == "single":
        await start(websocket, game_mode)
    # elif game_mode == "multi":
    #     await join(websocket)
    else:
        error("Unknown game mode")

async def start(websocket: WebSocketClientProtocol, game_mode: str) -> None:
    print("Asking server to start single player game")
    response: dict[str, object] = {
        "id": "connect",
        "game_mode": game_mode
    }
    await websocket.send(json.dumps(response))

    response = await websocket.recv()
    message: dict[str, object] = json.loads(response)

    response = await websocket.recv()
    player_id = 0
    await play(websocket)
    await websocket.wait_closed()

# async def join(websocket: WebSocketClientProtocol) -> None:
#     print(f"Asking server to join game with key {join_key_g}")
#     event: dict[str, str] = {
#         "game_mode": "join",
#         "join_key": join_key_g
#     }
#
#     message = json.dumps(event)
#     await websocket.send(message)
#
#     response: Data = await websocket.recv()
#     id_assignment: dict[str, int] = json.loads(response)
#     id: int = id_assignment["id"]
#     turn: int = id
#     await play(websocket, id, turn)
#
#     await websocket.wait_closed()


async def play(ws: WebSocketClientProtocol) -> None:
    gamestate: dict[str, object] = dict()
    async for message in ws:
        data = json.loads(message)
        match data["id"]:
            case "wants_to_play":
                await wants_to_play(ws, data) 
            case "select_gametype":
                await select_gametype(ws, data) 
            case "play_card":
                await play_card(ws, data) 
            case _:
                gamestate.update(data)


async def wants_to_play(ws: WebSocketClientProtocol, data: dict[str, object]) -> None:
    print("Your hand:")
    print(data["cards"])
    print("Decisions before you:")
    print(data["decisions"])
    response: dict[str, str] = {
        "decision": input("Do you want to play? (y/n) ")
    } 
    await ws.send(json.dumps(response))

async def select_gametype(ws: WebSocketClientProtocol, data: dict[str, object]) -> None:
    print("Choose a gamemode:")
    for index, gametype in enumerate(data["choosable_gametypes"]):
        print(f"{index}: {gametype}")
    gametype_index = int(input())
    response: dict[str, int] = {
        "gametype_index": gametype_index
    }
    await ws.send(json.dumps(response))

async def play_card(ws: WebSocketClientProtocol, data: dict[str, object]) -> None:
    print("The stack is:")
    print(data["stack"])
    print("Choose a card to play:")
    for index, card in enumerate(data["playable_cards"]):
        print(f"{index}: {card}")
    card_index = int(input())
    response: dict[str, int] = {
        "card_index": card_index
    }
    await ws.send(json.dumps(response))

async def send_action(websocket: WebSocketClientProtocol) -> None:
    action = {
        "action": "dummy_action"
    }
    message = json.dumps(action)
    await websocket.send(message)

async def get_response(ws: WebSocketClientProtocol) -> None:
    pass


asyncio.run(start_client("single"))
