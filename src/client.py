import asyncio
import json
import sys
import threading

import websockets
from websockets import WebSocketClientProtocol

from state.event import (
    CreateLobbyRequest,
    GameEndUpdate,
    JoinLobbyRequest,
    LobbyInformationUpdate,
    PlayerChooseGameGroupAnswer,
    PlayerChooseGameGroupQuery,
    PlayerPlayCardAnswer,
    PlayerPlayCardQuery,
    PlayerSelectGameTypeAnswer,
    PlayerSelectGameTypeQuery,
    PlayerWantsToPlayAnswer,
    PlayerWantsToPlayQuery,
    StartLobbyRequest,
    parse_as,
)


async def start_client() -> None:
    uri = "ws://localhost:8765"

    async with websockets.connect(uri) as websocket:
        print("Starting client...")
        await run_client(websocket)


async def run_client(websocket: WebSocketClientProtocol) -> None:
    i = input("create or lobby_id?\n")
    if i == "":
        await start(websocket)
    else:
        await join(websocket, i)


async def start(websocket: WebSocketClientProtocol) -> None:
    await websocket.send(CreateLobbyRequest().to_json())
    data = await websocket.recv()
    lobby_id = parse_as(data, LobbyInformationUpdate).lobby_id
    print(f"Created lobby with id {lobby_id}")

    # wait for input() in thread and then send start lobby request
    def wait_for_input():
        input("Press enter to start lobby...\n")
        asyncio.run(
            websocket.send(StartLobbyRequest(lobby_id, ["HANDCRAFTED"]).to_json())
        )

    threading.Thread(target=wait_for_input).start()

    await play(websocket)

    await websocket.wait_closed()


async def join(websocket: WebSocketClientProtocol, lobby_id: str) -> None:
    await websocket.send(JoinLobbyRequest(lobby_id).to_json())

    await play(websocket)

    await websocket.wait_closed()


async def play(ws: WebSocketClientProtocol) -> None:
    async for message in ws:
        print("")
        print("")
        dct = json.loads(message)
        key = "iD" if "iD" in dct else "id"
        match dct[key]:
            case PlayerWantsToPlayQuery.__name__:
                event = parse_as(message, PlayerWantsToPlayQuery)
                print(f"You have to play atleast {event.current_lowest_gamegroup}")
                decision = input("Do you want to play? (y/n) ")
                answer = PlayerWantsToPlayAnswer(decision == "y")
                await ws.send(answer.to_json())
            case PlayerChooseGameGroupQuery.__name__:
                event = parse_as(message, PlayerChooseGameGroupQuery)
                print("Choose a gamegroup:")
                for index, gamegroup in enumerate(event.available_groups):
                    print(f"{index}: {gamegroup}")
                decision = input()
                answer = PlayerChooseGameGroupAnswer(gamegroup_index=int(decision))
                await ws.send(answer.to_json())
            case PlayerSelectGameTypeQuery.__name__:
                event = parse_as(message, PlayerSelectGameTypeQuery)
                print("Choose a game type:")
                for index, gametype in enumerate(event.choosable_gametypes):
                    print(f"{index}: {gametype}")
                decision = input()
                answer = PlayerSelectGameTypeAnswer(gametype_index=int(decision))
                await ws.send(answer.to_json())
            case PlayerPlayCardQuery.__name__:
                event = parse_as(message, PlayerPlayCardQuery)
                print("Choose a card to play:")
                for index, card in enumerate(event.playable_cards):
                    print(f"{index}: {card}")
                decision = input()
                answer = PlayerPlayCardAnswer(card_index=int(decision))
                await ws.send(answer.to_json())
            case GameEndUpdate.__name__:
                event = parse_as(message, GameEndUpdate)
                print(event)
                print("GAME ENDED")
                sys.exit(0)
            case LobbyInformationUpdate.__name__:
                event = parse_as(message, LobbyInformationUpdate)
                print(f"Lobby {event.lobby_id} has {event.size} players")
            case _:
                print(dct)


if __name__ == "__main__":
    asyncio.run(start_client())
