import asyncio
import json
import sys
import websockets
from websockets import WebSocketClientProtocol

from state.event import (
    GameEndUpdate,
    PlayerChooseGameGroupQuery,
    PlayerPlayCardAnswer,
    PlayerPlayCardQuery,
    PlayerSelectGameTypeAnswer,
    PlayerWantsToPlayAnswer,
    PlayerWantsToPlayQuery,
    PlayerChooseGameGroupAnswer,
    PlayerSelectGameTypeQuery,
    parse_as,
)


async def start_client(game_mode: str) -> None:
    uri = "ws://localhost:8765"

    async with websockets.connect(uri) as websocket:
        print("Starting client...")
        while True:
            await run_client(websocket, game_mode)


async def run_client(websocket: WebSocketClientProtocol, game_mode: str) -> None:
    if game_mode == "single":
        await start(websocket, game_mode)
    else:
        raise NotImplementedError(f"Game mode {game_mode} not implemented")


async def start(websocket: WebSocketClientProtocol, game_mode: str) -> None:
    print("Asking server to start single player game")
    response: dict[str, object] = {"id": "lobby_host", "lobby_type": game_mode}
    await websocket.send(json.dumps(response))

    await play(websocket)

    await websocket.wait_closed()


async def play(ws: WebSocketClientProtocol) -> None:
    async for message in ws:
        print("")
        print("")
        dct = json.loads(message)
        match dct["id"]:
            case PlayerWantsToPlayQuery.__name__:
                event = parse_as(message, PlayerWantsToPlayQuery)
                print(f"You have to play atleast {event.current_lowest_gamegroup}")
                decision = input("Do you want to play? (y/n) ")
                answer = PlayerWantsToPlayAnswer(decision == "y")
                await ws.send(json.dumps(answer.to_json()))
            case PlayerChooseGameGroupQuery.__name__:
                event = parse_as(message, PlayerChooseGameGroupQuery)
                print("Choose a gamegroup:")
                for index, gamegroup in enumerate(event.available_groups):
                    print(f"{index}: {gamegroup}")
                decision = input()
                answer = PlayerChooseGameGroupAnswer(gamegroup_index=int(decision))
                await ws.send(json.dumps(answer.to_json()))
            case PlayerSelectGameTypeQuery.__name__:
                event = parse_as(message, PlayerSelectGameTypeQuery)
                print("Choose a game type:")
                for index, gametype in enumerate(event.choosable_gametypes):
                    print(f"{index}: {gametype}")
                decision = input()
                answer = PlayerSelectGameTypeAnswer(gametype_index=int(decision))
                await ws.send(json.dumps(answer.to_json()))
            case PlayerPlayCardQuery.__name__:
                event = parse_as(message, PlayerPlayCardQuery)
                print("The stack is:")
                print(event.stack)
                print("Choose a card to play:")
                for index, card in enumerate(event.playable_cards):
                    print(f"{index}: {card}")
                decision = input()
                answer = PlayerPlayCardAnswer(card_index=int(decision))
                await ws.send(json.dumps(answer.to_json()))
            case GameEndUpdate.__name__:
                event = parse_as(message, GameEndUpdate)
                print(event)
                print("GAME ENDED")
                sys.exit(0)
            case _:
                print(dct)


if __name__ == "__main__":
    asyncio.run(start_client("single"))
