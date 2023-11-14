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
    response: dict[str, object] = {"id": "connect", "game_mode": game_mode}
    await websocket.send(json.dumps(response))

    await websocket.recv()  # connect_response, not important for websocket client for now
    await websocket.recv()  # game_started message
    player_id = 0
    await play(websocket, player_id)
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


async def play(ws: WebSocketClientProtocol, player_id: int) -> None:
    gamestate: dict[str, object] = {}
    async for message in ws:
        data: dict[str, object] = json.loads(message)
        match data["id"]:
            # Actions
            case "wants_to_play":
                await wants_to_play(ws, data, player_id)
            case "select_gametype":
                await select_gametype(ws, data, player_id)
            case "play_card":
                await play_card(ws, data, player_id)
            case "new_game":
                print("Your game is over. Do you want to play a new game?")
                answer: str = input("Enter your answer. (y/n)")
                await ws.send(json.dumps({"wants_new_game": answer}))
            case _:  # Events
                gamestate = event_update(gamestate, data)


def event_update(
    gamestate: dict[str, object], data: dict[str, object]
) -> dict[str, object]:
    show_event(data)
    gamestate.update(data)
    return gamestate


def show_event(event: dict[str, object]) -> None:
    match event["id"]:
        case "PlayDecisionEvent":
            player = event["player"]
            player_id: int = player["id"]
            plays = event["wants_to_play"]
            print(f"Player {player_id} wants to play: {plays}")
        case "RoundResultEvent":
            winner = event["round_winner"]
            winner_id: int = winner["id"]
            points: int = event["points"]
            print(f"Round was won by {winner_id} with {points} points")
            print("============================================================")
        case "GameEndEvent":
            winners = event["winner"]
            points = event["points"]
            print(f"Game was won by {winners} with {points} points")
            print("============================================================")
        case "CardPlayedEvent":
            player = event["player"]
            player_id = player["id"]
            card = event["card"]
            suit = card["suit"]
            rank = card["rank"]
            print(f"Player {player_id} played the card {suit} {rank}")
        case "RoundResultEvent":
            winner = event["round_winner"]
            winner_id = winner["id"]
            print(f'Round was won by {winner_id} with {event["points"]} points')
        case _:
            pass


async def wants_to_play(
    ws: WebSocketClientProtocol, data: dict[str, object], player_id: int
) -> None:
    print("Player: " + str(player_id))
    print("Your hand:")
    for card in data["cards"]:
        suit = card["suit"]
        rank = card["rank"]
        print(f"{suit} {rank}")
    print("Decisions before you:")
    print(data["decisions"])
    response: dict[str, str] = {"decision": input("Do you want to play? (y/n) ")}
    await ws.send(json.dumps(response))


async def select_gametype(
    ws: WebSocketClientProtocol, data: dict[str, object], player_id: int
) -> None:
    print("Player: " + str(player_id))
    print("Choose a gamemode:")
    print(data["choosable_gametypes"])
    gametype_index = int(input("Gametype: "))
    response: dict[str, int] = {"gametype_index": gametype_index}
    await ws.send(json.dumps(response))


async def play_card(
    ws: WebSocketClientProtocol, data: dict[str, object], player_id: int
) -> None:
    print("Player: " + str(player_id))
    print("The stack is:")
    print(data["stack"])
    print("Choose a card to play:")
    print(data["playable_cards"])
    card_index = int(input("Card: "))
    response: dict[str, int] = {"card_index": card_index}
    await ws.send(json.dumps(response))


asyncio.run(start_client("single"))
