import asyncio
import json
import logging
import os

from websockets import Data, WebSocketServerProtocol, serve

from logic.lobby import Lobby, RunConfig
from state.bot_types import BotType
from state.event import (
    CreateLobbyRequest,
    JoinLobbyRequest,
    StartLobbyRequest,
    parse_as,
)

TIMEOUT_SECS = 5 * 60  # 5 Minutes timeout
CLIENTS: set[WebSocketServerProtocol] = set()

LOBBIES: dict[str, Lobby] = {}


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

    key = "iD" if message.get("iD") else "id"
    match message[key]:
        case CreateLobbyRequest.__name__:
            lobby = Lobby()
            LOBBIES[lobby.id] = lobby
            await lobby.add_player(ws)
            received = await ws.recv()
            response = json.loads(received)

            key = "iD" if message.get("iD") else "id"
            match response[key]:
                case StartLobbyRequest.__name__:
                    payload = parse_as(received, StartLobbyRequest)
                    await start_lobby(lobby.id, payload.bots)
                case _:
                    msg = {"id": "input_error", "message": "Unknown message"}
                    await ws.send(json.dumps(msg))
                    await ws.close()
                    return
        case JoinLobbyRequest.__name__:
            # parse message
            request = parse_as(data, JoinLobbyRequest)

            if request.lobby_id not in LOBBIES:
                msg = {"id": "input_error", "message": "Lobby does not exist"}
                await ws.send(json.dumps(msg))
                await ws.close()
                return

            lobby = LOBBIES[request.lobby_id]
            await lobby.add_player(ws)
            await ws.wait_closed()
        case _:
            logging.warning(f"Unknown message: {message}")
            msg = {"id": "input_error", "message": "Unknown message"}
            await ws.send(json.dumps(msg))
            await ws.close()
            return


async def start_lobby(lobby_id: str, bots: list[BotType]) -> None:
    lobby = LOBBIES[lobby_id]
    await lobby.run(RunConfig(bots))


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    asyncio.run(main())
    print("Server WebSocket stopped")
