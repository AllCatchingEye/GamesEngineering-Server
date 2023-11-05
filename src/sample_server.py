import asyncio

# import logging
from websockets.server import WebSocketServerProtocol, serve


async def handler(websocket: WebSocketServerProtocol) -> None:
    async for message in websocket:
        print(f"Server received: {message}")
        print(f"Server sends back message: {message}")
        await websocket.send(message)


async def main() -> None:
    print("Websocket starts")
    async with serve(handler, "localhost", 8765):
        print("Websocket runs")
        await asyncio.Future()  # run forever


def get_gamestate():
    pass


# logger = logging.getLogger('websockets')
# logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.StreamHandler)
asyncio.run(main())
