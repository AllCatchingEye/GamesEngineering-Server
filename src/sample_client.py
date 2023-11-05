# import asyncio
import json
from websockets.sync.client import connect


def hello() -> None:
    with connect("ws://localhost:8765") as websocket:

        # Text communication example
        websocket.send("Hello World")
        message = websocket.recv()
        print(f"Client Received: {message}")

        # Binary communication example
        binaryData: bytes = int.to_bytes(6)
        websocket.send(binaryData)
        binary = websocket.recv()
        print(f"Client Received binary: {binary}")

        # Json communication example
        dict = {
            "name": "Json test",
        }
        json_object = json.dumps(dict)
        websocket.send(json_object)
        json_object = websocket.recv()
        print(f"Client Received json: {json_object}")


hello()
