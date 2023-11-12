Options:

### Websocket Library von Python

#### Pro

- Simpel in der Umsetzung

#### Contra

- Begrenzte Features

Installation:

- `pip install websockets`

Example Code:

```python
import asyncio
import websockets
async def echo(websocket, path):
async for message in websocket:
await websocket.send(message)

start_server = websockets.serve(echo, "localhost", 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
```

### Tornado

#### Pro

- Vorteil: Simple Umsetzung

#### Contra

- Nachteil: Begrenzte Features

Installation:

- `pip install tornado`

Example Code:

```python
import tornado.ioloop
import tornado.web
import tornado.websocket

class WebSocketHandler(tornado.websocket.WebSocketHandler):
def on_message(self, message):
self.write_message("You said: " + message)

application = tornado.web.Application([
(r"/websocket", WebSocketHandler),
])

if **name** == "**main**":
application.listen(8888)
tornado.ioloop.IOLoop.current().start()
```

### FastAPI

#### Pro

- Integriert data validation
- Sehr schnell
- Evtl. für Deep Learning am besten

#### Contra

Installation:

- `pip install fastapi uvicorn`

Example Code:

```python
from fastapi import FastAPI
from fastapi.websockets import WebSocket

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send_text(f"Message text was: {data}")
```

```

```

```

```
