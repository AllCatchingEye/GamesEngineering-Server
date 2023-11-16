import random

from controller.random_controller import RandomController
from controller.websocket_controller import WebsocketController
from logic.game import Game

rng = random.Random(1)
game: Game = Game(rng)
game.controllers = [
    WebsocketController(game.players[0], ),
    RandomController(game.players[1], rng),
    RandomController(game.players[2], rng),
    RandomController(game.players[3], rng),
]
game.run()
