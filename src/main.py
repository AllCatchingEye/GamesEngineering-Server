import asyncio
import random

from controller.random_controller import RandomController
from controller.terminal_controller import TerminalController
from logic.game import Game

rng = random.Random(1)
game: Game = Game(rng)
game.controllers = [
    TerminalController(game.players[0]),
    RandomController(game.players[1], rng),
    RandomController(game.players[2], rng),
    RandomController(game.players[3], rng),
]
asyncio.run(game.run(games_to_play=20))
