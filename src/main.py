import asyncio
import random

from controller.ai_controller import AiController
from controller.random_controller import RandomController
from logic.game import Game

rng = random.Random(5)
game: Game = Game(rng)
game.controllers = [
    TerminalController(),
    AiController(),
    RandomController(rng),
    RandomController(rng),
]
asyncio.run(game.run(games_to_play=20))
