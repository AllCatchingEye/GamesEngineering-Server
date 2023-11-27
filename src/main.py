import asyncio
import random
from controller.ai_controller import AiController

from controller.random_controller import RandomController
from controller.terminal_controller import TerminalController
from logic.game import Game

rng = random.Random(1)
game: Game = Game(rng)
game.controllers = [
    TerminalController(),
    AiController(),
    RandomController(rng),
    RandomController(rng),
]
asyncio.run(game.run(games_to_play=20))
