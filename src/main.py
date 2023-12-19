import asyncio
import random

from controller.ai_controller import AiController
from controller.combi_contoller import CombiController
from controller.handcrafted_controller import HandcraftedController
from controller.random_controller import RandomController
from controller.terminal_controller import TerminalController
from logic.game import Game

rng = random.Random(5)
game: Game = Game(rng)
game.controllers = [
    TerminalController(),
    CombiController(AiController(), HandcraftedController()),
    RandomController(),
    AiController(),
]
asyncio.run(game.run(games_to_play=20))
