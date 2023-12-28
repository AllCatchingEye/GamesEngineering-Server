import asyncio
import random

from controller.ai_controller import AiController
from controller.combi_contoller import CombiController
from controller.handcrafted_controller import HandcraftedController
from controller.random_controller import RandomController
from controller.terminal_controller import TerminalController
from logic.game import Game

def get_ai_ctrl_256_256_256_256_256() -> AiController:
    return AiController([256, 256, 256, 256, 256])

rng = random.Random(5)
game: Game = Game(rng)
game.controllers = [
    TerminalController(),
    CombiController(get_ai_ctrl_256_256_256_256_256(), HandcraftedController()),
    RandomController(),
    get_ai_ctrl_256_256_256_256_256(),
]
asyncio.run(game.run(games_to_play=20))
