import asyncio
import random

from controller.ai_controller import AiController
from controller.random_controller import RandomController
from controller.terminal_controller import TerminalController
from logic.game import Game
from controller.handcrafted_controller import HandcraftedController
from controller.passive_controller import PassiveController

rng = random.Random(5)
game: Game = Game(rng)
game.controllers = [
    TerminalController(),
    HandcraftedController(),
    HandcraftedController(),
    HandcraftedController(),
]
asyncio.run(game.run(games_to_play=20))
