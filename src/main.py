import random

from controller.random_controller import RandomController
from controller.terminal_controller import TerminalController
from logic.game import Game
from state.money import Money

m1 = Money(10, 50)
m2 = Money(0, 90)
m3 = m1 + m2
m4 = m1 - m2
m5 = Money(1, 0) + Money(-1, -20)

rng = random.Random(1)
game: Game = Game(rng)
game.controllers = [
    TerminalController(game.players[0]),
    RandomController(game.players[1], rng),
    RandomController(game.players[2], rng),
    RandomController(game.players[3], rng),
]
game.run()
