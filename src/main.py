from controller.random_controller import RandomController
from controller.terminal_controller import TerminalController
from logic.game import Game

game: Game = Game()
game.controllers = [
    TerminalController(game.players[0]),
    RandomController(game.players[1]),
    RandomController(game.players[2]),
    RandomController(game.players[3]),
]
game.run()
