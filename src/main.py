from controller.terminal_controller import TerminalController
from logic.game import Game

game: Game = Game()
game.controllers = [
    TerminalController(game.players[0]),
    TerminalController(game.players[1]),
    TerminalController(game.players[2]),
    TerminalController(game.players[3]),
]
game.run()
