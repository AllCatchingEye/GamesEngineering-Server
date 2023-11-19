import asyncio
import time

from controller.random_controller import RandomController
from logic.game import Game


async def main() -> None:
    i = 1000

    now = time.time()

    for _ in range(i):
        game: Game = Game()
        game.controllers = [
            RandomController(game.players[0]),
            RandomController(game.players[1]),
            RandomController(game.players[2]),
            RandomController(game.players[3]),
        ]
        await game.run()

    print(f"Time for {i} games: {time.time() - now} seconds")
    print(f"Average time per game: {(time.time() - now) / i} seconds")


if __name__ == "__main__":
    asyncio.run(main())
