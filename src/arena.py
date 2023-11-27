from dataclasses import dataclass
from typing import Callable

from controller.player_controller import PlayerController
from controller.random_controller import RandomController


@dataclass
class ArenaConfig:
    games: int = 1000
    rounds_per_game: int = 10
    rng_seed: int | None = None


class Arena:
    """Arena provides a battle ground for multiple AIs to battle, and evaluates their performance."""

    config: ArenaConfig

    def __init__(self, config: ArenaConfig = ArenaConfig()) -> None:
        self.config = config
        
    def add_bot(self, bot_creator: Callable[[], PlayerController]) -> None:
        """Provides a function to create a bot and add it to the arena."""
        

    def run(self) -> None:
        pass


if __name__ == "__main__":
    arena = Arena()
    arena.add_bot(RandomController)
    arena.add_bot(RandomController)
    arena.add_bot(RandomController)
    arena.add_bot(RandomController)
    arena.run()
    # TODO: eval