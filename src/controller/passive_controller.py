import random
from controller.random_controller import RandomController

from state.gametypes import GameGroup


class PassiveController(RandomController):
    def __init__(self, rng: random.Random = random.Random()):
        super().__init__(rng)

    async def wants_to_play(self, current_lowest_gamegroup: GameGroup) -> bool:
        return False
