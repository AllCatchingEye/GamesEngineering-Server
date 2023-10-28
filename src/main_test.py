import random
from typing import override

from controller.random_controller import RandomController
from logic.game import Game
from state.event import *

class TestController(RandomController):
    sorted_events: dict[EventType, list[Event]] = {}
    event_history: list[Event] = []
    
    @override
    def on_game_event(self, event: Event) -> None:
        self.event_history.append(event)
        
        if event.event_type() not in self.sorted_events:
            self.sorted_events[event.event_type()] = []
        self.sorted_events[event.event_type()].append(event)

test_seed = 42
rng = random.Random(test_seed)
game: Game = Game(rng=rng)
sut = TestController(game.players[0], rng=rng)
game.controllers = [
    sut,
    TestController(game.players[1], rng=rng),
    TestController(game.players[2], rng=rng),
    TestController(game.players[3], rng=rng),
]

game.run(1)

assert len(sut.event_history) > 0

print(sut.sorted_events[EventType.GAME_START])
