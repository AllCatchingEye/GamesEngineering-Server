import random
import unittest

from controller.random_controller import RandomController
from logic.game import Game
from state.event import *

T = TypeVar("T", bound=Event)


class EventList:
    events: list[Event]

    def __init__(self):
        self.events = []

    def get_events_of_type(self, event_type: Type[T]) -> list[T]:
        return [event for event in self.events if isinstance(event, event_type)]


class TestClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.test_seed = 42
        cls.rng = random.Random(cls.test_seed)
        cls.game: Game = Game(rng=cls.rng)
        cls.sut = TestController(cls.game.players[0], rng=cls.rng)
        cls.game.controllers = [
            cls.sut,
            TestController(cls.game.players[1], rng=cls.rng),
            TestController(cls.game.players[2], rng=cls.rng),
            TestController(cls.game.players[3], rng=cls.rng),
        ]
        cls.game.run(1)

    def test_game_start(self):
        self.assertEqual(
            len(self.sut.event_history.get_events_of_type(GameStartEvent)), 1
        )

    # def test_play_decision(self):
    #     self.assertEqual(len(self.get_events(EventType.PLAY_DECISION)), 4)
    #     self.assertEqual(
    #         self.get_event(EventType.PLAY_DECISION, 0).splurge()["wants_to_play"], False
    #     )
    #     self.assertEqual(
    #         self.get_event(EventType.PLAY_DECISION, 1).splurge()["wants_to_play"], False
    #     )
    #     self.assertEqual(
    #         self.get_event(EventType.PLAY_DECISION, 2).splurge()["wants_to_play"], False
    #     )
    #     self.assertEqual(
    #         self.get_event(EventType.PLAY_DECISION, 3).splurge()["wants_to_play"], True
    #     )

    # def test_gametype_wish(self):
    #     self.assertEqual(len(self.get_events(EventType.GAMETYPE_WISH)), 1)
    #     self.assertEqual(
    #         self.get_event(EventType.GAMETYPE_WISH, 0).splurge()["gametype"],
    #         Gametype.SOLO,
    #     )
    #     self.assertEqual(
    #         self.get_event(EventType.GAMETYPE_WISH, 0).splurge()["player"],
    #         self.game.players[2],
    #     )

    # def test_gametype_determined(self):
    #     self.assertEqual(len(self.get_events(EventType.GAMETYPE_DETERMINED)), 1)
    #     self.assertEqual(
    #         self.get_event(EventType.GAMETYPE_DETERMINED, 0).splurge()["gametype"],
    #         Gametype.SOLO,
    #     )
    #     self.assertEqual(
    #         self.get_event(EventType.GAMETYPE_DETERMINED, 0).splurge()["player"],
    #         self.game.players[2],
    #     )

    # def test_all_rounds(self):
    #     self.assertEqual(len(self.get_events(EventType.CARD_PLAYED)), 32)
    #     self.assertEqual(len(self.get_events(EventType.ROUND_RESULT)), 8)

    # def test_round_1(self):  # TODO: this seems wrong
    #     print(self.sut.sorted_events[EventType.CARD_PLAYED][0:4])
    #     print(self.sut.sorted_events[EventType.ROUND_RESULT][0])


class TestController(RandomController):
    event_history: EventList

    def __init__(self, player: Player, rng: random.Random):
        super().__init__(player, rng)
        self.event_history = EventList()

    def on_game_event(self, event: Event) -> None:
        self.event_history.events.append(event)


if __name__ == "__main__":
    unittest.main()
