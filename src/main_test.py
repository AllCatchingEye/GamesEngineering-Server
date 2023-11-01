import random
import unittest

from controller.random_controller import RandomController
from logic.game import Game
from state.event import *
from state.gametypes import Gametype
from state.suits import Suit
from state.card import Card
from state.ranks import Rank

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

        cls.player0 = cls.game.players[0]
        cls.player1 = cls.game.players[1]
        cls.player2 = cls.game.players[2]
        cls.player3 = cls.game.players[3]

        cls.sut_player = cls.player0
        cls.sut = TestController(cls.player0, rng=cls.rng)

        cls.game.controllers = [
            cls.sut,
            TestController(cls.player1, rng=cls.rng),
            TestController(cls.player2, rng=cls.rng),
            TestController(cls.player3, rng=cls.rng),
        ]
        cls.game.run(1)

        print(cls.sut.event_history.get_events_of_type(PlayDecisionEvent))
        print(cls.sut.event_history.get_events_of_type(CardPlayedEvent)[0:4])

    def test_game_start(self):
        self.assertEqual(
            len(self.sut.event_history.get_events_of_type(GameStartEvent)), 1
        )

    def test_play_decision(self):
        play_decisions = self.sut.event_history.get_events_of_type(PlayDecisionEvent)
        self.assertEqual(len(play_decisions), 4)

        self.assertEqual(play_decisions[0].player, self.sut_player)
        self.assertEqual(play_decisions[0].wants_to_play, False)

        self.assertEqual(play_decisions[1].player, self.player1)
        self.assertEqual(play_decisions[1].wants_to_play, False)

        self.assertEqual(play_decisions[2].player, self.player2)
        self.assertEqual(play_decisions[2].wants_to_play, False)

        self.assertEqual(play_decisions[3].player, self.player3)
        self.assertEqual(play_decisions[3].wants_to_play, True)

    def test_gametype_wish(self):
        wish_events = self.sut.event_history.get_events_of_type(GametypeWishedEvent)
        self.assertEqual(len(wish_events), 1)
        self.assertEqual(wish_events[0].player, self.player3)

    def test_gametype_determined(self):
        determined_events = self.sut.event_history.get_events_of_type(GametypeDeterminedEvent)
        self.assertEqual(len(determined_events), 1)
        self.assertEqual(determined_events[0].player, self.player3)
        self.assertEqual(determined_events[0].gametype, Gametype.SOLO)

    def test_game_end(self):
        end_events = self.sut.event_history.get_events_of_type(GameEndEvent)

        self.assertEqual(len(end_events), 1)

        self.assertEqual(end_events[0].winner, self.player3)
        self.assertEqual(end_events[0].points, 39)
    
    def test_all_rounds(self):
        self.assertEqual(len(self.sut.event_history.get_events_of_type(CardPlayedEvent)), 32)
        self.assertEqual(len(self.sut.event_history.get_events_of_type(RoundResultEvent)), 8)

    def test_round_1(self):
        player_turn_event = self.sut.event_history.get_events_of_type(PlayerTurnEvent)[0]
        cards_played = self.sut.event_history.get_events_of_type(CardPlayedEvent)[0:4]
        round_result = self.sut.event_history.get_events_of_type(RoundResultEvent)[0]


        # sut players turn event
        self.assertEqual(player_turn_event[0].layable_cards, list([Card(Suit.HEARTS, Rank.SEVEN)]))

        # cards played
        self.assertEqual(cards_played[0].player, self.sut_player)
        self.assertEqual(cards_played[0].card, Card(Suit.HEARTS, Rank.SEVEN))
        
        self.assertEqual(cards_played[1].player, self.player1)
        self.assertEqual(cards_played[1].card, Card(Suit.HEARTS, Rank.SEVEN))

        self.assertEqual(cards_played[2].player, self.player2)
        self.assertEqual(cards_played[2].card, Card(Suit.HEARTS, Rank.SEVEN))
        
        self.assertEqual(cards_played[3].player, self.player3)
        self.assertEqual(cards_played[3].card, Card(Suit.HEARTS, Rank.SEVEN))

        # round result
        self.assertEqual(round_result[0].round_winner, self.player3)
        self.assertEqual(round_result[0].points, 39)

class TestController(RandomController):
    event_history: EventList

    def __init__(self, player: Player, rng: random.Random):
        super().__init__(player, rng)
        self.event_history = EventList()

    def on_game_event(self, event: Event) -> None:
        self.event_history.events.append(event)


if __name__ == "__main__":
    unittest.main()
