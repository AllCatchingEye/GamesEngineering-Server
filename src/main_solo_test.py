import random
import unittest
from typing import Type, TypeVar

from controller.random_controller import RandomController
from logic.game import Game
from state.card import Card
from state.event import *
from state.gametypes import Gametype
from state.ranks import Rank
from state.stack import PlayedCard, Stack
from state.suits import Suit

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

        cls.sut = TestController(cls.player0, rng=cls.rng)

        cls.game.controllers = [
            cls.sut,
            TestController(cls.player1, rng=cls.rng),
            TestController(cls.player2, rng=cls.rng),
            TestController(cls.player3, rng=cls.rng),
        ]
        cls.game.run()

    def test_game_start(self):
        self.assertEqual(
            len(self.sut.event_history.get_events_of_type(GameStartEvent)), 1
        )

    def test_play_decision(self):
        play_decisions = self.sut.event_history.get_events_of_type(PlayDecisionEvent)
        self.assertEqual(len(play_decisions), 4)

        self.assertEqual(play_decisions[0].player, self.player0)
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
        determined_events = self.sut.event_history.get_events_of_type(
            GametypeDeterminedEvent
        )
        self.assertEqual(len(determined_events), 1)
        self.assertEqual(determined_events[0].player, self.player3)
        self.assertEqual(determined_events[0].suit, Suit.SCHELLEN)
        self.assertEqual(determined_events[0].gametype, Gametype.SOLO)

    def test_game_end(self):
        end_events = self.sut.event_history.get_events_of_type(GameEndEvent)

        self.assertEqual(len(end_events), 1)

        self.assertEqual(end_events[0].winner, self.player1)
        self.assertEqual(end_events[0].points, 44)

    def test_all_rounds(self):
        self.assertEqual(
            len(self.sut.event_history.get_events_of_type(CardPlayedEvent)), 32
        )
        self.assertEqual(
            len(self.sut.event_history.get_events_of_type(RoundResultEvent)), 8
        )

    def test_round_1(self):
        r = 1
        playable_cards, stack = self.sut.player_turns[r - 1]
        cards_played = self.sut.event_history.get_events_of_type(CardPlayedEvent)[
            4 * (r - 1) : 4 * (r - 1) + 4
        ]
        round_result = self.sut.event_history.get_events_of_type(RoundResultEvent)[
            r - 1
        ]

        # player had first turn
        self.assertEqual(len(playable_cards), 8)
        self.assertEqual(len(stack), 0)

        # played cards
        self.assertEqual(cards_played[0].player, self.player0)
        self.assertEqual(cards_played[0].card, Card(Suit.GRAS, Rank.OBER))

        self.assertEqual(cards_played[1].player, self.player1)
        self.assertEqual(cards_played[1].card, Card(Suit.HERZ, Rank.OBER))

        self.assertEqual(cards_played[2].player, self.player2)
        self.assertEqual(cards_played[2].card, Card(Suit.SCHELLEN, Rank.ACHT))

        self.assertEqual(cards_played[3].player, self.player3)
        self.assertEqual(cards_played[3].card, Card(Suit.SCHELLEN, Rank.SIEBEN))

        # round result
        self.assertEqual(round_result.points, 6)
        self.assertEqual(
            round_result.stack.get_first_card(), Card(Suit.GRAS, Rank.OBER)
        )
        self.assertTrue(
            round_result.stack.get_played_cards(),
            [
                PlayedCard(Card(Suit.GRAS, Rank.OBER), self.player0),
                PlayedCard(Card(Suit.HERZ, Rank.OBER), self.player1),
                PlayedCard(Card(Suit.EICHEL, Rank.ZEHN), self.player2),
                PlayedCard(Card(Suit.EICHEL, Rank.SIEBEN), self.player3),
            ],
        )
        self.assertEqual(round_result.round_winner, self.player0)

    def test_round_2(self):
        r = 2
        playable_cards, stack = self.sut.player_turns[r - 1]
        cards_played = self.sut.event_history.get_events_of_type(CardPlayedEvent)[
            4 * (r - 1) : 4 * (r - 1) + 4
        ]
        round_result = self.sut.event_history.get_events_of_type(RoundResultEvent)[
            r - 1
        ]

        # player had first turn, order did not change
        self.assertEqual(len(playable_cards), 7)
        self.assertEqual(len(stack), 0)

        # played cards
        self.assertEqual(cards_played[0].player, self.player0)
        self.assertEqual(
            cards_played[0].card,
            Card(
                Suit.SCHELLEN,
                Rank.KOENIG,
            ),
        )

        self.assertEqual(cards_played[1].player, self.player1)
        self.assertEqual(cards_played[1].card, Card(Suit.SCHELLEN, Rank.ASS))

        self.assertEqual(cards_played[2].player, self.player2)
        self.assertEqual(cards_played[2].card, Card(Suit.EICHEL, Rank.OBER))

        self.assertEqual(cards_played[3].player, self.player3)
        self.assertEqual(cards_played[3].card, Card(Suit.SCHELLEN, Rank.OBER))

        # round result
        self.assertEqual(round_result.points, 21)
        self.assertEqual(
            round_result.stack.get_first_card(), Card(Suit.SCHELLEN, Rank.KOENIG)
        )
        self.assertTrue(
            round_result.stack.get_played_cards(),
            [
                PlayedCard(Card(Suit.SCHELLEN, Rank.KOENIG), self.player0),
                PlayedCard(Card(Suit.SCHELLEN, Rank.ASS), self.player1),
                PlayedCard(Card(Suit.SCHELLEN, Rank.ACHT), self.player2),
                PlayedCard(Card(Suit.SCHELLEN, Rank.SIEBEN), self.player3),
            ],
        )
        self.assertEqual(round_result.round_winner, self.player2)

    def test_round_3(self):
        r = 3
        playable_cards, stack = self.sut.player_turns[r - 1]
        cards_played = self.sut.event_history.get_events_of_type(CardPlayedEvent)[
            4 * (r - 1) : 4 * (r - 1) + 4
        ]
        round_result = self.sut.event_history.get_events_of_type(RoundResultEvent)[
            r - 1
        ]

        # player had second last turn, order changed
        self.assertEqual(len(playable_cards), 1)
        self.assertEqual(len(stack), 2)

        # played cards
        self.assertEqual(cards_played[0].player, self.player2)
        self.assertEqual(cards_played[0].card, Card(Suit.HERZ, Rank.KOENIG))

        self.assertEqual(cards_played[1].player, self.player3)
        self.assertEqual(cards_played[1].card, Card(Suit.HERZ, Rank.SIEBEN))

        self.assertEqual(cards_played[2].player, self.player0)
        self.assertEqual(cards_played[2].card, Card(Suit.HERZ, Rank.ACHT))

        self.assertEqual(cards_played[3].player, self.player1)
        self.assertEqual(cards_played[3].card, Card(Suit.HERZ, Rank.ASS))

        # round result
        self.assertEqual(round_result.points, 15)
        self.assertEqual(
            round_result.stack.get_first_card(), Card(Suit.HERZ, Rank.KOENIG)
        )
        self.assertTrue(
            round_result.stack.get_played_cards(),
            [
                PlayedCard(Card(Suit.GRAS, Rank.KOENIG), self.player0),
                PlayedCard(Card(Suit.EICHEL, Rank.ASS), self.player1),
                PlayedCard(Card(Suit.EICHEL, Rank.ACHT), self.player2),
                PlayedCard(Card(Suit.EICHEL, Rank.SIEBEN), self.player3),
            ],
        )
        self.assertEqual(round_result.round_winner, self.player1)


class TestController(RandomController):
    event_history: EventList
    player_turns: list[tuple[list[Card], list[PlayedCard]]]

    def __init__(self, player: Player, rng: random.Random):
        super().__init__(player, rng)
        self.event_history = EventList()
        self.player_turns = []

    def on_game_event(self, event: Event) -> None:
        self.event_history.events.append(event)

    def play_card(self, stack: Stack, playable_cards: list[Card]) -> Card:
        self.player_turns.append(
            (playable_cards.copy(), stack.get_played_cards().copy())
        )
        return super().play_card(stack, playable_cards)


if __name__ == "__main__":
    unittest.main()
