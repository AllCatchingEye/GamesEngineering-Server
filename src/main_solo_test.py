import asyncio
import random
import unittest

from controller.random_controller import RandomController
from logic.game import Game
from state.card import Card
from state.event import *
from state.gametypes import Gametype
from state.ranks import Rank
from state.stack import PlayedCard, Stack
from state.suits import Suit
from state.player import Player

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
        asyncio.run(cls.game.run())

    def test_game_start(self):
        gameStartEvent = self.sut.event_history.get_events_of_type(GameStartUpdate)
        self.assertEqual(len(gameStartEvent), 1)
        self.assertEqual(gameStartEvent[0].player, self.player0.id)
        self.assertEqual(len(gameStartEvent[0].hand), 8)

    def test_play_decision(self):
        play_decisions = self.sut.event_history.get_events_of_type(PlayDecisionUpdate)
        self.assertEqual(len(play_decisions), 4)

        self.assertEqual(play_decisions[0].player, self.player0.id)
        self.assertEqual(play_decisions[0].wants_to_play, False)

        self.assertEqual(play_decisions[1].player, self.player1.id)
        self.assertEqual(play_decisions[1].wants_to_play, False)

        self.assertEqual(play_decisions[2].player, self.player2.id)
        self.assertEqual(play_decisions[2].wants_to_play, False)

        self.assertEqual(play_decisions[3].player, self.player3.id)
        self.assertEqual(play_decisions[3].wants_to_play, True)

    def test_gamegroupChosen(self):
        wish_events = self.sut.event_history.get_events_of_type(GameGroupChosenUpdate)
        self.assertEqual(len(wish_events), 0)

    def test_gametype_determined(self):
        determined_events = self.sut.event_history.get_events_of_type(
            GametypeDeterminedUpdate
        )
        self.assertEqual(len(determined_events), 1)
        self.assertEqual(determined_events[0].player, self.player3.id)
        self.assertEqual(determined_events[0].suit, Suit.SCHELLEN)
        self.assertEqual(determined_events[0].gametype, Gametype.SOLO)

    def test_game_end(self):
        end_events = self.sut.event_history.get_events_of_type(GameEndUpdate)
        money_pay_event = self.sut.event_history.get_events_of_type(MoneyUpdate)

        self.assertEqual(len(end_events), 1)
        self.assertEqual(len(money_pay_event), 1)

        self.assertEqual(
            end_events[0].winner, [self.player0.id, self.player1.id, self.player2.id]
        )
        self.assertEqual(end_events[0].points, [31, 89])
        self.assertEqual(money_pay_event[0].player, self.player0.id)
        self.assertEqual(money_pay_event[0].money.cent, 500)

    def test_all_rounds(self):
        self.assertEqual(
            len(self.sut.event_history.get_events_of_type(CardPlayedUpdate)), 32
        )
        self.assertEqual(
            len(self.sut.event_history.get_events_of_type(RoundResultUpdate)), 8
        )

    def test_round_1(self):
        r = 1
        playable_cards, stack = self.sut.player_turns[r - 1]
        play_order = self.sut.event_history.get_events_of_type(PlayOrderUpdate)[r-1]
        cards_played = self.sut.event_history.get_events_of_type(CardPlayedUpdate)[
            4 * (r - 1) : 4 * (r - 1) + 4
        ]
        round_result = self.sut.event_history.get_events_of_type(RoundResultUpdate)[
            r - 1
        ]

        # player had first turn
        self.assertTrue(play_order is not None)
        self.assertEqual(play_order.order, [self.player0.id, self.player1.id, self.player2.id, self.player3.id])
        self.assertEqual(len(playable_cards), 8)
        self.assertEqual(len(stack), 0)

        # played cards
        self.assertEqual(cards_played[0].player, self.player0.id)
        self.assertEqual(cards_played[0].card, Card(Suit.GRAS, Rank.OBER))

        self.assertEqual(cards_played[1].player, self.player1.id)
        self.assertEqual(cards_played[1].card, Card(Suit.HERZ, Rank.OBER))

        self.assertEqual(cards_played[2].player, self.player2.id)
        self.assertEqual(cards_played[2].card, Card(Suit.SCHELLEN, Rank.ACHT))

        self.assertEqual(cards_played[3].player, self.player3.id)
        self.assertEqual(cards_played[3].card, Card(Suit.SCHELLEN, Rank.SIEBEN))

        # round result
        self.assertEqual(round_result.points, 6)
        self.assertEqual(round_result.round_winner, self.player0.id)

    def test_round_2(self):
        r = 2
        playable_cards, stack = self.sut.player_turns[r - 1]
        play_order = self.sut.event_history.get_events_of_type(PlayOrderUpdate)[r-1]
        cards_played = self.sut.event_history.get_events_of_type(CardPlayedUpdate)[
            4 * (r - 1) : 4 * (r - 1) + 4
        ]
        round_result = self.sut.event_history.get_events_of_type(RoundResultUpdate)[
            r - 1
        ]

        # player had first turn, order did not change
        self.assertTrue(play_order is not None)
        self.assertEqual(play_order.order, [self.player0.id, self.player1.id, self.player2.id, self.player3.id])
        self.assertEqual(len(playable_cards), 7)
        self.assertEqual(len(stack), 0)

        # played cards
        self.assertEqual(cards_played[0].player, self.player0.id)
        self.assertEqual(
            cards_played[0].card,
            Card(
                Suit.SCHELLEN,
                Rank.KOENIG,
            ),
        )

        self.assertEqual(cards_played[1].player, self.player1.id)
        self.assertEqual(cards_played[1].card, Card(Suit.SCHELLEN, Rank.ASS))

        self.assertEqual(cards_played[2].player, self.player2.id)
        self.assertEqual(cards_played[2].card, Card(Suit.EICHEL, Rank.OBER))

        self.assertEqual(cards_played[3].player, self.player3.id)
        self.assertEqual(cards_played[3].card, Card(Suit.SCHELLEN, Rank.OBER))

        # round result
        self.assertEqual(round_result.points, 21)
        self.assertEqual(round_result.round_winner, self.player2.id)

    def test_round_3(self):
        r = 3
        playable_cards, stack = self.sut.player_turns[r - 1]
        play_order = self.sut.event_history.get_events_of_type(PlayOrderUpdate)[r-1]
        cards_played = self.sut.event_history.get_events_of_type(CardPlayedUpdate)[
            4 * (r - 1) : 4 * (r - 1) + 4
        ]
        round_result = self.sut.event_history.get_events_of_type(RoundResultUpdate)[
            r - 1
        ]

        # player had second last turn, order changed
        self.assertTrue(play_order is not None)
        self.assertEqual(play_order.order, [self.player2.id, self.player3.id, self.player0.id, self.player1.id])
        self.assertEqual(len(playable_cards), 1)
        self.assertEqual(len(stack), 2)

        # played cards
        self.assertEqual(cards_played[0].player, self.player2.id)
        self.assertEqual(cards_played[0].card, Card(Suit.HERZ, Rank.KOENIG))

        self.assertEqual(cards_played[1].player, self.player3.id)
        self.assertEqual(cards_played[1].card, Card(Suit.HERZ, Rank.SIEBEN))

        self.assertEqual(cards_played[2].player, self.player0.id)
        self.assertEqual(cards_played[2].card, Card(Suit.HERZ, Rank.ACHT))

        self.assertEqual(cards_played[3].player, self.player1.id)
        self.assertEqual(cards_played[3].card, Card(Suit.HERZ, Rank.ASS))

        # round result
        self.assertEqual(round_result.points, 15)
        self.assertEqual(round_result.round_winner, self.player1.id)


class TestController(RandomController):
    event_history: EventList
    player_turns: list[tuple[list[Card], list[PlayedCard]]]

    def __init__(self, player: Player, rng: random.Random):
        super().__init__(player, rng)
        self.event_history = EventList()
        self.player_turns = []

    async def on_game_event(self, event: Event) -> None:
        self.event_history.events.append(event)

    async def play_card(self, stack: Stack, playable_cards: list[Card]) -> Card:
        self.player_turns.append(
            (playable_cards.copy(), stack.get_played_cards().copy())
        )
        return await super().play_card(stack, playable_cards)


if __name__ == "__main__":
    unittest.main()
