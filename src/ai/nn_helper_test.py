import unittest

from ai.nn_helper import (
    NUM_CARDS,
    NUM_RANKS,
    decode_game_type,
    get_one_hot_encoding_index_from_card,
    one_hot_encode_card,
    one_hot_encode_cards,
)
from state.card import Card
from state.gametypes import Gametype
from state.ranks import Rank
from state.suits import Suit

CARD_ACTION_MAPPING = {
    0: Card(Suit.EICHEL, Rank.OBER),
    1: Card(Suit.EICHEL, Rank.UNTER),
    7: Card(Suit.EICHEL, Rank.SIEBEN),
    8: Card(Suit.GRAS, Rank.OBER),
    8 + 7: Card(Suit.GRAS, Rank.SIEBEN),
    (2 * 8): Card(Suit.HERZ, Rank.OBER),
    (2 * 8) + 7: Card(Suit.HERZ, Rank.SIEBEN),
    (3 * 8): Card(Suit.SCHELLEN, Rank.OBER),
    (3 * 8) + 7: Card(Suit.SCHELLEN, Rank.SIEBEN),
}


class TestClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def test_nn_game_type_mapping(self):
        result = decode_game_type(0)
        self.assertEqual(result[0], Gametype.FARBGEIER)
        self.assertEqual(result[1], Suit.EICHEL)
        result = decode_game_type(1)
        self.assertEqual(result[0], Gametype.FARBGEIER)
        self.assertEqual(result[1], Suit.GRAS)
        result = decode_game_type(2)
        self.assertEqual(result[0], Gametype.FARBGEIER)
        self.assertEqual(result[1], Suit.HERZ)
        result = decode_game_type(3)
        self.assertEqual(result[0], Gametype.FARBGEIER)
        self.assertEqual(result[1], Suit.SCHELLEN)

        result = decode_game_type(4 + 0)
        self.assertEqual(result[0], Gametype.FARBWENZ)
        self.assertEqual(result[1], Suit.EICHEL)
        result = decode_game_type(4 + 1)
        self.assertEqual(result[0], Gametype.FARBWENZ)
        self.assertEqual(result[1], Suit.GRAS)
        result = decode_game_type(4 + 2)
        self.assertEqual(result[0], Gametype.FARBWENZ)
        self.assertEqual(result[1], Suit.HERZ)
        result = decode_game_type(4 + 3)
        self.assertEqual(result[0], Gametype.FARBWENZ)
        self.assertEqual(result[1], Suit.SCHELLEN)

        result = decode_game_type(8)
        self.assertEqual(result[0], Gametype.GEIER)
        self.assertEqual(result[1], None)

        result = decode_game_type(9 + 0)
        self.assertEqual(result[0], Gametype.SAUSPIEL)
        self.assertEqual(result[1], Suit.EICHEL)
        result = decode_game_type(9 + 1)
        self.assertEqual(result[0], Gametype.SAUSPIEL)
        self.assertEqual(result[1], Suit.GRAS)
        result = decode_game_type(9 + 2)
        self.assertEqual(result[0], Gametype.SAUSPIEL)
        self.assertEqual(result[1], Suit.SCHELLEN)

        result = decode_game_type(12 + 0)
        self.assertEqual(result[0], Gametype.SOLO)
        self.assertEqual(result[1], Suit.EICHEL)
        result = decode_game_type(12 + 1)
        self.assertEqual(result[0], Gametype.SOLO)
        self.assertEqual(result[1], Suit.GRAS)
        result = decode_game_type(12 + 2)
        self.assertEqual(result[0], Gametype.SOLO)
        self.assertEqual(result[1], Suit.HERZ)
        result = decode_game_type(12 + 3)
        self.assertEqual(result[0], Gametype.SOLO)
        self.assertEqual(result[1], Suit.SCHELLEN)

        result = decode_game_type(16)
        self.assertEqual(result[0], Gametype.WENZ)
        self.assertEqual(result[1], None)

        result = decode_game_type(17)
        self.assertEqual(result[0], Gametype.RAMSCH)
        self.assertEqual(result[1], None)

    def test_card_to_action(self):
        for action, card in CARD_ACTION_MAPPING.items():
            self.assertEqual(get_one_hot_encoding_index_from_card(card), action)

    def test_one_hot_encode_card(self):
        plain_target = [0] * NUM_CARDS
        target = plain_target.copy()
        target[NUM_RANKS * 0 + 0] = 1
        self.assertListEqual(one_hot_encode_card(Card(Suit.EICHEL, Rank.OBER)), target)
        target = plain_target.copy()
        target[NUM_RANKS * 1 + 0] = 1
        self.assertListEqual(one_hot_encode_card(Card(Suit.GRAS, Rank.OBER)), target)
        target = plain_target.copy()
        target[NUM_RANKS * 2 + 0] = 1
        self.assertListEqual(one_hot_encode_card(Card(Suit.HERZ, Rank.OBER)), target)
        target = plain_target.copy()
        target[NUM_RANKS * 3 + 0] = 1
        self.assertListEqual(
            one_hot_encode_card(Card(Suit.SCHELLEN, Rank.OBER)), target
        )
        target = plain_target.copy()
        target[NUM_RANKS * 0 + 1] = 1
        self.assertListEqual(one_hot_encode_card(Card(Suit.EICHEL, Rank.UNTER)), target)
        target = plain_target.copy()
        target[NUM_RANKS * 0 + 2] = 1
        self.assertListEqual(one_hot_encode_card(Card(Suit.EICHEL, Rank.ASS)), target)
        target = plain_target.copy()
        target[NUM_RANKS * 0 + 3] = 1
        self.assertListEqual(one_hot_encode_card(Card(Suit.EICHEL, Rank.ZEHN)), target)
        target = plain_target.copy()
        target[NUM_RANKS * 0 + 4] = 1
        self.assertListEqual(
            one_hot_encode_card(Card(Suit.EICHEL, Rank.KOENIG)), target
        )
        target = plain_target.copy()
        target[NUM_RANKS * 0 + 5] = 1
        self.assertListEqual(one_hot_encode_card(Card(Suit.EICHEL, Rank.NEUN)), target)
        target = plain_target.copy()
        target[NUM_RANKS * 0 + 6] = 1
        self.assertListEqual(one_hot_encode_card(Card(Suit.EICHEL, Rank.ACHT)), target)
        target = plain_target.copy()
        target[NUM_RANKS * 0 + 7] = 1
        self.assertListEqual(
            one_hot_encode_card(Card(Suit.EICHEL, Rank.SIEBEN)), target
        )

    def test_one_hot_encode_cards(self):
        plain_target = [0] * NUM_CARDS
        target = plain_target
        self.assertListEqual(one_hot_encode_cards([]), target)
        target[NUM_RANKS * 0 + 0] = 1
        self.assertListEqual(
            one_hot_encode_cards([Card(Suit.EICHEL, Rank.OBER)]), target
        )
        target[NUM_RANKS * 0 + 1] = 1
        self.assertListEqual(
            one_hot_encode_cards(
                [Card(Suit.EICHEL, Rank.OBER), Card(Suit.EICHEL, Rank.UNTER)]
            ),
            target,
        )
        target[NUM_RANKS * 1 + 0] = 1
        self.assertListEqual(
            one_hot_encode_cards(
                [
                    Card(Suit.EICHEL, Rank.OBER),
                    Card(Suit.EICHEL, Rank.UNTER),
                    Card(Suit.GRAS, Rank.OBER),
                ]
            ),
            target,
        )
