import unittest

from ai.nn_helper import decode_game_type, get_one_hot_encoding_index_from_card
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
