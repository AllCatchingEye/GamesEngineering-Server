import unittest

from ai.nn_helper import (
    card_to_nn_input_values_index,
    code_to_game_type,
    nn_output_code_to_card,
)
from state.card import Card
from state.gametypes import Gametype
from state.ranks import Rank
from state.suits import Suit


class TestClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def test_nn_input_index(self):
        self.assertEqual(card_to_nn_input_values_index(Card(Suit.EICHEL, Rank.OBER)), 0)
        self.assertEqual(
            card_to_nn_input_values_index(Card(Suit.EICHEL, Rank.UNTER)), 1
        )
        self.assertEqual(card_to_nn_input_values_index(Card(Suit.EICHEL, Rank.ASS)), 2)
        self.assertEqual(card_to_nn_input_values_index(Card(Suit.EICHEL, Rank.ZEHN)), 3)
        self.assertEqual(
            card_to_nn_input_values_index(Card(Suit.EICHEL, Rank.KOENIG)),
            4,
        )
        self.assertEqual(card_to_nn_input_values_index(Card(Suit.EICHEL, Rank.NEUN)), 5)
        self.assertEqual(card_to_nn_input_values_index(Card(Suit.EICHEL, Rank.ACHT)), 6)
        self.assertEqual(
            card_to_nn_input_values_index(Card(Suit.EICHEL, Rank.SIEBEN)),
            7,
        )
        self.assertEqual(card_to_nn_input_values_index(Card(Suit.GRAS, Rank.OBER)), 8)
        self.assertEqual(
            card_to_nn_input_values_index(Card(Suit.GRAS, Rank.SIEBEN)),
            8 + 7,
        )
        self.assertEqual(
            card_to_nn_input_values_index(Card(Suit.HERZ, Rank.OBER)),
            8 * 2,
        )
        self.assertEqual(
            card_to_nn_input_values_index(Card(Suit.HERZ, Rank.SIEBEN)),
            8 * 2 + 7,
        )
        self.assertEqual(
            card_to_nn_input_values_index(Card(Suit.SCHELLEN, Rank.OBER)),
            8 * 3,
        )
        self.assertEqual(
            card_to_nn_input_values_index(Card(Suit.SCHELLEN, Rank.SIEBEN)),
            8 * 3 + 7,
        )

    def test_nn_game_type_mapping(self):
        mapping = {
            "Farbgeier Eichel": Gametype.FARBGEIER,
            "Farbgeier Gras": Gametype.FARBGEIER,
            "Farbgeier Herz": Gametype.FARBGEIER,
            "Farbgeier Schelle": Gametype.FARBGEIER,
            "Farbwenz Eichel": Gametype.FARBWENZ,
            "Farbwenz Gras": Gametype.FARBWENZ,
            "Farbwenz Herz": Gametype.FARBWENZ,
            "Farbwenz Schelle": Gametype.FARBWENZ,
            "Geier": Gametype.GEIER,
            "Sauspiel Alte": Gametype.SAUSPIEL,
            "Sauspiel Blaue": Gametype.SAUSPIEL,
            "Sauspiel Hundsgfickte": Gametype.SAUSPIEL,
            "Solo Eichel": Gametype.SOLO,
            "Solo Gras": Gametype.SOLO,
            "Solo Herz": Gametype.SOLO,
            "Solo Schelle": Gametype.SOLO,
            "Wenz": Gametype.WENZ,
            "weiter": Gametype.RAMSCH,
        }

        for index, game_type in enumerate(mapping.values()):
            self.assertEqual(code_to_game_type(index).value, game_type.value)
    
    def __assert_card_equal(self, card: Card, target: Card):
        self.assertEqual(card.suit, target.suit)
        self.assertEqual(card.rank, target.rank)


    def test_nn_output_code_to_card(self):
        self.__assert_card_equal(nn_output_code_to_card(0), Card(Suit.EICHEL, Rank.OBER))
        self.__assert_card_equal(nn_output_code_to_card(1), Card(Suit.EICHEL, Rank.UNTER))
        self.__assert_card_equal(nn_output_code_to_card(7), Card(Suit.EICHEL, Rank.SIEBEN))
        self.__assert_card_equal(nn_output_code_to_card(8), Card(Suit.GRAS, Rank.OBER))
        self.__assert_card_equal(nn_output_code_to_card(8 + 7), Card(Suit.GRAS, Rank.SIEBEN))
        self.__assert_card_equal(nn_output_code_to_card(2 * 8), Card(Suit.HERZ, Rank.OBER))
        self.__assert_card_equal(nn_output_code_to_card(2 * 8 + 7), Card(Suit.HERZ, Rank.SIEBEN))
        self.__assert_card_equal(nn_output_code_to_card(3 * 8), Card(Suit.SCHELLEN, Rank.OBER))
        self.__assert_card_equal(nn_output_code_to_card(3 * 8 + 7), Card(Suit.SCHELLEN, Rank.SIEBEN))