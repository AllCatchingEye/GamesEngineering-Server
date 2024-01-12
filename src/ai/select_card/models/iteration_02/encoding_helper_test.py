import unittest

from ai.select_card.models.iteration_02.encoding_helper import encode_stack
from state.card import Card
from state.player import PlayerId
from state.ranks import Rank
from state.suits import Suit


class TestClass(unittest.TestCase):
    def test_encode_stack(self):
        own_id = PlayerId("own-id")
        ally_id = PlayerId("ally-id")
        opponent_id_1 = PlayerId("opponent-1-id")
        opponent_id_2 = PlayerId("opponent-2-id")
        allies = [own_id, ally_id]
        # 4 = order for four played cards, 32 = one hot encoded 32 cards, 2 = ally or not
        target = [0] * (4 + 32 * 2)

        self.assertListEqual(
            encode_stack([], allies), target, "Not the right encoding for empty stack"
        )

        rank_value = 0  # Ober
        suit_value = 0  # Eichel
        target_index = suit_value * 8 + rank_value
        target[0] = target_index + 1
        target[4 + 2 * target_index + 0] = 1
        target[4 + 2 * target_index + 1] = 2
        self.assertListEqual(
            encode_stack([(Card(Suit.EICHEL, Rank.OBER), own_id)], allies),
            target,
            "Not the right encoding for Eichel Ober",
        )

        rank_value = 2  # Ass
        suit_value = 1  # Gras
        target_index = suit_value * 8 + rank_value
        target[1] = target_index + 1
        target[4 + 2 * target_index + 0] = 1
        target[4 + 2 * target_index + 1] = 2
        print(target)
        print(
            encode_stack(
                [
                    (Card(Suit.EICHEL, Rank.OBER), own_id),
                    (Card(Suit.GRAS, Rank.ASS), ally_id),
                ],
                allies,
            )
        )
        self.assertListEqual(
            encode_stack(
                [
                    (Card(Suit.EICHEL, Rank.OBER), own_id),
                    (Card(Suit.GRAS, Rank.ASS), ally_id),
                ],
                allies,
            ),
            target,
            "Not the right encoding for Eichel Ober, Gras Ass",
        )

        rank_value = 4  # König
        suit_value = 2  # Herz
        target_index = suit_value * 8 + rank_value
        target[2] = target_index + 1
        target[4 + 2 * target_index + 0] = 1
        target[4 + 2 * target_index + 1] = 1
        print(target)
        self.assertListEqual(
            encode_stack(
                [
                    (Card(Suit.EICHEL, Rank.OBER), own_id),
                    (Card(Suit.GRAS, Rank.ASS), ally_id),
                    (Card(Suit.HERZ, Rank.KOENIG), opponent_id_1),
                ],
                allies,
            ),
            target,
            "Not the right encoding for Eichel Ober, Gras Ass, Herz König",
        )

        rank_value = 7  # Sieben
        suit_value = 3  # Schellen
        target_index = suit_value * 8 + rank_value
        target[3] = target_index + 1
        target[4 + 2 * target_index + 0] = 1
        target[4 + 2 * target_index + 1] = 1
        print(target)
        self.assertListEqual(
            encode_stack(
                [
                    (Card(Suit.EICHEL, Rank.OBER), own_id),
                    (Card(Suit.GRAS, Rank.ASS), ally_id),
                    (Card(Suit.HERZ, Rank.KOENIG), opponent_id_1),
                    (Card(Suit.SCHELLEN, Rank.SIEBEN), opponent_id_2),
                ],
                allies,
            ),
            target,
            "Not the right encoding for Eichel Ober, Gras Ass, Herz König, Schellen Sieben",
        )
