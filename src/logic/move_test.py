import unittest

from logic.move import Move
from state.cards import Card


class MoveTest(unittest.TestCase):
    def test_move(self):
        card = Card("A", "S")
        move = Move(card)

        self.assertEqual(move.card, card)


if __name__ == "__main__":
    unittest.main()
