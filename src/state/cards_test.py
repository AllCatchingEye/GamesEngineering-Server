import unittest

from state.cards import Card


class TestCard(unittest.TestCase):
    def test_card(self):
        card = Card("A", "Spades")
        self.assertEqual(card.rank, "A")
        self.assertEqual(card.suit, "Spades")
        self.assertEqual(card.__str__(), "A of Spades")


if __name__ == "__main__":
    unittest.main()
