import unittest

from src.ai.select_card.simple_deep_q_learning.rl_agent import DQLAgent, DQLAgentConfig
from state.card import Card
from state.ranks import Rank
from state.stack import Stack
from state.suits import Suit


class TestClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def setUp(self) -> None:
        config = DQLAgentConfig("./model/params.pth")
        self.agent = DQLAgent(config)
        self.agent.initialize()
        return super().setUp()

    def skip_test_init_rl_agent(self):
        self.assertIsNotNone(self.agent)

    def skip_test_run_play_card(self):
        some_hand_cards = [
            Card(Suit.EICHEL, Rank.ACHT),
            Card(Suit.EICHEL, Rank.NEUN),
            Card(Suit.GRAS, Rank.ASS),
            Card(Suit.GRAS, Rank.UNTER),
            Card(Suit.HERZ, Rank.SIEBEN),
            Card(Suit.SCHELLEN, Rank.OBER),
            Card(Suit.SCHELLEN, Rank.UNTER),
        ]
        output = self.agent.select_card(Stack(), some_hand_cards)
        self.assertIsInstance(output, Card)
