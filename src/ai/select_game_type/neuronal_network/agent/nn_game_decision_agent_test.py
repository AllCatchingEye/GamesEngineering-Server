import os
import unittest

from ai.select_game_type.neuronal_network.agent.nn_game_decision_agent import NNAgent, NNAgentConfig

from state.card import Card
from state.gametypes import Gametype
from state.ranks import Rank
from state.suits import Suit

class TestClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
    
    def setUp(self):
        super().setUp()
        from_here = os.path.dirname(os.path.abspath(__file__))
        config = NNAgentConfig(
            model_params_path=os.path.join(from_here, "..", "model", "params.pth")
        )
        self.agent = NNAgent(config)

    def test_agent_init(self):
        # Must not raise
        self.agent.initialize()

    def test_agent_works(self):
        some_hand_cards = [
            Card(Suit.EICHEL, Rank.ACHT),
            Card(Suit.EICHEL, Rank.NEUN),
            Card(Suit.GRAS, Rank.ASS),
            Card(Suit.GRAS, Rank.UNTER),
            Card(Suit.HERZ, Rank.SIEBEN),
            Card(Suit.SCHELLEN, Rank.OBER),
            Card(Suit.SCHELLEN, Rank.UNTER),
        ]
        # Must not raise
        should_play = self.agent.should_play(hand_cards=some_hand_cards, decisions=[])
        self.assertEqual(type(should_play), bool)
        game_type = self.agent.select_game_type(hand_cards=some_hand_cards, choosable_game_types=[])
        self.assertIsInstance(game_type, Gametype)