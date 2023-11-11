import unittest

from ai.select_card.rl_agent import RLAgent, RLAgentConfig

class TestClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def test_init(self):
        config = RLAgentConfig("", True)
        agent = RLAgent(config)
        agent.initialize()


