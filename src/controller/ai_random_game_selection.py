import os
import random
from ai.select_card.simple_deep_q_learning.rl_agent import DQLAgent, DQLAgentConfig
from ai.select_card.simple_deep_q_learning.rl_agent_trainer import DQLAgentTrainer
from ai.select_game_type.neuronal_network.agent.nn_game_decision_agent import NNAgent, NNAgentConfig
from controller.ai_controller import AiController
from controller.random_controller import RandomController

from state.gametypes import GameGroup


class AiRandomGameTypeController(AiController):

    def __init__(self, train: bool = False):
        super().__init__()
        from_here = os.path.dirname(os.path.abspath(__file__))

        nn_agents_model_params_path = os.path.join(
            from_here,
            "..",
            "ai",
            "select_game_type",
            "neuronal_network",
            "model",
            "params.pth",
        )
        nn_agent_config = NNAgentConfig(model_params_path=nn_agents_model_params_path)
        self.select_game_agent = NNAgent(nn_agent_config)

        rl_agents_model_params_path = os.path.join(
            from_here,
            "..",
            "ai",
            "select_card",
            "simple_deep_q_learning",
            "model",
        )
        rl_agent_config = DQLAgentConfig(
            policy_model_base_path=rl_agents_model_params_path
        )
        self.play_game_agent = (
            DQLAgentTrainer(rl_agent_config)
            if train is True
            else DQLAgent(rl_agent_config)
        )
    
    async def wants_to_play(self, current_lowest_gamegroup: GameGroup) -> bool:
        self.select_game_agent.should_play(
            hand_cards=self.hand_cards,
            current_lowest_gamegroup=current_lowest_gamegroup,
        )
        rng = random.Random()
        return rng.choice([True, False])
