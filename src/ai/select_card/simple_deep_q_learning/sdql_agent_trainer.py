import logging

import torch

from ai.nn_helper import card_to_action
from ai.select_card.drl_agent_trainer import DRLAgentTrainer, DRLAgentTrainerConfig
from ai.select_card.simple_deep_q_learning.policyNN import PolicyNN
from ai.select_card.simple_deep_q_learning.sdql_agent import SDQLAgent, SDQLAgentConfig
from state.card import Card
from state.event import Event, GametypeDeterminedUpdate
from state.player import PlayerId

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 1
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4


class SDQLAgentTrainer(DRLAgentTrainer):
    __logger = logging.getLogger("SDQLAgentTrainer")

    def __init__(self, config: SDQLAgentConfig):
        agent_config = SDQLAgentConfig(
            policy_model_base_path=config.policy_model_base_path
        )
        agent = SDQLAgent(agent_config)
        agent_trainer_config = DRLAgentTrainerConfig(
            batch_size=BATCH_SIZE,
            discount_factor=GAMMA,
            exploration_rate_start=EPS_START,
            exploration_rate_end=EPS_END,
            exploration_rate_decay=EPS_DECAY,
            target_net_update_rate=TAU,
            learning_rate=LR,
        )
        super().__init__(
            agent=agent, target_model=PolicyNN(), config=agent_trainer_config
        )

    def persist_trained_policy(self):
        if self.agent.game_type is None:
            raise Exception("game_type is not defined, yet.")

        model_path = self.agent.get_model_path(self.agent.game_type)
        self.__logger.debug(
            "Persist trained policy net parameters to file %s", model_path
        )

        torch.save(
            self.agent.model.state_dict(),
            model_path,
        )

    def _encode_state(
        self,
        stack: list[tuple[Card, PlayerId]],
        allies: list[PlayerId],
        playable_cards: list[Card],
    ) -> torch.Tensor:
        state = self.agent.encode_state(stack, allies, playable_cards)
        return torch.tensor([state], dtype=torch.float, device=self._device)

    def _encode_card(self, card: Card) -> torch.Tensor:
        action = card_to_action(card)
        return torch.tensor([[action]], dtype=torch.int64, device=self._device)

    def _encode_reward(self, reward: int) -> torch.Tensor:
        return torch.tensor([reward], dtype=torch.float, device=self._device)

    def __handle_model_initialization_on_demand(self, event: Event):
        if isinstance(event, GametypeDeterminedUpdate):
            self._target_net.load_state_dict(self.agent.model.state_dict())
            self._target_net.eval()

    def on_game_event(self, event: Event, player_id: PlayerId):
        super().on_game_event(event, player_id)
        self.__handle_model_initialization_on_demand(event)
