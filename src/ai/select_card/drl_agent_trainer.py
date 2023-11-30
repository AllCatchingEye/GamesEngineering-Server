import logging
import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

from ai.select_card.drl_agent import DRLAgent
from ai.select_card.rl_agent_trainer import RLBaseAgentTrainer
from ai.select_card.simple_deep_q_learning.dql_processor import DQLProcessor
from state.card import Card
from state.event import Event, GameEndUpdate, RoundResultUpdate
from state.player import PlayerId
from state.stack import Stack


@dataclass
class DRLAgentTrainerConfig:
    batch_size: int
    discount_factor: float
    exploration_rate_start: float
    exploration_rate_end: float
    exploration_rate_decay: float
    target_net_update_rate: float
    learning_rate: float


class DRLAgentTrainer(RLBaseAgentTrainer, ABC):
    __logger = logging.getLogger("DRLAgentTrainer")

    def __init__(
        self,
        agent: DRLAgent,
        target_model: torch.nn.Module,
        config: DRLAgentTrainerConfig,
    ):
        super().__init__(agent)
        self.agent = agent
        self._target_net = target_model
        self.config = config
        self.dql_processor = DQLProcessor(
            batch_size=config.batch_size,
            policy_model=self.agent.model,
            target_model=self._target_net,
            lr=config.learning_rate,
            gamma=config.discount_factor,
            tau=config.target_net_update_rate,
        )
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._state: torch.Tensor | None
        self._played_card: Card | None
        self.__steps_done = 0

    def __get_eps_threshold(self, steps_done: int) -> float:
        return self.config.exploration_rate_decay + (
            self.config.exploration_rate_start - self.config.exploration_rate_end
        ) * math.exp(-1.0 * steps_done / self.config.exploration_rate_decay)

    def select_card(self, stack: Stack, playable_cards: list[Card]) -> Card:
        if self.agent.hand_cards is None:
            raise ValueError("Agents hand cards are None")

        encoded_state = self.agent.encode_state(
            [
                (played_card.card, played_card.player.id)
                for played_card in stack.get_played_cards()
            ],
            self.agent.get_allies(),
            self.agent.hand_cards,
        )
        self._state = encoded_state
        sample = random.random()
        eps_threshold = self.__get_eps_threshold(self.__steps_done)
        self.__steps_done += 1

        if sample > eps_threshold:
            card = self.agent.select_card(stack, playable_cards)
            self.__logger.debug("🃏 Select card using exploitation: %s", card)
        else:
            card = random.choice(playable_cards)
            self.__logger.debug("🃏 Select card using exploration: %s", card)

        self._played_card = card
        return card

    @abstractmethod
    def _encode_card(self, card: Card) -> torch.Tensor:
        """Encode card as tensor"""

    @abstractmethod
    def _encode_reward(self, reward: int) -> torch.Tensor:
        """Encode reward as tensor"""

    def __memoize_step_on_demand(self, event: Event):
        if isinstance(event, GameEndUpdate) or isinstance(event, RoundResultUpdate):
            if self._state is None:
                raise ValueError(
                    "state is None but must be not None to memoize the step."
                )
            if self._played_card is None:
                raise ValueError(
                    "action (played card) must not be None to memoize the step."
                )
            if self.agent.hand_cards is None:
                raise ValueError("Agents hand cards are None")

            encoded_next_state = self.agent.encode_state(
                self._round_cards, self.agent.get_allies(), self.agent.hand_cards
            )
            encoded_card = self._encode_card(self._played_card)
            encoded_reward = self._encode_reward(self._reward)
            self.dql_processor.memoize_state(
                self._state, encoded_card, encoded_reward, encoded_next_state
            )

    def __apply_training_on_demand(self, event: Event):
        if isinstance(event, GameEndUpdate) or isinstance(event, RoundResultUpdate):
            self.__logger.debug("🤖 Optimize agent's model")
            self.dql_processor.optimize_model()
            self.__logger.debug("🤖 Update target network")
            self.dql_processor.update_network()

    def on_game_event(self, event: Event, player_id: PlayerId):
        super().on_game_event(event, player_id)
        self.__memoize_step_on_demand(event)
        self.__apply_training_on_demand(event)
