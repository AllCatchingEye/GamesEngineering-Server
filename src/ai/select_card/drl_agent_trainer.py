import logging
import math
import random
from dataclasses import dataclass

import torch

from ai.nn_helper import NUM_ROUNDS, get_one_hot_encoding_index_from_card, one_hot_encode_cards
from ai.select_card.dql_processor import DQLProcessor
from ai.select_card.drl_agent import DRLAgent
from ai.select_card.models.model_interface import ModelInterface
from ai.select_card.rl_agent_trainer import RLBaseAgentTrainer
from state.card import Card
from state.event import (
    Event,
    GameEndUpdate,
    GametypeDeterminedUpdate,
    RoundResultUpdate,
)
from state.gametypes import Gametype
from state.player import PlayerId
from state.stack import Stack

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 100
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 10_000
TAU = 0.005
LR = 1e-6


@dataclass
class DRLAgentTrainerConfig:
    batch_size: int
    discount_factor: float
    exploration_rate_start: float
    exploration_rate_end: float
    exploration_rate_decay: float
    target_net_update_rate: float
    learning_rate: float


DRLAgentTrainerDefaultConfig = DRLAgentTrainerConfig(
    batch_size=BATCH_SIZE,
    discount_factor=GAMMA,
    exploration_rate_start=EPS_START,
    exploration_rate_end=EPS_END,
    exploration_rate_decay=EPS_DECAY,
    target_net_update_rate=TAU,
    learning_rate=LR,
)


class DRLAgentTrainer(RLBaseAgentTrainer):
    __logger = logging.getLogger("DRLAgentTrainer")

    def __init__(
        self,
        agent: DRLAgent,
        target_model: ModelInterface,
        training_config: DRLAgentTrainerConfig = DRLAgentTrainerDefaultConfig,
    ):
        super().__init__(agent)
        self.agent = agent
        self._target_net = target_model
        self.config = training_config
        self.dql_processor = DQLProcessor(
            batch_size=training_config.batch_size,
            policy_model=self.agent.model.get_raw_model(),
            target_model=self._target_net.get_raw_model(),
            lr=training_config.learning_rate,
            gamma=training_config.discount_factor,
            tau=training_config.target_net_update_rate,
        )
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._state: torch.Tensor | None
        self._played_card: Card | None
        self.steps_done = 0
        self.former_game_type: Gametype | None = None
        self._allowed_next_states: torch.Tensor | None


    def persist_trained_policy(self):
        self.agent.model.persist_parameters(self.agent.get_game_type_safe())

    def get_eps_threshold(self, steps_done: int) -> float:
        return self.config.exploration_rate_end + (
            self.config.exploration_rate_start - self.config.exploration_rate_end
        ) * math.exp(-1.0 * steps_done / self.config.exploration_rate_decay)

    def select_card(
        self, player_id: PlayerId, stack: Stack, playable_cards: list[Card]
    ) -> Card:
        encoded_state = self.agent.encode_state(
            player_id=player_id,
            play_order=self.agent.get_play_order_safe(),
            current_stack=[
                (played_card.card, player_id)
                for played_card in stack.get_played_cards()
            ],
            previous_stacks=self.agent.get_previous_stacks(),
            allies=self.agent.get_allies(),
            playable_cards=self.agent.get_hand_cards_safe(),
        )
        self._state = encoded_state
        sample = random.random()
        eps_threshold = self.get_eps_threshold(self.steps_done)
        self.steps_done += 1

        if sample > eps_threshold:
            card = self.agent.select_card(player_id, stack, playable_cards)
            self.__logger.debug("🃏 Select card using exploitation: %s", card)
        else:
            card = random.choice(playable_cards)
            self.__logger.debug("🃏 Select card using exploration: %s", card)

        self._played_card = card
        return card

    def _encode_card(self, card: Card) -> torch.Tensor:
        action = get_one_hot_encoding_index_from_card(card)
        return torch.tensor([[action]], dtype=torch.int64, device=self._device)

    def _encode_reward(self, reward: float) -> torch.Tensor:
        return torch.tensor([reward], dtype=torch.float, device=self._device)

    def _encode_final(self, is_final: bool) -> torch.Tensor:
        return torch.tensor([is_final], dtype=torch.float, device=self._device)

    def _encode_allowed_targets(self, allowed_targets: list[int]) -> torch.Tensor:
        return torch.tensor([allowed_targets], dtype=torch.float, device=self._device)

    def __handle_model_initialization_on_demand(self, event: Event):
        if isinstance(event, GametypeDeterminedUpdate):
            if self.former_game_type != event.gametype:
                self.former_game_type = event.gametype
                self.__logger.debug(
                    "🤖 Load parameters for game type %s and target model",
                    event.gametype.name,
                )
                try:
                    self._target_net.init_params(event.gametype)
                    self.__logger.debug(
                        "🤖 Use existing model parameters for target model"
                    )
                except ValueError:
                    self.__logger.debug(
                        "🤖 Use initial model parameters for target model"
                    )
            else:
                self.__logger.debug(
                    "🤖 Use loaded model parameters for target model since the game type hasn't changed"
                )

    def __memoize_step_on_demand(self, event: Event, player_id: PlayerId):
        if isinstance(event, GameEndUpdate) or isinstance(event, RoundResultUpdate):
            self.__logger.debug(
                "🧠 Memoize step including played card %s, reward %f and more",
                self._played_card,
                self._reward,
            )
            if self._state is None:
                raise ValueError(
                    "state is None but must be not None to memoize the step."
                )
            if self._played_card is None:
                raise ValueError(
                    "action (played card) must not be None to memoize the step."
                )

            encoded_next_state = self.agent.encode_state(
                player_id=player_id,
                play_order=self.agent.get_play_order_safe(),
                current_stack=self.agent.round_cards,
                previous_stacks=self.agent.previous_stacks[: (NUM_ROUNDS - 1)],
                allies=self.agent.get_allies(),
                playable_cards=self.agent.get_hand_cards_safe(),
            )
            allowed_targets = one_hot_encode_cards(self.agent.get_hand_cards_safe())
            is_final = isinstance(event, GameEndUpdate) or all(element == 0 for element in allowed_targets)
           
            encoded_card = self._encode_card(self._played_card)
            encoded_reward = self._encode_reward(self._reward)
            encoded_is_final = self._encode_final(is_final)
            encoded_allowed_targets = self._encode_allowed_targets(allowed_targets) == 1.0
            self.dql_processor.memoize_state(
                self.agent.get_game_type_safe(),
                self._state,
                encoded_card,
                encoded_reward,
                encoded_next_state,
                encoded_is_final,
                encoded_allowed_targets
            )

    def __apply_training_on_demand(self, event: Event):
        if isinstance(event, GameEndUpdate) or isinstance(event, RoundResultUpdate):
            self.__logger.debug("🤖 Optimize agent's model")
            self.dql_processor.optimize_model(self.agent.get_game_type_safe())
            self.__logger.debug("🤖 Update target network")
            self.dql_processor.update_network()

    def on_game_event(self, event: Event, player_id: PlayerId):
        super().on_game_event(event, player_id)
        self.__memoize_step_on_demand(event, player_id)
        self.__apply_training_on_demand(event)

    def on_pre_game_event(self, event: Event, player_id: PlayerId) -> None:
        super().on_pre_game_event(event, player_id)
        self.__handle_model_initialization_on_demand(event)
