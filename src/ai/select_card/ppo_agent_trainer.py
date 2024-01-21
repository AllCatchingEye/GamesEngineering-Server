import logging
from typing import cast

import torch
from ai.select_card.models.model_interface import ModelInterface
from ai.select_card.models.ppo.iteration_01.shared import INPUT_SIZE
from ai.select_card.models.shared.encoding import (
    get_card_from_one_hot_encoding_index,
)
from ai.select_card.rl_agent_trainer import RLBaseAgentTrainer
from ai.select_card.ppo_agent import PpoAgent
from ai.nn_helper import NUM_CARDS, NUM_ROUNDS
from ai.select_card.ppo_processor import Agent, PpoProcessor, PpoProcessorConfig
from state.card import Card
from state.gametypes import Gametype
from state.player import PlayerId
from state.event import (
    Event,
    GameEndUpdate,
    GameStartUpdate,
    GametypeDeterminedUpdate,
    RoundResultUpdate,
)
from state.stack import Stack

class PpoAgentTrainer(RLBaseAgentTrainer):
    __logger = logging.getLogger("PpoAgentTrainer")

    def __init__(
        self,
        agent: PpoAgent,
        critic: ModelInterface,
    ):
        super().__init__(agent=agent)
        self.ppo_agent = cast(PpoAgent, self.agent)
        self.critic = critic
        ppo_processor_config = PpoProcessorConfig(
            observation_space_shape=((INPUT_SIZE,)),
            action_space_shape=((NUM_CARDS,)),
            agent=Agent(actor=self.ppo_agent.actor.get_raw_model(), critic=self.critic.get_raw_model()),
            num_steps=NUM_ROUNDS + 1,  # TODO: check if +1 is necessary for final result
        )
        self.ppo_processor = PpoProcessor(ppo_processor_config)
        self.step = 0

        self._state: torch.Tensor | None
        self._logprob: torch.Tensor | None
        self._action: torch.Tensor | None
        self._value: torch.Tensor | None
        self._played_card: Card | None
        self.former_game_type: Gametype | None = None

    def persist_agent(self):
        self.ppo_agent.actor.persist_parameters(self.ppo_agent.get_game_type_safe())
        self.critic.persist_parameters(self.ppo_agent.get_game_type_safe())

    def __reset_step_on_demand(self, event: Event):
        if isinstance(event, GameStartUpdate):
            self.step = 0

    def __handle_model_initialization_on_demand(self, event: Event):
        if isinstance(event, GametypeDeterminedUpdate):
            if self.former_game_type != event.gametype:
                self.former_game_type = event.gametype
                self.__logger.debug(
                    "🤖 Load parameters for game type %s and target model",
                    event.gametype.name,
                )
                try:
                    self.critic.init_params(event.gametype)
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

    def select_card(
        self, player_id: PlayerId, stack: Stack, playable_cards: list[Card]
    ) -> Card:
        encoded_state = self.ppo_agent.encode_state(
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

        action, logprob, _, value = self.ppo_processor.combined_agent.get_action_and_value(
            encoded_state, playable_cards=playable_cards
        )
        card = get_card_from_one_hot_encoding_index(action.item())
        self._action = action
        self._logprob = logprob
        self._played_card = card
        self._state = encoded_state
        self._value = value
        return card
    
    def __increase_step_on_demand(self, event: Event):
        if isinstance(event, RoundResultUpdate):
            self.step += 1

    def __memoize_step_on_demand(self, event: Event, player_id: PlayerId):
        if isinstance(event, GameEndUpdate) or isinstance(event, RoundResultUpdate):
            if self._state is None:
                raise ValueError(
                    "state is None but must be not None to memoize the step."
                )
            if self._action is None:
                raise ValueError(
                    "action is None but must be not None to memoize the step"
                )
            if self._value is None:
                raise ValueError(
                    "value is None but must be not None to memoize the step"
                )
            if self._logprob is None:
                raise ValueError(
                    "value is None but must be not None to memoize the step"
                )
            if self._played_card is None:
                raise ValueError(
                    "action (played card) must not be None to memoize the step."
                )

            encoded_next_state = self.ppo_agent.encode_state(
                player_id=player_id,
                play_order=self.agent.get_play_order_safe(),
                current_stack=self.agent.round_cards,
                previous_stacks=self.agent.previous_stacks[: (NUM_ROUNDS - 1)],
                allies=self.agent.get_allies(),
                playable_cards=self.agent.get_hand_cards_safe(),
            )
            self.ppo_processor.memoize_timestep(
                step=self.step,
                initial_done=self.step < NUM_ROUNDS - 1,
                next_done=self.step < NUM_ROUNDS,
                initial_obs=self._state,
                next_obs=encoded_next_state,
                action=int(self._action.item()),
                critic_value=self._value.item(),
                reward=self._reward,
                logprob=self._logprob.item(),
            )

    def __apply_training_on_demand(self, event: Event):
        if isinstance(event, GameEndUpdate):
            self.apply_training()

    def apply_training(self):
        self.__logger.debug("🤖 Compute advantage Training")
        advantages, returns = self.ppo_processor.compute_advantages_and_returns()
        self.__logger.debug("🤖 Apply Training")
        self.ppo_processor.optimize_actor_critic(advantages, returns)

    def on_pre_game_event(self, event: Event, player_id: PlayerId) -> None:
        super().on_pre_game_event(event, player_id)
        self.__reset_step_on_demand(event)
        self.__handle_model_initialization_on_demand(event)

    def on_game_event(self, event: Event, player_id: PlayerId):
        super().on_game_event(event, player_id)
        self.__memoize_step_on_demand(event, player_id)
        self.__increase_step_on_demand(event)
        self.__apply_training_on_demand(event)
