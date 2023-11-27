import math
import random

import torch

from ai.nn_helper import card_to_action, encode_dqn_input
from ai.select_card.simple_deep_q_learning.dql_processor import DQLProcessor
from ai.select_card.simple_deep_q_learning.policyNN import PolicyNN
from ai.select_card.simple_deep_q_learning.rl_agent import DQLAgent, DQLAgentConfig
from state.card import Card
from state.event import CardPlayedUpdate, Event, GameEndUpdate, RoundResultUpdate
from state.player import Player, PlayerId
from state.stack import Stack

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

GAME_WON_REWARD = 10
ROUND_WON_REWARD = 1


class DQLAgentTrainer(DQLAgent):
    def __init__(self, config: DQLAgentConfig):
        super().__init__(config)
        self.__target_net = PolicyNN()
        self.dql_processor = DQLProcessor(
            batch_size=BATCH_SIZE,
            policy_model=self.model,
            target_model=self.__target_net,
            lr=LR,
            gamma=GAMMA,
            tau=TAU,
        )
        self.__steps_done = 0
        self.__state: list[int] | None = None
        self.__reward: int = 0
        self.__played_card: int | None = None
        self.__round_cards: list[tuple[Card, PlayerId]] = []

    def _initialize_model(self, model_path: str):
        super()._initialize_model(model_path)
        self.model.train()

        self.__target_net.load_state_dict(self.model.state_dict())
        self.__target_net.eval()

    def persist_trained_policy(self):
        if self._game_type is None:
            raise Exception("game_type is not defined, yet.")
        
        torch.save(self.model.state_dict(), self._get_model_path(self._game_type))

    def __get_eps_threshold(self, steps_done: int) -> float:
        return EPS_DECAY + (EPS_START - EPS_END) * math.exp(
            -1.0 * steps_done / EPS_DECAY
        )

    def select_card(self, stack: Stack, playable_cards: list[Card]):
        self.__state = encode_dqn_input(
            [
                (played_card.card, played_card.player.id)
                for played_card in stack.get_played_cards()
            ],
            self.get_allies(),
            playable_cards,
        )
        sample = random.random()
        eps_threshold = self.__get_eps_threshold(self.__steps_done)
        self.__steps_done += 1

        if sample > eps_threshold:
            card = self._compute_best_card(stack, playable_cards)
        else:
            card = random.choice(playable_cards)

        self.__played_card = card_to_action(card)
        # print("Selected card for state %s and handcards %s: %s"%(stack.__str__(), playable_cards.__str__(), card.__str__()))
        return card

    def __update_last_played_card_on_demand(self, event: Event):
        if isinstance(event, CardPlayedUpdate):
            self.__round_cards.append((event.card, event.player))

        if isinstance(event, RoundResultUpdate):
            self.__round_cards.clear()

    def __update_reward_on_demand(self, event: Event, player: Player):
        if isinstance(event, GameEndUpdate):
            if player.id in event.winner:
                self.__reward = GAME_WON_REWARD
            else:
                self.__reward = -1
        if isinstance(event, RoundResultUpdate):
            if player.id == event.round_winner:
                self.__reward = ROUND_WON_REWARD
            else:
                self.__reward = 0

    def __memoize_step_on_demand(self, event: Event, player: Player):
        if isinstance(event, GameEndUpdate) or isinstance(event, RoundResultUpdate):
            if self.__state is None:
                raise ValueError(
                    "state is None but must be not None to memoize the step."
                )
            if self.__played_card is None:
                raise ValueError(
                    "action (played card) must not be None to memoize the step."
                )

            encoded_cards_of_this_round = encode_dqn_input(
                self.__round_cards, self.get_allies(), player.hand.cards
            )
            next_state = [
                a | b for a, b in zip(encoded_cards_of_this_round, self.__state) # TODO: self.state contains former hand cards??
            ]
            self.dql_processor.memoize_state(
                self.__state, self.__played_card, self.__reward, next_state
            )

    def __apply_training_on_demand(self, event: Event):
        if isinstance(event, GameEndUpdate) or isinstance(event, RoundResultUpdate):
            self.dql_processor.optimize_model()
            self.dql_processor.update_network()

    def on_game_event(self, event: Event, player: Player):
        super().on_game_event(event, player)
        self.__update_reward_on_demand(event, player)
        self.__update_last_played_card_on_demand(event)
        self.__memoize_step_on_demand(event, player)
        self.__apply_training_on_demand(event)

    def public__get_state(self):
        return (self.__state, self.__played_card, self.__reward)
