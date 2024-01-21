import os

import numpy as np
import torch
from torch import Tensor, nn

from ai.nn_helper import one_hot_encode_cards
from ai.select_card.models.model_interface import ModelInterface
from ai.select_card.models.shared.encoding import pick_highest_valid_card
from ai.select_card.models.shared.nn_state import (
    as_params_file,
    load_and_apply_parameters_to_model,
    persist_model_parameters,
)
from state.card import Card
from state.gametypes import Gametype
from state.player import PlayerId
from state.ranks import Rank, get_all_ranks
from state.suits import get_all_suits

NUM_RANKS = len(get_all_ranks())
NUM_SUITS = len(get_all_suits())
NUM_CARDS = NUM_RANKS * NUM_SUITS

INPUT_SIZE = 96
OUTPUT_SIZE = 32

rank_values = {
    Rank.OBER: 0,
    Rank.UNTER: 1,
    Rank.ASS: 2,
    Rank.ZEHN: 3,
    Rank.KOENIG: 4,
    Rank.NEUN: 5,
    Rank.ACHT: 6,
    Rank.SIEBEN: 7,
}


class PolicyNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(INPUT_SIZE, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, OUTPUT_SIZE),
        )

    def forward(self, x: Tensor):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class ModelIter01(ModelInterface):
    def __init__(self):
        self.model = PolicyNN()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_raw_model(self):
        return self.model

    def _forward(self, input_values: Tensor) -> Tensor:
        return self.model.forward(input_values)

    def encode_input(
        self,
        player_id: PlayerId,
        play_order: list[PlayerId],
        current_stack: list[tuple[Card, PlayerId]],
        previous_stacks: list[list[tuple[Card, PlayerId]]],
        allies: list[PlayerId],
        playable_cards: list[Card],
    ) -> Tensor:
        return torch.tensor(
            np.array([self.__encode_dqn_input(current_stack, allies, playable_cards)]),
            dtype=torch.float,
            device=self.device,
        )

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def decode_output(self, output: torch.Tensor, playable_cards: list[Card]) -> Card:
        return pick_highest_valid_card(output, playable_cards)

    def __encode_dqn_input(
        self,
        stack: list[tuple[Card, PlayerId]],
        allies: list[PlayerId],
        playable_cards: list[Card],
    ) -> list[int]:
        encoded_input: list[int] = []
        played_cards = [played_card for (played_card, _) in stack]
        encoded_stack_input = one_hot_encode_cards(played_cards)
        encoded_ally_input = self.__allied_card_nn_input(stack, allies)
        encoded_playable_cards = one_hot_encode_cards(playable_cards)
        encoded_input.extend(encoded_stack_input)
        encoded_input.extend(encoded_ally_input)
        encoded_input.extend(encoded_playable_cards)
        return encoded_input

    def __allied_card_nn_input(
        self, stack: list[tuple[Card, PlayerId]], allies: list[PlayerId]
    ) -> list[int]:
        allied_cards = [card for [card, player_id] in stack if player_id in allies]
        return one_hot_encode_cards(allied_cards)

    def get_model_params_path(self, game_type: Gametype) -> str:
        from_here = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(from_here, "params", as_params_file(game_type.name.lower()))

    def init_params(self, game_type: Gametype):
        if os.path.exists(self.get_model_params_path(game_type)):
            load_and_apply_parameters_to_model(
                self.model, self.get_model_params_path(game_type), self.device
            )

    def persist_parameters(self, game_type: Gametype):
        persist_model_parameters(self.model, self.get_model_params_path(game_type))
