import logging
import os

import numpy as np
import torch
from torch import Tensor, nn

from ai.nn_helper import (
    NUM_CARDS,
    NUM_ROUNDS,
    NUM_STACK_CARDS,
    get_one_hot_encoding_index_from_card,
    one_hot_encode_cards,
)
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
from state.ranks import Rank

NUM_SITTING_ORDER = NUM_STACK_CARDS
NUM_ENCODED_STACK = NUM_STACK_CARDS + NUM_CARDS * 2
NUM_ENCODED_PREVIOUS_STACKS = NUM_ENCODED_STACK * (NUM_ROUNDS - 1)
NUM_ENCODED_HAND_CARDS = NUM_CARDS

INPUT_SIZE = (
    NUM_SITTING_ORDER
    + NUM_ENCODED_STACK
    + NUM_ENCODED_PREVIOUS_STACKS
    + NUM_ENCODED_HAND_CARDS
)
OUTPUT_SIZE = 32

ENEMY_ENCODING = 1
ALLY_ENCODING = 2
US_ENCODING = 3


class Encoding:
    NOT_PLAYED = 0
    PLAYED = 1
    ENEMY = 1
    ALLY = 2
    US = 3


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
    def __init__(self, layers: list[int]):
        super().__init__()
        self.flatten = nn.Flatten()
        all_layers = [INPUT_SIZE] + layers + [OUTPUT_SIZE]
        self.linear_relu_stack = nn.Sequential()

        for index, layer in enumerate(all_layers):
            if index <= len(all_layers) - 1 - 1:
                self.linear_relu_stack.append(nn.Linear(layer, all_layers[index + 1]))
            if index <= len(all_layers) - 1 - 2:
                self.linear_relu_stack.append(nn.ReLU())

    def forward(self, x: Tensor):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class ModelIter02(ModelInterface):
    __logger = logging.getLogger("ModelIter02")

    def __init__(self, layers: list[int]):
        self.layers = layers
        self.model = PolicyNN(layers)
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
            data=np.array(
                [
                    self.__encode_dqn_input(
                        player_id,
                        play_order,
                        current_stack,
                        previous_stacks,
                        allies,
                        playable_cards,
                    )
                ]
            ),
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
        player_id: PlayerId,
        play_order: list[PlayerId],
        current_stack: list[tuple[Card, PlayerId]],
        previous_stacks: list[list[tuple[Card, PlayerId]]],
        allies: list[PlayerId],
        playable_cards: list[Card],
    ) -> list[int]:
        encoded_play_order = self.__encode_play_order(play_order, player_id, allies)
        encoded_current_stack = self.__encode_stack(current_stack, allies)
        encoded_previous_stacks = self.__encode_stacks(previous_stacks, allies)
        encoded_playable_cards = one_hot_encode_cards(playable_cards)
        return (
            encoded_play_order
            + encoded_current_stack
            + encoded_previous_stacks
            + encoded_playable_cards
        )

    def __encode_play_order(
        self, play_order: list[PlayerId], player_id: PlayerId, allies: list[PlayerId]
    ):
        return [
            Encoding.US
            if id == player_id
            else Encoding.ALLY
            if id in allies
            else Encoding.ENEMY
            for id in play_order
        ]

    def __encode_stacks(
        self, stacks: list[list[tuple[Card, PlayerId]]], allies: list[PlayerId]
    ) -> list[int]:
        if len(stacks) > NUM_ROUNDS:
            raise ValueError(
                "Provided more than %i previous stacks: %i" % (NUM_ROUNDS, len(stacks))
            )

        result: list[int] = []
        for index in range(NUM_ROUNDS - 1):
            if index < len(stacks):
                result += self.__encode_stack(stacks[index], allies)
            else:
                result += [Encoding.NOT_PLAYED] * NUM_ENCODED_STACK
        return result

    def __encode_stack(
        self, stack: list[tuple[Card, PlayerId]], allies: list[PlayerId]
    ) -> list[int]:
        # embed the order of the stack + for every card if it was played and in case if from an ally
        result = [Encoding.NOT_PLAYED] * NUM_ENCODED_STACK
        for index, [card, player_id] in enumerate(stack):
            if index >= NUM_STACK_CARDS:
                raise ValueError(
                    "The stack contains more cards than allowed: %i vs. max %i"
                    % (len(stack), NUM_STACK_CARDS)
                )

            encoded_index = get_one_hot_encoding_index_from_card(card)
            result[index] = encoded_index
            result[NUM_STACK_CARDS + index * 2 + 0] = Encoding.PLAYED
            result[NUM_STACK_CARDS + index * 2 + 1] = (
                Encoding.ALLY if player_id in allies else Encoding.ENEMY
            )

        return result

    def get_model_params_path(self, game_type: Gametype) -> str:
        from_here = os.path.dirname(os.path.abspath(__file__))
        params_path = ["params", "x".join([str(it) for it in self.layers])]
        folder_path = from_here
        for path_segment in params_path:
            folder_path = os.path.join(folder_path, path_segment)
            if not os.path.exists(folder_path):
                self.__logger.debug("Create directory %s", folder_path)
                os.mkdir(folder_path)
        return os.path.join(folder_path, as_params_file(game_type.name.lower()))

    def init_params(self, game_type: Gametype):
        path = self.get_model_params_path(game_type)
        if not os.path.exists(path):
            raise ValueError(
                "Cannot load model parameters from path %s: The path doesn't exist."
                % path
            )
        load_and_apply_parameters_to_model(self.model, path, self.device)

    def persist_parameters(self, game_type: Gametype):
        persist_model_parameters(self.model, self.get_model_params_path(game_type))
