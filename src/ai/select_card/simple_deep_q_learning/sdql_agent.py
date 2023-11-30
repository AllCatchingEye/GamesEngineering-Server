from dataclasses import dataclass

import numpy as np
import torch

from ai.nn_helper import action_to_card, card_to_nn_input_values, encode_dqn_input
from ai.select_card.drl_agent import DRLAgent
from ai.select_card.simple_deep_q_learning.policyNN import PolicyNN
from state.card import Card
from state.gametypes import Gametype
from state.player import PlayerId


@dataclass
class SDQLAgentConfig:
    policy_model_base_path: str


class SDQLAgent(DRLAgent):
    def __init__(self, config: SDQLAgentConfig):
        super().__init__(PolicyNN())
        self._config = config

    def apply_parameters_to_model(self, model: torch.nn.Module, model_path: str):
        params = torch.load(model_path, map_location=self._device)
        assert isinstance(params, dict), (
            "The path "
            + model_path
            + " does not contain the required parameters for that model."
        )
        model.load_state_dict(params)

    def get_model_path(self, game_type: Gametype):
        return "%s/%s.pth" % (
            self._config.policy_model_base_path,
            game_type.name.lower(),
        )

    def reset(self):
        super()._reset_allies()

    def encode_state(
        self,
        stack: list[tuple[Card, PlayerId]],
        allies: list[PlayerId],
        playable_cards: list[Card],
    ) -> torch.Tensor:
        input_values = encode_dqn_input(
            stack,
            allies,
            playable_cards,
        )
        return torch.tensor(np.array([input_values]).astype(np.float32))

    def decode_card(self, output: torch.Tensor, playable_cards: list[Card]) -> Card:
        best_card = None
        optimal_q_value: float = float("-inf")
        tensor_list: list[float] = output.tolist()[0]
        encoded_playable_cards = card_to_nn_input_values(playable_cards)
        for index, q_value in enumerate(tensor_list):
            if q_value > optimal_q_value and encoded_playable_cards[index] == 1:
                optimal_q_value = q_value
                best_card = action_to_card(index)

        assert best_card is not None, "could not found a best card to play: " + str(
            len(tensor_list)
        )
        return best_card
