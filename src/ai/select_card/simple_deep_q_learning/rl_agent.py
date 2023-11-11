from dataclasses import dataclass

import numpy as np
import torch

from ai.nn_helper import card_to_nn_input_values, nn_output_code_to_card
from ai.select_card.agent import ISelectCardAgent
from ai.select_card.simple_deep_q_learning.policyNN import PolicyNN
from state.card import Card
from state.event import Event
from state.stack import Stack


@dataclass
class RLAgentConfig:
    policy_model_path: str
    train: bool


class RLAgent(ISelectCardAgent):
    def __init__(self, config: RLAgentConfig):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.decision = None
        self.model: torch.nn.Module = PolicyNN()
        self.last_cards = []

    def initialize(self):
        if not self.config.train:
            self.load_model()
            self.model.eval()
        else:
            self.model.train()

    def reset(self) -> None:
        self.last_cards = []
        self.decision = None

    def __get_best_playable_card(self, tensor: torch.Tensor, playable_cards: list[int]):
        best_card = None
        max_prob = -10000
        tensor_list: list[float] = tensor.tolist()[0]

        for index, value in enumerate(tensor_list):
            if value > max_prob and playable_cards[index] == 1:
                best_card = nn_output_code_to_card(index)

        assert best_card is not None, "could not found a best card to play: " + str(
            playable_cards
        )

        return best_card

    def play_card(self, stack: Stack, playable_cards: list[Card]) -> Card:
        input_values = card_to_nn_input_values(playable_cards)
        input_tensor = torch.tensor(np.array([input_values]).astype(np.float32))
        output: torch.Tensor = self.model(input_tensor)
        return self.__get_best_playable_card(output, input_values)

    def load_model(self):
        params = torch.load(self.config.policy_model_path, map_location=self.device)
        assert isinstance(params, dict), (
            "The path "
            + self.config.policy_model_path
            + " does not contain the required parameters for that model."
        )
        self.model.load_state_dict(params)

    def on_game_event(self, event: Event) -> None:
        """Handle game events"""
