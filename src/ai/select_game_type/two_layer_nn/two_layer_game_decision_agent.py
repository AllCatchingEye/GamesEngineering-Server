from dataclasses import dataclass

import numpy as np
import torch

from ai.nn_helper import card_to_nn_input_values, code_to_game_type
from ai.select_game_type.agent import ISelectGameAgent
from ai.select_game_type.gametype_helper import (
    game_type_to_game_group,
)
from ai.select_game_type.two_layer_nn.select_game_type_nn import SelectGameTypeNN
from ai.select_game_type.two_layer_nn.should_play_nn import ShouldPlayNN
from state.card import Card
from state.gametypes import GameGroup, Gametype
from state.suits import Suit


@dataclass
class NNAgentConfig:
    model_should_play_params_path: str
    model_select_gametype_params_path: str


class SelectGameAgent(ISelectGameAgent):
    def __init__(self, config: NNAgentConfig):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_select_game_type = SelectGameTypeNN()
        self.model_should_play = ShouldPlayNN()
        
        self.decision_tensor: torch.Tensor | None = None
        self.initialize()

    def initialize(self):
        """Invoked to initialize the neuronal network which the agent will use for its decisions"""
        should_play_params = self.load_model_params(self.config.model_should_play_params_path)  
        select_game_params = self.load_model_params(self.config.model_select_gametype_params_path)
        self.model_should_play.load_state_dict(should_play_params)
        self.model_select_game_type.load_state_dict(select_game_params)
        self.model_should_play.eval()
        self.model_select_game_type.eval()

    def load_model_params(self, path: str):
        params = torch.load(path, map_location=self.device)
        assert isinstance(params, dict), (
            "The path "
            + path
            + " does not contain the required parameters for that model."
        )
        return params

    def reset(self):
        self.targeted_game_type = None

    def should_play(
        self, hand_cards: list[Card], current_lowest_gamegroup: GameGroup
    ) -> bool:
        """Invoked to receive a decision if the agent would play"""
        input_values = card_to_nn_input_values(hand_cards)
        input_tensor = torch.tensor(np.array([input_values]).astype(np.float32))
        output = self.model_should_play(input_tensor)
        selected_game_type_code = torch.max(output, 1).indices[0].item()
        should_play = selected_game_type_code == 1
        if should_play:
            self.update_decision_tensor(hand_cards)
        return should_play

    def update_decision_tensor(self, hand_cards: list[Card]):
        input_values = card_to_nn_input_values(hand_cards)
        input_tensor = torch.tensor(np.array([input_values]).astype(np.float32))
        self.decision_tensor = self.model_select_game_type(input_tensor)

    def select_game_type(
        self,
        hand_cards: list[Card],
        choosable_game_types: list[tuple[Gametype, Suit | None]],
    ):
        assert self.decision_tensor is not None, (
            "SelectGameNN was not executed"
        )
        return self.get_best_game_type(choosable_game_types)

    def choose_game_group(self, available_groups: list[GameGroup]) -> GameGroup:
        assert self.decision_tensor is not None, (
            "SelectGameNN was not executed"
        )
        return self.get_best_game_group(available_groups)

    def get_best_game_type(self, choosable_game_types: list[tuple[Gametype, Suit | None]]) -> tuple[Gametype, Suit | None]:
        best_gametype: tuple[Gametype, Suit | None] | None = None
        max_val = float('-inf')
        for idx, val in enumerate(self.decision_tensor.tolist()[0]):
            gametype = code_to_game_type(idx)
            if val > max_val and gametype in choosable_game_types:
                max_val: float = val
                best_gametype = gametype
        assert best_gametype is not None, (
            "No playable gametype found"
        )
        return best_gametype

    def get_best_game_group(self, available_groups: list[GameGroup]) -> GameGroup:
        best_game_group: GameGroup | None = None
        max_val = float('-inf')
        for idx, val in enumerate(self.decision_tensor.tolist()[0]):
            game_type: tuple[Gametype, Suit | None] = code_to_game_type(idx)
            game_group = game_type_to_game_group(game_type[0])
            if val > max_val and game_group in available_groups:
                best_game_group = game_group
                max_val:float = val
        assert best_game_group is not None, (
            "best game group not found"
        )
        return best_game_group
