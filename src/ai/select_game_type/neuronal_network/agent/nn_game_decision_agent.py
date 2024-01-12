import logging
from dataclasses import dataclass
from math import floor

import numpy as np
import torch

from ai.nn_helper import decode_game_type, one_hot_encode_cards
from ai.select_game_type.agent import ISelectGameAgent
from ai.select_game_type.gametype_helper import game_type_to_game_group
from ai.select_game_type.neuronal_network.agent.select_game_nn import SelectGameNN
from state.card import Card
from state.gametypes import GameGroup, Gametype
from state.ranks import Rank
from state.suits import Suit


@dataclass
class NNAgentConfig:
    model_params_path: str


class NNAgent(ISelectGameAgent):
    __logger = logging.getLogger("NNAgent")

    def __init__(self, config: NNAgentConfig):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.targeted_game_type = None
        self.model = SelectGameNN()

    def initialize(self):
        """Invoked to initialize the neuronal network which the agent will use for its decisions"""
        params = torch.load(self.config.model_params_path, map_location=self.device)
        assert isinstance(params, dict), (
            "The path "
            + self.config.model_params_path
            + " does not contain the required parameters for that model."
        )
        self.model.load_state_dict(params)
        self.model.eval()

    def reset(self):
        self.targeted_game_type = None

    def should_play(
        self, hand_cards: list[Card], current_lowest_gamegroup: GameGroup
    ) -> bool:
        """Invoked to receive a decision if the agent would play"""
        input_values = one_hot_encode_cards(hand_cards)
        input_tensor = torch.tensor(np.array([input_values]).astype(np.float32))
        output: torch.Tensor = self.model(input_tensor)
        selected_game_type_code = torch.max(output, 1).indices[0].item()

        self.targeted_game_type = decode_game_type(floor(selected_game_type_code))
        if self.targeted_game_type[0] is Gametype.SAUSPIEL:
            sauspiel = self.get_best_sauspiel(output.tolist()[0], hand_cards)
            if sauspiel is not None:
                self.targeted_game_type = sauspiel
            else:
                # weiter if there is no valid sauspiel
                self.__logger.debug("🤷‍♂️ Should play: False")
                return False
        should_play = (
            self.targeted_game_type[0] != Gametype.RAMSCH
            and game_type_to_game_group(self.targeted_game_type[0]).value
            <= current_lowest_gamegroup.value
        )
        self.__logger.debug("🤷‍♂️ Should play: %s", str(should_play))
        return should_play

    def get_best_sauspiel(
        self, tensor: list[float], hand_cards: list[Card]
    ) -> tuple[Gametype, Suit | None] | None:
        max_value = float("-inf")
        sauspiel = None
        for idx, value in enumerate(tensor):
            game_type = decode_game_type(floor(idx))
            if game_type[0] is Gametype.SAUSPIEL:
                if value > max_value and self.is_valid_sauspiel(game_type, hand_cards):
                    max_value = value
                    sauspiel = game_type
        return sauspiel

    def is_valid_sauspiel(
        self, sauspiel: tuple[Gametype, Suit | None], hand_cards: list[Card]
    ) -> bool:
        hand_suits = list(map(lambda card: card.suit, hand_cards))
        if sauspiel[1] is not None:
            return (
                Card(sauspiel[1], Rank.ASS) not in hand_cards
                and sauspiel[1] in hand_suits
            )
        return False

    def select_game_type(
        self,
        hand_cards: list[Card],
        choosable_game_types: list[tuple[Gametype, Suit | None]],
    ):
        """Invoked to receive a decision which game type the agent would play. Note, that `choosable_game_types` is ignored for now."""
        assert (
            self.targeted_game_type != None
        ), "The agents decision if agent would play wasn't made, yet. Invoke `should_play` to do so."
        self.__logger.debug("🗣️ Select game type %s", self.targeted_game_type)
        return self.targeted_game_type
