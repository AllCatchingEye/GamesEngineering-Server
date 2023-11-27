from dataclasses import dataclass
from os import path

import numpy as np
import torch

from ai.nn_helper import action_to_card, card_to_nn_input_values, encode_dqn_input
from ai.select_card.agent import ISelectCardAgent
from ai.select_card.simple_deep_q_learning.policyNN import PolicyNN
from state.card import Card
from state.event import (
    AnnouncePlayPartyUpdate,
    Event,
    GameStartUpdate,
    GametypeDeterminedUpdate,
)
from state.gametypes import Gametype
from state.player import PlayerId
from state.stack import Stack


@dataclass
class DQLAgentConfig:
    policy_model_base_path: str


class DQLAgent(ISelectCardAgent):
    def __init__(self, config: DQLAgentConfig):
        self._config = config
        self.model = PolicyNN()
        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.__allies = []
        self._game_type = None

    def __load_model(self, model_path: str):
        params = torch.load(model_path, map_location=self.__device)
        assert isinstance(params, dict), (
            "The path "
            + model_path
            + " does not contain the required parameters for that model."
        )
        self.model.load_state_dict(params)

    def _get_model_path(self, game_type: Gametype):
        return "%s/%s.pth" % (
            self._config.policy_model_base_path,
            game_type.name.lower(),
        )

    def _initialize_model(self, model_path: str):
        self.__load_model(model_path)
        self.model.eval()

    def reset(self):
        self.__allies = []

    def __get_input_tensor(self, stack: Stack, playable_cards: list[Card]):
        input_values = encode_dqn_input(
            [
                (played_card.card, played_card.player.id)
                for played_card in stack.played_cards
            ],
            self.__allies,
            playable_cards,
        )
        return torch.tensor(np.array([input_values]).astype(np.float32))

    def __find_best_card(
        self, tensor: torch.Tensor, playable_cards: list[Card]
    ) -> Card:
        best_card = None
        optimal_q_value: float = float("-inf")
        tensor_list: list[float] = tensor.tolist()[0]
        encoded_playable_cards = card_to_nn_input_values(playable_cards)
        for index, q_value in enumerate(tensor_list):
            if q_value > optimal_q_value and encoded_playable_cards[index] == 1:
                optimal_q_value = q_value
                best_card = action_to_card(index)

        assert best_card is not None, "could not found a best card to play: " + str(
            len(tensor_list)
        )
        return best_card

    def _compute_best_card(self, stack: Stack, playable_cards: list[Card]):
        with torch.no_grad():
            input_tensor = self.__get_input_tensor(stack, playable_cards)
            output: torch.Tensor = self.model(input_tensor)
            return self.__find_best_card(output, playable_cards)

    def select_card(self, stack: Stack, playable_cards: list[Card]):
        return self._compute_best_card(stack, playable_cards)

    def __get_allies(self, parties: list[list[PlayerId]]) -> list[PlayerId]:
        for party in parties:
            if self.player_id in party:
                return party
        return []

    def get_allies(self):
        return self.__allies

    def __handle_allies(self, event: Event):
        if (
            isinstance(event, GametypeDeterminedUpdate)
            and event.gametype != Gametype.SAUSPIEL
            and event.parties is not None
        ):
            self.__allies = self.__get_allies(event.parties)
        elif isinstance(event, AnnouncePlayPartyUpdate):
            self.__allies = self.__get_allies(event.parties)

    def __handle_model_loading(self, event: Event):
        if isinstance(event, GametypeDeterminedUpdate):
            self._game_type = event.gametype
            model_path = self._get_model_path(event.gametype)
            if path.exists(model_path):
                self._initialize_model(model_path)

    def on_game_event(self, event: Event):
        if isinstance(event, GameStartUpdate):
            self.player_id = event.player

        self.__handle_model_loading(event)
        self.__handle_allies(event)
