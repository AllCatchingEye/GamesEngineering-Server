import logging
from abc import ABC, abstractmethod
from os import path

import torch

from ai.select_card.rl_agent import RLBaseAgent
from state.card import Card
from state.event import Event, GametypeDeterminedUpdate
from state.gametypes import Gametype
from state.player import PlayerId
from state.stack import Stack


class DRLAgent(RLBaseAgent, ABC):
    __logger = logging.getLogger("DRLAgent")

    def __init__(self, policy_model: torch.nn.Module):
        super().__init__()
        self.model = policy_model
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.game_type: Gametype | None

    @abstractmethod
    def apply_parameters_to_model(self, model: torch.nn.Module, model_path: str):
        """Loads the model's parameters"""

    @abstractmethod
    def get_model_path(self, game_type: Gametype) -> str:
        """Computes the path where the model's parameters were persisted"""

    def initialize_model(self, model_path: str):
        self.apply_parameters_to_model(self.model, model_path)
        self.model.eval()

    def reset(self):
        super()._reset_allies()

    @abstractmethod
    def encode_state(
        self,
        stack: list[tuple[Card, PlayerId]],
        allies: list[PlayerId],
        playable_cards: list[Card],
    ) -> torch.Tensor:
        """Returns the input values as input tensor"""

    @abstractmethod
    def decode_card(self, output: torch.Tensor, playable_cards: list[Card]) -> Card:
        """Returns the valid card to play"""

    def _compute_best_card(self, stack: Stack, playable_cards: list[Card]):
        with torch.no_grad():
            transformed_state = [
                (card.card, card.player.id) for card in stack.played_cards
            ]
            input_tensor = self.encode_state(
                transformed_state, self.allies, playable_cards
            )
            output: torch.Tensor = self.model(input_tensor)
            return self.decode_card(output, playable_cards)

    def select_card(self, stack: Stack, playable_cards: list[Card]):
        best_card = self._compute_best_card(stack, playable_cards)
        self.__logger.debug("🃏 Selected card %s", best_card)
        return best_card

    def __handle_model_initialization_on_demand(self, event: Event):
        if isinstance(event, GametypeDeterminedUpdate):
            self.game_type = event.gametype
            model_path = self.get_model_path(event.gametype)
            if path.exists(model_path):
                self.__logger.debug(
                    "🤖 Initialize model parameters for game type %s from %s",
                    event.gametype,
                    model_path,
                )
                self.initialize_model(model_path)

    def on_game_event(self, event: Event, player_id: PlayerId):
        super().on_game_event(event, player_id)
        self.__handle_model_initialization_on_demand(event)
