from abc import ABC, abstractmethod

import torch

from state.card import Card
from state.gametypes import Gametype
from state.player import PlayerId


class ModelInterface(ABC):
    def forward(
        self,
        player_id: PlayerId,
        play_order: list[PlayerId],
        current_stack: list[tuple[Card, PlayerId]],
        previous_stacks: list[list[tuple[Card, PlayerId]]],
        allies: list[PlayerId],
        playable_cards: list[Card],
    ):
        return self.decode_output(
            self._forward(
                self.encode_input(
                    player_id,
                    play_order,
                    current_stack,
                    previous_stacks,
                    allies,
                    playable_cards,
                )
            ),
            playable_cards,
        )

    @abstractmethod
    def get_raw_model(self) -> torch.nn.Module:
        """Returns the underlying torch Module"""

    @abstractmethod
    def encode_input(
        self,
        player_id: PlayerId,
        play_order: list[PlayerId],
        current_stack: list[tuple[Card, PlayerId]],
        previous_stacks: list[list[tuple[Card, PlayerId]]],
        allies: list[PlayerId],
        playable_cards: list[Card],
    ) -> torch.Tensor:
        """encodes the input for the model"""

    @abstractmethod
    def decode_output(self, output: torch.Tensor, playable_cards: list[Card]) -> Card:
        """decodes the output of the model to a card"""

    @abstractmethod
    def _forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """passes the input into the model and returns the output"""

    @abstractmethod
    def get_model_params_path(self, game_type: Gametype) -> str:
        """Returns the path where the model parameters have to be persisted"""

    @abstractmethod
    def persist_parameters(self, game_type: Gametype):
        """Persists the model for the given game type"""

    @abstractmethod
    def init_params(self, game_type: Gametype):
        """Initializes the model for the given game type"""

    @abstractmethod
    def eval(self):
        """prepare model for regular usage"""

    @abstractmethod
    def train(self):
        """prepare model for training"""
