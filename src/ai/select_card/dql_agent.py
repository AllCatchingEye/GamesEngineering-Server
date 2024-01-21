import logging

import torch

from ai.select_card.models.model_interface import ModelInterface
from ai.select_card.rl_agent import RLBaseAgent
from state.card import Card
from state.event import Event, GametypeDeterminedUpdate
from state.gametypes import Gametype
from state.player import PlayerId
from state.stack import Stack


class DQLAgent(RLBaseAgent):
    __logger = logging.getLogger("DRLAgent")

    def __init__(self, policy_model: ModelInterface):
        super().__init__()
        self.model = policy_model
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.game_type: Gametype | None = None

    def get_game_type_safe(self):
        if self.game_type is None:
            raise ValueError("game_type is not defined, yet.")

        return self.game_type

    def initialize_model(self):
        self.model.init_params(self.get_game_type_safe())
        self.model.eval()

    def encode_state(
        self,
        player_id: PlayerId,
        play_order: list[PlayerId],
        current_stack: list[tuple[Card, PlayerId]],
        previous_stacks: list[list[tuple[Card, PlayerId]]],
        allies: list[PlayerId],
        playable_cards: list[Card],
    ) -> torch.Tensor:
        return self.model.encode_input(
            player_id,
            play_order,
            current_stack,
            previous_stacks,
            allies,
            playable_cards,
        )

    def _compute_best_card(
        self, player_id: PlayerId, stack: Stack, playable_cards: list[Card]
    ):
        with torch.no_grad():
            transformed_state = [
                (card.card, card.player) for card in stack.played_cards
            ]
            return self.model.forward(
                player_id,
                self.get_play_order_safe(),
                transformed_state,
                self.previous_stacks,
                self.get_allies(),
                playable_cards,
            )

    def select_card(
        self, player_id: PlayerId, stack: Stack, playable_cards: list[Card]
    ):
        best_card = self._compute_best_card(player_id, stack, playable_cards)
        self.__logger.debug("🃏 Selected card %s", best_card)
        return best_card

    def __handle_model_initialization_on_demand(self, event: Event):
        if isinstance(event, GametypeDeterminedUpdate):
            self.__logger.debug("🎯 Announced Game Type: %s", event.gametype)
            if self.game_type != event.gametype:
                self.game_type = event.gametype
                self.__logger.debug(
                    "🤖 Load parameters for game type %s and for policy model",
                    event.gametype.name,
                )
                try:
                    self.model.init_params(event.gametype)
                    self.__logger.debug(
                        "🤖 Use existing model parameters for policy model"
                    )
                except ValueError:
                    self.__logger.debug(
                        "🤖 Use initial model parameters for policy model"
                    )
            else:
                self.__logger.debug(
                    "🤖 Use loaded model parameters for policy model since the game type hasn't changed"
                )

    def on_pre_game_event(self, event: Event, player_id: PlayerId):
        super().on_pre_game_event(event, player_id)
        self.__handle_model_initialization_on_demand(event)
