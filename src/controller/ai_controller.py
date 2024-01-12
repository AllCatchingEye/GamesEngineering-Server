import os

from ai.select_card.drl_agent import DRLAgent
from ai.select_card.drl_agent_trainer import DRLAgentTrainer
from ai.select_card.models.iteration_02.model import ModelIter02
from ai.select_game_type.two_layer_nn.two_layer_game_decision_agent import (
    NNAgentConfig,
    SelectGameAgent,
)
from controller.player_controller import PlayerController
from state.card import Card
from state.event import Event, GameStartUpdate
from state.gametypes import GameGroup, Gametype
from state.player import PlayerId
from state.stack import Stack
from state.suits import Suit


class AiController(PlayerController):
    def __init__(self, net_layers: list[int], train: bool = False):
        super().__init__()
        self.hand_cards: list[Card] | None
        self.player_id: PlayerId | None

        from_here = os.path.dirname(os.path.abspath(__file__))

        should_play_params_path = os.path.join(
            from_here,
            "..",
            "ai",
            "select_game_type",
            "two_layer_nn",
            "models",
            "binary_classifier.pth",
        )

        select_game_params_path = os.path.join(
            from_here,
            "..",
            "ai",
            "select_game_type",
            "two_layer_nn",
            "models",
            "game_classifier.pth",
        )
        nn_agent_config = NNAgentConfig(
            should_play_params_path, select_game_params_path
        )
        self.select_game_agent = SelectGameAgent(nn_agent_config)

        drl_agent = DRLAgent(ModelIter02(net_layers))
        if train is True:
            self.play_game_agent = DRLAgentTrainer(
                agent=drl_agent, target_model=ModelIter02(net_layers)
            )
        else:
            self.play_game_agent = drl_agent

    async def select_gametype(
        self, choosable_gametypes: list[tuple[Gametype, Suit | None]]
    ) -> tuple[Gametype, Suit | None]:
        if self.hand_cards is None:
            raise ValueError("Controller's hand cards are None")

        return self.select_game_agent.select_game_type(
            hand_cards=self.hand_cards,
            choosable_game_types=choosable_gametypes,
        )

    async def play_card(self, stack: Stack, playable_cards: list[Card]) -> Card:
        if self.player_id is None:
            raise ValueError("Player ID is not defined, yet")

        selected_card = self.play_game_agent.select_card(
            player_id=self.player_id, stack=stack, playable_cards=playable_cards
        )

        if self.hand_cards is None:
            raise ValueError("Controllers hand cards are None")

        index = self.hand_cards.index(selected_card)
        if index < 0:
            raise ValueError(
                "Agent selected a card that is not in the hand cards. %s not in %s"
                % (selected_card, self.hand_cards)
            )

        self.hand_cards.pop(index)
        self.play_game_agent.set_hand_cards(self.hand_cards)
        return selected_card

    def __handle_game_start_on_demand(self, event: Event):
        if isinstance(event, GameStartUpdate):
            self.player_id = event.player
            self.hand_cards = event.hand
            self.play_game_agent.set_hand_cards(self.hand_cards)

    async def on_game_event(self, event: Event) -> None:
        self.__handle_game_start_on_demand(event)

        if self.player_id is None:
            raise ValueError("Controller's player id is None")
        self.play_game_agent.on_pre_game_event(event=event, player_id=self.player_id)
        self.play_game_agent.on_game_event(event=event, player_id=self.player_id)
        self.play_game_agent.on_post_game_event(event=event, player_id=self.player_id)

    async def wants_to_play(self, current_lowest_gamegroup: GameGroup):
        if self.hand_cards is None:
            raise ValueError("Controllers hand cards are None")

        return self.select_game_agent.should_play(
            hand_cards=self.hand_cards,
            current_lowest_gamegroup=current_lowest_gamegroup,
        )

    async def choose_game_group(self, available_groups: list[GameGroup]) -> GameGroup:
        """Choose the highest game group you would play"""

        return self.select_game_agent.choose_game_group(available_groups)

    def persist_training_results(self):
        if not isinstance(self.play_game_agent, DRLAgentTrainer):
            raise ValueError(
                "The controller doesn't get trained so the parameters can't be persisted."
            )

        self.play_game_agent.persist_trained_policy()
