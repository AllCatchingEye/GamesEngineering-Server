import os

from ai.select_card.simple_deep_q_learning.sdql_agent import SDQLAgent, SDQLAgentConfig
from ai.select_card.simple_deep_q_learning.sdql_agent_trainer import SDQLAgentTrainer
from ai.select_game_type.neuronal_network.agent.gametype_helper import (
    game_type_to_game_group,
)
from ai.select_game_type.neuronal_network.agent.nn_game_decision_agent import (
    NNAgent,
    NNAgentConfig,
)
from controller.player_controller import PlayerController
from state.card import Card
from state.event import Event, GameStartUpdate
from state.gametypes import GameGroup, Gametype
from state.player import PlayerId
from state.stack import Stack
from state.suits import Suit


class AiController(PlayerController):
    def __init__(self, train: bool = False):
        super().__init__()
        self.hand_cards: list[Card] | None
        self.player_id: PlayerId | None

        from_here = os.path.dirname(os.path.abspath(__file__))

        nn_agents_model_params_path = os.path.join(
            from_here,
            "..",
            "ai",
            "select_game_type",
            "neuronal_network",
            "model",
            "params.pth",
        )
        nn_agent_config = NNAgentConfig(model_params_path=nn_agents_model_params_path)
        self.select_game_agent = NNAgent(nn_agent_config)

        rl_agents_model_params_path = os.path.join(
            from_here,
            "..",
            "ai",
            "select_card",
            "simple_deep_q_learning",
            "model",
        )
        rl_agent_config = SDQLAgentConfig(
            policy_model_base_path=rl_agents_model_params_path
        )
        self.play_game_agent = (
            SDQLAgentTrainer(rl_agent_config)
            if train is True
            else SDQLAgent(rl_agent_config)
        )

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
        selected_card = self.play_game_agent.select_card(
            stack=stack, playable_cards=playable_cards
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
        self.play_game_agent.on_game_event(event=event, player_id=self.player_id)

    async def wants_to_play(self, current_lowest_gamegroup: GameGroup):
        if self.hand_cards is None:
            raise ValueError("Controllers hand cards are None")

        return self.select_game_agent.should_play(
            hand_cards=self.hand_cards,
            current_lowest_gamegroup=current_lowest_gamegroup,
        )

    async def choose_game_group(self, available_groups: list[GameGroup]) -> GameGroup:
        """Choose the highest game group you would play"""
        assert (
            self.select_game_agent.targeted_game_type is not None
        ), "First ask the ai controller if it wants to play, then ask for the game group"
        return game_type_to_game_group(self.select_game_agent.targeted_game_type[0])

    def persist_training_results(self):
        assert isinstance(
            self.play_game_agent, SDQLAgentTrainer
        ), "The controller doesn't get trained."

        self.play_game_agent.persist_trained_policy()
