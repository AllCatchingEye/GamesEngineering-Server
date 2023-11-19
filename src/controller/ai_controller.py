import os

from ai.select_card.simple_deep_q_learning.rl_agent import RLAgent, RLAgentConfig
from ai.select_game_type.neuronal_network.agent.gametype_helper import (
    game_type_to_game_group,
)
from ai.select_game_type.neuronal_network.agent.nn_game_decision_agent import (
    NNAgent,
    NNAgentConfig,
)
from controller.player_controller import PlayerController
from state.card import Card
from state.event import Event
from state.gametypes import GameGroup, Gametype
from state.player import Player
from state.stack import Stack
from state.suits import Suit


class AiController(PlayerController):
    def __init__(self, player: Player, train: bool = False):
        super().__init__(player)
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
            "params.pth",
        )
        rl_agent_config = RLAgentConfig(
            policy_model_path=rl_agents_model_params_path, train=train
        )
        self.play_game_agent = RLAgent(rl_agent_config)

    async def select_gametype(
        self, choosable_gametypes: list[tuple[Gametype, Suit | None]]
    ) -> tuple[Gametype, Suit | None]:
        return self.select_game_agent.select_game_type(
            hand_cards=self.player.hand.get_all_cards(),
            choosable_game_types=choosable_gametypes,
        )

    async def play_card(self, stack: Stack, playable_cards: list[Card]) -> Card:
        return self.play_game_agent.play_card(
            stack=stack, playable_cards=playable_cards
        )

    async def on_game_event(self, event: Event) -> None:
        self.play_game_agent.on_game_event(event=event)

    async def wants_to_play(self, current_lowest_gamegroup: GameGroup):
        return self.select_game_agent.should_play(
            hand_cards=self.player.hand.get_all_cards(),
            current_lowest_gamegroup=current_lowest_gamegroup,
        )

    async def choose_game_group(self, available_groups: list[GameGroup]) -> GameGroup:
        """Choose the highest game group you would play"""
        assert (
            self.select_game_agent.targeted_game_type is not None
        ), "First ask the ai controller if it wants to play, then ask for the game group"
        return game_type_to_game_group(self.select_game_agent.targeted_game_type[0])
