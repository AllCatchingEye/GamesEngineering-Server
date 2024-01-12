import logging
from abc import ABC

from ai.select_card.agent import ISelectCardAgent
from ai.select_card.rl_agent import RLBaseAgent
from state.card import Card
from state.event import Event, GameEndUpdate, RoundResultUpdate
from state.gametypes import Gametype
from state.player import PlayerId
from state.ranks import Rank

GAME_WON_REWARD = 10
ROUND_WON_REWARD = 1
# Ass provides the most values
MAX_ROUND_POINTS = 4 * Rank.ASS.value


class RLBaseAgentTrainer(ISelectCardAgent, ABC):
    __logger = logging.getLogger("RLBaseAgentTrainer")

    def __init__(self, agent: RLBaseAgent):
        super().__init__()
        self.agent = agent
        self._reward: float = 0

    def __update_reward_on_demand(self, event: Event, player_id: PlayerId):
        if isinstance(event, GameEndUpdate):
            if player_id in event.winner:
                self._reward += GAME_WON_REWARD
                result = "win"
            else:
                self._reward += -1
                result = "lost"
            self.__logger.debug(
                "🎁 Set reward %i for the game agent's team %s", self._reward, result
            )
        if isinstance(event, RoundResultUpdate):
            if self.agent.game_type == Gametype.RAMSCH:
                self._reward += 0
                result = "(ramsch)"
            elif player_id == event.round_winner:
                self._reward += ROUND_WON_REWARD / MAX_ROUND_POINTS * event.points
                result = "won"
            else:
                self._reward += 0
                result = "lost"
            self.__logger.debug(
                "🎁 Set reward %i for the round agent's team %s", self._reward, result
            )

    def reset(self) -> None:
        self.__logger.debug("🎁 Reset reward")
        self._reward = 0

    def set_hand_cards(self, hand_cards: list[Card]):
        return self.agent.set_hand_cards(hand_cards)

    def __handle_reset_on_demand(self, event: Event):
        if isinstance(event, GameEndUpdate):
            self.reset()

    def __handle_reset_on_demand(self, event: Event):
        if isinstance(event, GameEndUpdate):
            self.reset()

    def on_game_event(self, event: Event, player_id: PlayerId):
        self.__update_reward_on_demand(event, player_id)

    def on_pre_game_event(self, event: Event, player_id: PlayerId) -> None:
        self.agent.on_pre_game_event(event, player_id)

    def on_post_game_event(self, event: Event, player_id: PlayerId) -> None:
        self.agent.on_post_game_event(event, player_id)
        self.__handle_reset_on_demand(event)
