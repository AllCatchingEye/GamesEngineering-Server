import logging
from abc import ABC

from ai.select_card.agent import ISelectCardAgent
from ai.select_card.rl_agent import RLBaseAgent
from state.card import Card
from state.event import CardPlayedUpdate, Event, GameEndUpdate, RoundResultUpdate
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
        self.rewards: list[float] = []
        self._round_cards: list[tuple[Card, PlayerId]] = []

    def __update_last_played_card_on_demand(self, event: Event):
        if isinstance(event, CardPlayedUpdate):
            self.__logger.debug(
                "🃏 Store last played card %s of %s",
                event.card,
                "own team"
                if event.player in self.agent.get_allies()
                else "others team",
            )
            self._round_cards.append((event.card, event.player))

        if isinstance(event, RoundResultUpdate):
            self.__logger.debug("🃏 Clear last played cards")
            self._round_cards.clear()

    def __update_reward_on_demand(self, event: Event, player_id: PlayerId):
        if isinstance(event, GameEndUpdate):
            if player_id in event.winner:
                self._reward = GAME_WON_REWARD
                result = "win"
            else:
                self._reward = -1
                result = "lost"
            self.__logger.debug(
                "🎁 Set reward %i for the game agent's team %s", self._reward, result
            )
        if isinstance(event, RoundResultUpdate):
            if self.agent.game_type == Gametype.RAMSCH:
                self._reward = 0
                result = "(ramsch)"
            elif player_id == event.round_winner:
                self._reward = ROUND_WON_REWARD / MAX_ROUND_POINTS * event.points
                result = "won"
            else:
                self._reward = 0
                result = "lost"
            self.__logger.debug(
                "🎁 Set reward %i for the round agent's team %s", self._reward, result
            )
        if isinstance(event, GameEndUpdate) or isinstance(event, RoundResultUpdate):
            self.rewards.append(self._reward)

    def reset(self) -> None:
        self.rewards.clear()
        self.agent.reset()

    def set_hand_cards(self, hand_cards: list[Card]):
        return self.agent.set_hand_cards(hand_cards)

    def on_game_event(self, event: Event, player_id: PlayerId):
        self.agent.on_game_event(event, player_id)
        self.__update_reward_on_demand(event, player_id)
        self.__update_last_played_card_on_demand(event)
