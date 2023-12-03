import logging
from abc import ABC
from dataclasses import dataclass

from ai.select_card.agent import ISelectCardAgent
from state.card import Card
from state.event import AnnouncePlayPartyUpdate, Event, GametypeDeterminedUpdate
from state.gametypes import Gametype
from state.player import PlayerId


@dataclass
class DQLAgentConfig:
    policy_model_base_path: str


class RLBaseAgent(ISelectCardAgent, ABC):
    __logger = logging.getLogger("RLBaseAgent")

    def __init__(self):
        self.game_type: Gametype | None
        self.allies = []
        self.hand_cards: list[Card] | None

    def __get_allies(
        self, parties: list[list[PlayerId]], player_id: PlayerId
    ) -> list[PlayerId]:
        for party in parties:
            if player_id in party:
                return party
        return []

    def _reset_allies(self):
        self.__logger.debug("Reset allies")
        self.allies = []

    def get_allies(self):
        return self.allies

    def __handle_allies(self, event: Event, player_id: PlayerId):
        if (
            isinstance(event, GametypeDeterminedUpdate)
            and event.gametype != Gametype.SAUSPIEL
            and event.parties is not None
        ):
            self.__logger.debug("👬 Setting allies")
            self.allies = self.__get_allies(event.parties, player_id)
        elif isinstance(event, AnnouncePlayPartyUpdate):
            self.__logger.debug("👬 Setting allies")
            self.allies = self.__get_allies(event.parties, player_id)

    def set_hand_cards(self, hand_cards: list[Card]):
        self.hand_cards = hand_cards

    def on_game_event(self, event: Event, player_id: PlayerId):
        self.__handle_allies(event, player_id)
