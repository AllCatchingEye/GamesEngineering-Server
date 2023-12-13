import logging
from abc import ABC
from dataclasses import dataclass

from ai.select_card.agent import ISelectCardAgent
from state.card import Card
from state.event import (
    AnnouncePlayPartyUpdate,
    CardPlayedUpdate,
    Event,
    GameStartUpdate,
    GametypeDeterminedUpdate,
    PlayOrderUpdate,
    RoundResultUpdate,
)
from state.gametypes import Gametype
from state.player import PlayerId, struct_play_parties


@dataclass
class DQLAgentConfig:
    policy_model_base_path: str


class RLBaseAgent(ISelectCardAgent, ABC):
    __logger = logging.getLogger("RLBaseAgent")

    def __init__(self):
        self.game_type: Gametype | None
        self.allies: list[PlayerId] = []
        self.play_order: list[PlayerId] | None = None
        self.hand_cards: list[Card] | None
        self.previous_stacks: list[list[tuple[Card, PlayerId]]] = []
        self.round_cards: list[tuple[Card, PlayerId]] = []

    def __get_allies(
        self, parties: list[list[PlayerId]], player_id: PlayerId
    ) -> list[PlayerId]:
        for party in parties:
            if player_id in party:
                return party
        return []

    def _reset(self):
        self.__logger.debug("Reset allies")
        self.allies = []
        self.previous_stacks = []
        self.play_order = None

    def get_previous_stacks(self):
        return self.previous_stacks

    def get_allies(self):
        return self.allies

    def get_play_order_safe(self):
        if self.play_order is None:
            raise ValueError("Play order is not defined, yet")
        return self.play_order

    def get_hand_cards_safe(self):
        if self.hand_cards is None:
            raise ValueError("Hand cards are not defined, yet")
        return self.hand_cards

    def get_play_game_type_safe(self):
        if self.game_type is None:
            raise ValueError("Game type is not defined, yet")
        return self.game_type

    def __handle_reset_on_demand(self, event: Event):
        if isinstance(event, GameStartUpdate):
            self.reset()

    def __handle_allies_on_demand(self, event: Event, player_id: PlayerId):
        if (
            isinstance(event, GametypeDeterminedUpdate)
            and event.gametype != Gametype.SAUSPIEL
            and event.parties is not None
        ):
            self.__logger.debug("👬 Setting allies")
            self.allies = self.__get_allies(
                struct_play_parties(event.parties), player_id
            )
        elif isinstance(event, AnnouncePlayPartyUpdate):
            self.__logger.debug("👬 Setting allies")
            self.allies = self.__get_allies(
                struct_play_parties(event.parties), player_id
            )

    def __handle_player_order_on_demand(self, event: Event):
        if isinstance(event, PlayOrderUpdate):
            self.play_order = event.order

    def set_hand_cards(self, hand_cards: list[Card]):
        self.hand_cards = hand_cards

    def __update_last_played_card_on_demand(self, event: Event):
        if isinstance(event, CardPlayedUpdate):
            self.__logger.debug(
                "🃏 Store last played card %s of %s",
                event.card,
                "own team" if event.player in self.get_allies() else "others team",
            )
            self.round_cards.append((event.card, event.player))

        if isinstance(event, RoundResultUpdate):
            self.__logger.debug("🃏 Clear last played cards")
            self.previous_stacks.append(self.round_cards)
            self.round_cards.clear()

    def on_game_event(self, event: Event, player_id: PlayerId):
        self.__handle_reset_on_demand(event)
        self.__handle_player_order_on_demand(event)
        self.__handle_allies_on_demand(event, player_id)
        self.__update_last_played_card_on_demand(event)
