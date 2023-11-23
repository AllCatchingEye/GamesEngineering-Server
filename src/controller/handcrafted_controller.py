
from controller.player_controller import PlayerController
from state.card import Card
from src.state.event import *
from state.gametypes import GameGroup, Gametype
from state.player import Player
from state.stack import Stack
from state.suits import Suit


class HandcraftedController(PlayerController):
    hand: list[Card]
    player: Player
    played_cards: list[Card]
    ally: list[Player]
    current_gametype: Gametype
    def __init__(self, player: Player):
        super().__init__(player)
        self.player = player
        self.played_cards = []

    async def wants_to_play(self, current_lowest_gamegroup: GameGroup) -> bool:

    async def select_gametype(
        self, choosable_gametypes: list[tuple[Gametype, Suit | None]]
    ) -> tuple[Gametype, Suit | None]:
        return self.rng.choice(choosable_gametypes)

    async def play_card(self, stack: Stack, playable_cards: list[Card]) -> Card:
        return self.rng.choice(playable_cards)

    async def on_game_event(self, event: Event) -> None:
        match event:
            case GameStartUpdate():
                self.hand = event.hand
            case GametypeDeterminedUpdate():
                self.current_gametype = event.gametype
                for party in event.parties:
                    if self.player in party:
                        self.ally = party
            case CardPlayedUpdate():
                self.played_cards.append(event.card)

    async def choose_game_group(self, available_groups: list[GameGroup]) -> GameGroup:
        return self.rng.choice(available_groups)
