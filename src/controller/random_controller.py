import random

from controller.player_controller import PlayerController
from state.card import Card
from state.event import Event
from state.gametypes import GameGroup, Gametype
from state.player import Player
from state.stack import Stack
from state.suits import Suit


class RandomController(PlayerController):
    def __init__(self, player: Player, rng: random.Random = random.Random()):
        super().__init__(player)
        self.rng = rng

    def wants_to_play(self, current_lowest_gamegroup: GameGroup) -> bool:
        return self.rng.choice([True, False])

    async def select_gametype(
        self, choosable_gametypes: list[tuple[Gametype, Suit | None]]
    ) -> tuple[Gametype, Suit | None]:
        return self.rng.choice(choosable_gametypes)

    async def play_card(self, stack: Stack, playable_cards: list[Card]) -> Card:
        return self.rng.choice(playable_cards)

    async def on_game_event(self, event: Event) -> None:
        pass

    def choose_game_group(self, available_groups: list[GameGroup]) -> GameGroup:
        return self.rng.choice(available_groups)
