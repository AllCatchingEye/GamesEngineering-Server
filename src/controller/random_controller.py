import random
from controller.player_controller import PlayerController
from state.card import Card
from state.gametypes import Gametype
from state.player import Player
from state.stack import Stack


class RandomController(PlayerController):
    def __init__(self, player: Player, rng: random.Random = random.Random()):
        super().__init__(player)
        self.rng = rng

    def wants_to_play(self, decisions: list[bool | None]) -> bool:
        return self.rng.choice([True, False])

    def select_gametype(self, choosable_gametypes: list[Gametype]) -> Gametype:
        return self.rng.choice(choosable_gametypes)

    def play_card(self, stack: Stack, playable_cards: list[Card]) -> Card:
        return self.rng.choice(playable_cards)

    def announce_round_result(self) -> None:
        pass

    def announce_game_result(self) -> None:
        pass

    def announce_gametype(self, gametype: Gametype) -> None:
        pass
