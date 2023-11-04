from logic.gamemodes.gamemode import GameMode
from state.card import Card
from state.hand import Hand
from state.player import Player
from state.stack import Stack


# TODO: Implement
class GameModeRamsch(GameMode):
    def __init__(self):
        super().__init__(None)

    def get_trump_cards(self) -> list[Card]:
        return []

    def get_playable_cards(self, stack: Stack, hand: Hand) -> list[Card]:
        return []

    def determine_stitch_winner(self, stack: Stack) -> Player:
        return stack.get_played_cards()[0].get_player()
