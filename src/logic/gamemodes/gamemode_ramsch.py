from logic.gamemodes.gamemode import GameMode
from logic.gamemodes.gamemode_solo import GameModeSolo
from state.card import Card
from state.hand import Hand
from state.player import Player
from state.stack import Stack
from state.suits import Suit


class GameModeRamsch(GameMode):
    def __init__(self):
        super().__init__(None)
        self.heart_solo = GameModeSolo(Suit.HERZ)

    def get_trump_cards(self) -> list[Card]:
        return self.heart_solo.get_trump_cards()

    def get_playable_cards(self, stack: Stack, hand: Hand) -> list[Card]:
        return self.heart_solo.get_playable_cards(stack, hand)

    def determine_stitch_winner(self, stack: Stack) -> Player:
        return self.heart_solo.determine_stitch_winner(stack)
