from logic.gamemodes.gamemode import GameMode
from logic.gamemodes.gamemode_solo import GameModeSolo
from state.card import Card
from state.hand import Hand
from state.player import Player
from state.ranks import Rank
from state.stack import Stack
from state.suits import Suit


class GameModeSauspiel(GameMode):
    def __init__(self, suit: Suit | None):
        super().__init__(suit)
        self.heart_solo = GameModeSolo(Suit.HERZ)
        self.trumps = self.heart_solo.trumps

    def get_trump_cards(self) -> list[Card]:
        return self.heart_solo.get_trump_cards()

    def get_playable_cards(self, stack: Stack, hand: Hand) -> list[Card]:

        if stack.is_empty():
            return hand.get_all_cards()

        if hand.has_card_of_rank_and_suit(self.suit, Rank.ASS):
            if stack.get_first_card() not in self.trumps and stack.get_first_card().suit == self.suit:
                # played ass is being searched
                return [hand.get_card_of_rank_and_suit(self.suit, Rank.ASS)]
            else:
                playable_cards = self.heart_solo.get_playable_cards(stack, hand)
                playable_cards.remove(hand.get_card_of_rank_and_suit(self.suit, Rank.ASS))
                return playable_cards
        else:
            return self.heart_solo.get_playable_cards(stack, hand)

    def determine_stitch_winner(self, stack: Stack) -> Player:
        return self.heart_solo.determine_stitch_winner(stack)
