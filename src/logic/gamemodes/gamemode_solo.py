from logic.gamemodes.gamemode import GameMode
from state.card import Card
from state.deck import DECK
from state.hand import Hand
from state.player import Player
from state.ranks import Rank
from state.stack import Stack
from state.suits import Suit


class GameModeSolo(GameMode):
    trumps: list[Card]

    def __init__(self, suit: Suit):
        super().__init__(suit)

        self.trumps = []

        for card in DECK.get_full_deck():
            if card.rank == Rank.OBER or card.rank == Rank.UNTER or card.suit == suit:
                self.trumps.append(card)

    def get_trump_cards(self) -> list[Card]:
        return self.trumps

    def get_playable_cards(self, stack: Stack, hand: Hand) -> list[Card]:
        # TODO: FIXME - this might not be correct

        if stack.is_empty():
            return hand.get_all_cards()

        # is trump round?
        trump_round = stack.get_first_card() in self.trumps

        if not trump_round:
            # First same color
            played_suit = stack.get_first_card().get_suit()
            same_suit = hand.get_all_cards_for_suit(played_suit)
            if len(same_suit) > 0:
                return same_suit

        # Check trumps
        playable_trumps = hand.get_all_trumps_in_deck(self.trumps)

        if len(playable_trumps) > 0:
            return playable_trumps

        # Third any card
        return hand.get_all_cards()

    def determine_stitch_winner(self, stack: Stack) -> Player:
        # TODO: implement
        return stack.get_played_cards()[0].get_player()
