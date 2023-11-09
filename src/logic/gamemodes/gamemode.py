from abc import ABC
from typing import Tuple, List, Any

from state.card import Card
from state.hand import Hand
from state.player import Player
from state.stack import Stack
from state.suits import Suit


class GameMode(ABC):
    suit: Suit | None
    trumps: list[Card]

    def __init__(self, suit: Suit | None, trumps: list[Card]):
        self.suit = suit
        self.trumps = trumps

    def get_playable_cards(self, stack: Stack, hand: Hand) -> list[Card]:
        """Determine which cards can be played"""
        if stack.is_empty():
            return hand.get_all_cards()

        # is trump round?
        trump_round = stack.get_first_card() in self.trumps

        if trump_round:
            # Check trumps
            playable_trumps = hand.get_all_trumps_in_deck(self.trumps)

            if len(playable_trumps) > 0:
                return playable_trumps
        else:
            # Same color
            played_suit = stack.get_first_card().get_suit()
            same_suit = hand.get_all_cards_for_suit(played_suit, self.trumps)
            if len(same_suit) > 0:
                return same_suit

        # Any card - free to choose
        return hand.get_all_cards()

    def get_trump_cards(self) -> list[Card]:
        """Returns a list of all trump cards"""
        return self.trumps

    def determine_stitch_winner(self, stack: Stack) -> Player:
        """Returns the player who won the current stitch"""
        strongest_played_card = stack.get_played_cards()[0]
        for played_card in stack.get_played_cards()[1:]:
            if self.__card_is_stronger_than(
                    played_card.get_card(), strongest_played_card.get_card()
            ):
                strongest_played_card = played_card
        return strongest_played_card.get_player()

    def get_game_winner(self, play_party: list[list[Player]]) -> tuple[list[Player], list[int]]:
        """Determine the winner of the entire game."""
        party_points = []
        for i, party in enumerate(play_party):
            for player in play_party[0]:
                party_points[i] += player.points

        game_winner_index = party_points.index(max(party_points))
        return play_party[game_winner_index], party_points

    def __card_is_stronger_than(self, card_one: Card, card_two: Card) -> bool:
        """Helping-Method to determine which of two cards is stronger"""
        if card_one in self.trumps:
            if card_two in self.trumps:
                # Compare two trump-cards
                return self.trumps.index(card_one) < self.trumps.index(card_two)
            else:
                # Trump-Card wins over regular card
                return True
        elif card_one.get_suit() == card_two.get_suit():
            # Compare two cards of the same suit
            return card_one.get_rank().value > card_two.get_rank().value
        else:
            # Other card does not fulfill the leading suit
            return False
