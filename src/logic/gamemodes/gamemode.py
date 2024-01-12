from abc import ABC

from state.card import Card
from state.hand import Hand
from state.player import Player, PlayerId
from state.stack import Stack
from state.suits import Suit


class GameMode(ABC):
    suit: Suit | None
    trumps: list[Card]
    trumps_set: set[Card]

    def __init__(self, suit: Suit | None, trumps: list[Card]):
        self.suit = suit
        self.trumps = trumps
        self.trumps_set = set(trumps)

    def get_playable_cards(self, stack: Stack, hand: Hand) -> list[Card]:
        """Determine which cards can be played"""
        if stack.is_empty():
            return hand.get_all_cards()

        # is trump round?
        trump_round = stack.get_first_card() in self.trumps_set

        if trump_round:
            # Check trumps
            playable_trumps = hand.get_all_trumps_in_deck(self.trumps_set)

            if len(playable_trumps) > 0:
                return playable_trumps
        else:
            # Same color
            played_suit = stack.get_first_card().get_suit()
            same_suit = hand.get_all_cards_for_suit(played_suit, self.trumps_set)
            if len(same_suit) > 0:
                return same_suit

        # Any card - free to choose
        return hand.get_all_cards()

    def get_trump_cards(self) -> set[Card]:
        """Returns a list of all trump cards"""
        return self.trumps_set

    def determine_stitch_winner(self, stack: Stack) -> PlayerId:
        """Returns the player who won the current stitch"""
        strongest_played_card = stack.get_played_cards()[0]
        for played_card in stack.get_played_cards()[1:]:
            if self.__card_is_stronger_than(
                played_card.get_card(), strongest_played_card.get_card()
            ):
                strongest_played_card = played_card
        stitch_winner_id = strongest_played_card.get_player()
        return stitch_winner_id

    def get_game_winner(
        self, play_party: list[list[Player]]
    ) -> tuple[list[Player], list[int]]:
        """Determine the winner of the entire game."""
        party_points: list[int] = [0] * len(play_party)
        for i, party in enumerate(play_party):
            for player in party:
                party_points[i] += player.points
        game_winner_index = (
            len(party_points) - 1 - party_points[::-1].index(max(party_points))
        )
        return play_party[game_winner_index], party_points

    def __card_is_stronger_than(self, card_one: Card, card_two: Card) -> bool:
        """Helping-Method to determine which of two cards is stronger"""
        if card_one in self.trumps_set:
            if card_two in self.trumps_set:
                # Compare two trump-cards
                return self.trumps.index(card_one) < self.trumps.index(card_two)
            # Trump-Card wins over regular card
            return True
        if (
            card_one.get_suit() == card_two.get_suit()
            and card_two not in self.trumps_set
        ):
            # Compare two cards of the same suit
            return card_one.get_rank().value > card_two.get_rank().value
        # Other card does not fulfill the leading suit
        return False

    def get_trump_suit(self) -> Suit:
        return self.suit
