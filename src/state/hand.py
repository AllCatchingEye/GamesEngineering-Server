from dataclasses import dataclass

from state.card import Card
from state.ranks import Rank
from state.suits import Suit


@dataclass
class Hand:
    cards: list[Card]

    def __init__(self, cards: list[Card]) -> None:
        self.cards = cards

    def get_card(self, index: int) -> Card:
        return self.cards[index]

    def get_all_cards(self) -> list[Card]:
        return self.cards.copy()

    def remove_card(self, card: Card) -> None:
        self.cards.remove(card)

    def get_all_cards_for_suit(self, suit: Suit, trumps: set[Card]) -> list[Card]:
        """Returns all cards in the hand for the given suit that are not trumps"""
        suits: list[Card] = []
        for card in self.cards:
            if card.get_suit() == suit and card not in trumps:
                suits.append(card)
        return suits

    def get_all_cards_for_rank(self, rank: Rank) -> list[Card]:
        """Returns all cards in the hand for the given rank"""
        ranks: list[Card] = []
        for card in self.cards:
            if card.get_rank() == rank:
                ranks.append(card)
        return ranks

    def has_card_of_rank_and_suit(self, suit: Suit, rank: Rank) -> bool:
        """Checks if the hand has a card of the given suit and rank in it"""
        for card in self.cards:
            if suit == card.get_suit() and rank == card.get_rank():
                return True
        return False

    def get_card_of_rank_and_suit(self, suit: Suit, rank: Rank) -> Card | None:
        """Returns the card of the given suit and rank in it or else None"""
        for card in self.cards:
            if suit == card.get_suit() and rank == card.get_rank():
                return card

        return None

    def get_all_trumps_in_deck(self, trumps: set[Card]) -> list[Card]:
        """Returns all trumps in the hand that match the given trumps list."""
        available_trumps: list[Card] = []
        for card in self.cards:
            if card in trumps:
                available_trumps.append(card)
        return available_trumps

    def get_all_non_trumps_in_deck(self, trumps: list[Card]) -> list[Card]:
        """Returns all non trumps in the hand that do not match the given trumps list."""
        non_trumps: list[Card] = []
        for card in self.cards:
            if card not in trumps:
                non_trumps.append(card)
        return non_trumps

    def __str__(self) -> str:
        return str(self.cards)

    def __repr__(self) -> str:
        return str(self.cards)
