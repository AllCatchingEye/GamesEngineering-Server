from dataclasses import dataclass

from state.card import Card
from state.ranks import Rank, get_all_ranks
from state.suits import Suit, get_all_suits


@dataclass
class Deck:
    deck: list[Card]

    def __init__(self) -> None:
        self.deck = self.__create_full_deck()

    def __create_full_deck(self) -> list[Card]:
        """Creates a full deck of cards for the game

        Returns:
            A full deck of cards, unshuffled
        """
        all_suits: list[Suit] = get_all_suits()
        deck: list[Card] = []
        for suit in all_suits:
            full_suit: list[Card] = self.__create_cards_for_suit(suit)
            deck += full_suit

        return deck

    def __create_cards_for_suit(self, suit: Suit) -> list[Card]:
        """Creates all cards for a given suit

        Args:
            suit: The suit for which the cards will be created

        Returns:
            A list of cards for the given suit
        """
        all_ranks: list[Rank] = get_all_ranks()
        full_suit: list[Card] = []
        for rank in all_ranks:
            card = Card(suit, rank)
            full_suit.append(card)

        return full_suit

    def get_cards_by_rank(self, rank: Rank) -> list[Card]:
        """Returns all cards of a specific rank

        Args:
            rank: The rank for which you want the cards

        Returns:
            A list of cards for the given rank
        """
        rank_cards: list[Card] = []
        for card in self.deck:
            if card.get_rank() == rank:
                rank_cards.append(card)

        return rank_cards

    def get_cards_by_suit(self, suit: Suit) -> list[Card]:
        """Returns all cards of a specific suit

        Args:
            suit: The suit for which you want the cards

        Returns:
            A list of cards for the given suit
        """
        suit_cards: list[Card] = []
        for card in self.deck:
            if card.suit == suit:
                suit_cards.append(card)

        return suit_cards

    def get_full_deck(self) -> list[Card]:
        """Returns a full deck

        Returns:
            A full deck, unshuffled
        """
        return self.deck.copy()


DECK = Deck()
