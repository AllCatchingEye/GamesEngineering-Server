from state.card import Card
from state.gametypes import Gametype
from state.ranks import Rank
from state.suits import Suit, get_all_suits


class Hand:
    def __init__(self, cards: list[Card]) -> None:
        self.cards: list[Card] = cards

    def get_card(self, index: int) -> Card:
        return self.cards[index]

    def get_all_cards(self) -> list[Card]:
        return self.cards

    def remove_card(self, card: Card) -> None:
        self.cards.remove(card)

    def has_card_for_suit(self, suit: Suit, trumps: list[Card]) -> bool:
        """Checks if the hand has a card of the given suit in it that is not a trump"""
        for card in self.cards:
            if suit == card.get_suit() and card not in trumps:
                return True
        return False

    def get_all_cards_for_suit(self, suit: Suit, trumps: list[Card]) -> list[Card]:
        """Returns all cards in the hand for the given suit that are not trumps"""
        suits: list[Card] = []
        for card in self.cards:
            if card.get_suit() == suit and card not in trumps:
                suits.append(card)
        return suits

    def has_cards_for_ranks(self, ranks: list[Rank]) -> bool:
        """Checks if the hand has a card of the given rank in it."""
        for rank in ranks:
            if self.__has_cards_for_rank(rank):
                return True
        return False

    def __has_cards_for_rank(self, rank: Rank) -> bool:
        """Checks if the hand has a card of the given suit in it"""
        for card in self.cards:
            if rank == card.get_rank():
                return True
        return False

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

    def get_all_trumps_in_deck(self, trumps: list[Card]) -> list[Card]:
        """Returns all trumps in the hand that match the given trumps list."""
        available_trumps: list[Card] = []
        for card in self.cards:
            if card in trumps:
                available_trumps.append(card)
        return available_trumps

    def get_playable_gametypes(self, trumps: list[Card]) -> list[Gametype]:
        """Returns all playable gametype with that hand."""
        types = [Gametype.SOLO]
        sauspiel_suits = get_all_suits()
        sauspiel_suits.remove(Suit.HERZ)
        for suit in sauspiel_suits:
            suit_cards = self.get_all_cards_for_suit(suit, trumps)
            if len(suit_cards) > 0 and Card(suit, Rank.ASS) not in suit_cards:
                types.append(Gametype.SAUSPIEL)
        # TODO check other gametypes than sauspiel
        return types

    def __str__(self) -> str:
        return str(self.cards)

    def __repr__(self) -> str:
        return str(self.cards)
