from state.cards import Card
from state.ranks import Rank
from state.suits import Suit


class Hand:
    def __init__(self, cards: list[Card]) -> None:
        self.cards: list[Card] = cards

    def get_card(self, index: int) -> Card:
        return self.cards[index]

    def get_all_cards(self) -> list[Card]:
        return self.cards

    def remove_card(self, card: Card) -> None:
        self.cards.remove(card)

    def show_cards_for_suit(self, suit: Suit) -> None:
        """Prints all cards currently in the hand for a given suit"""
        for index, card in enumerate(self.cards):
            if card.get_suit() == suit:
                print(f"{index}: {card}")

    def has_card_for_suit(self, suit: Suit) -> bool:
        """Checks if the hand has a card of the given suit in it"""
        for card in self.cards:
            if suit == card.get_suit():
                return True
        return False

    def get_all_cards_for_suit(self, suit: Suit) -> list[tuple[int, Card]]:
        """Returns all cards in the hand for the given suit"""
        suits: list[tuple[int, Card]] = []
        for index, card in enumerate(self.cards):
            if card.get_suit() == suit:
                suits.append((index, card))
        return suits

    def show_cards_for_ranks(self, ranks: list[Rank]) -> None:
        """Prints all cards in the hand whose rank matches the given list of ranks"""
        for rank in ranks:
            self.__show_rank(rank)

    def __show_rank(self, rank: Rank) -> None:
        """Prints all cards in the hand for a given rank"""
        for index, card in enumerate(self.cards):
            if card.get_rank() == rank:
                print(f"{index}: {card}")

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

    def get_all_trumps_in_deck(self, trumps: list[Card]) -> list[tuple[int, Card]]:
        """Returns all trumps in the hand that match the given trumps list."""
        available_trumps: list[tuple[int, Card]] = []
        for index, card in enumerate(self.cards):
            if card in trumps:
                available_trumps.append((index, card))
        return available_trumps

    def show_all_cards(self) -> None:
        """Prints all cards in the hand."""
        for index, card in enumerate(self.cards):
            print(f"{index}: {card}")
