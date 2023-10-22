from cards import Card
from ranks import Rank
from suits import Suit


class Hand():
    def __init__(self, cards: list[Card]) -> None:
        self.cards: list[Card] = cards

    def get_card(self, index: int) -> Card:
        return self.cards[index]

    def get_cards(self) -> list[Card]:
        return self.cards

    def remove_card(self, card: Card):
        self.cards.remove(card)

    def show_suits(self, suit: Suit):
        for index, card in enumerate(self.cards):
            if card.get_suit() == suit:
                print(f"{index}: {card}")

    def has_suit(self, suit: Suit) -> bool:
        for card in self.cards:
            if suit == card.get_suit():
                return True
        return False

    def show_ranks(self, ranks: list[Rank]):
        for rank in ranks:
            self.__show_rank(rank)

    def get_suits(self, suit: Suit) -> list[Card]:
        suits: list[Card] = []
        for card in self.cards:
            if card.get_suit() == suit:
                suits.append(card)
        return suits

    def get_ranks(self, ranks: list[Rank]) -> list[Card]:
        cards: list[Card] = []
        for rank in ranks:
            cards += self.__get_rank(rank)
        return cards

    def __get_rank(self, rank: Rank) -> list[Card]:
        ranks: list[Card] = []
        for card in self.cards:
            if card.get_rank() == rank:
                ranks.append(card)
        return ranks

    def __show_rank(self, rank: Rank):
        for index, card in enumerate(self.cards):
            if card.get_rank() == rank:
                print(f"{index}: {card}")

    def show(self):
        for index, card in enumerate(self.cards):
            print(f"{index}: {card}")

    def has_ranks(self, ranks: list[Rank]) -> bool:
        for rank in ranks:
            if self.__has_rank(rank):
                return True
        return False

    def __has_rank(self, rank: Rank) -> bool:
        for card in self.cards:
            if rank == card.get_rank():
                return True
        return False

