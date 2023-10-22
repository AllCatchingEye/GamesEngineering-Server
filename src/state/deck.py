from cards import Card
import suits
import ranks

class Deck():
    def __init__(self) -> None:
        self.deck: list[Card] = self.__get_full_deck()

    def __get_full_deck(self) -> list[Card]:
        all_suits: list[suits.Suit] = suits.get_all_suits()
        deck: list[Card] = []
        for suit in all_suits:
            full_suit = self.get_cards_for(suit)
            deck += full_suit

        return deck

    def get_cards_for(self, suit: suits.Suit) -> list[Card]:
        all_ranks: list[ranks.Rank] = ranks.get_all_ranks()
        full_suit: list[Card] = []
        for rank in all_ranks:
            card = Card(rank, suit)
            full_suit.append(card)

        return full_suit

    def get_ranks_of(self, rank: ranks.Rank) -> list[Card]:
        rank_cards: list[Card] = []
        for card in self.deck:
            if card.get_rank() == rank:
                rank_cards.append(card)

        return rank_cards

    def get_suits_of(self, suit: suits.Suit) -> list[Card]:
        suit_cards: list[Card] = []
        for card in self.deck:
            if card.suit == suit:
                suit_cards.append(card)

        return suit_cards

    def get_deck(self) -> list[Card]:
        return self.deck.copy()


