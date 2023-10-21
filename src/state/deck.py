from state.cards import Card
import state.suits
import state.ranks

class Deck():
    def __init__(self) -> None:
        self.deck: list[Card] = self.__get_full_deck()

    def __get_full_deck(self) -> list[Card]:
        suits: list[state.suits.Suit] = state.suits.get_all_suits()
        deck: list[Card] = []
        for suit in suits:
            full_suit = self.get_cards_for(suit)
            deck += full_suit

        return deck

    def get_cards_for(self, suit: state.suits.Suit) -> list[Card]:
        ranks: list[state.ranks.Rank] = state.ranks.get_all_ranks()
        full_suit: list[Card] = []
        for rank in ranks:
            card = Card(rank, suit)
            full_suit.append(card)

        return full_suit

    def get_ranks_of(self, rank: state.ranks.Rank) -> list[Card]:
        rank_cards: list[Card] = []
        for card in self.deck:
            if card.rank == rank:
                rank_cards.append(card)

        return rank_cards

    def get_suits_of(self, suit: state.suits.Suit) -> list[Card]:
        suit_cards: list[Card] = []
        for card in self.deck:
            if card.suit == suit:
                suit_cards.append(card)

        return suit_cards

    def get_deck(self) -> list[Card]:
        return self.deck.copy()


