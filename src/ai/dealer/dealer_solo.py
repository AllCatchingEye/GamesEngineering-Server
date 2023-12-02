from ai.dealer.dealer import Dealer
from ai.dealer.deck_manipulation import take, take_trumps
from logic.gamemodes.gamemode_solo import GameModeSolo
from state.card import Card
from state.deck import DECK
from state.suits import Suit


class DealerSolo(Dealer):
    def deal(self, suit: Suit | None) -> tuple[list[Card], list[list[Card]]]:
        if suit is None or suit is Suit.HERZ:
            raise ValueError("Suit must be one of EICHEL, GRAS, or SCHELLEN")

        trumps = GameModeSolo(suit).trumps

        deck = DECK.get_full_deck()
        self.rng.shuffle(deck)

        good_cards: list[Card] = []

        good_cards.extend(take_trumps(deck, trumps, 4))

        good_cards.extend(take(deck, 8 - len(good_cards)))

        other_cards: list[list[Card]] = []

        for _ in range(3):
            other_cards.append(take(deck, 8))

        return good_cards, other_cards
