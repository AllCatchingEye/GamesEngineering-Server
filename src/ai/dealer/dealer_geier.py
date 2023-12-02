from ai.dealer.dealer import Dealer
from ai.dealer.deck_manipulation import take, take_rank, take_trumps
from logic.gamemodes.gamemode_geier import GameModeGeier
from state.card import Card
from state.deck import DECK
from state.ranks import Rank
from state.suits import Suit


class DealerGeier(Dealer):
    def deal(self, suit: Suit | None) -> tuple[list[Card], list[list[Card]]]:
        trumps = GameModeGeier(suit).trumps

        deck = DECK.get_full_deck()
        self.rng.shuffle(deck)

        good_cards: list[Card] = []

        good_cards.append(take_rank(deck, Rank.OBER))
        good_cards.append(take_rank(deck, Rank.OBER))

        if suit is not None:
            good_cards.extend(take_trumps(deck, trumps, 2))

        good_cards.extend(take(deck, 8 - len(good_cards)))

        other_cards: list[list[Card]] = []

        for _ in range(3):
            other_cards.append(take(deck, 8))

        return good_cards, other_cards
