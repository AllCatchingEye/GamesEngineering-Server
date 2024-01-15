from ai.dealer.dealer import Dealer
from ai.dealer.deck_manipulation import take, take_matching
from state.card import Card
from state.deck import DECK
from state.ranks import Rank
from state.suits import Suit


class DealerRamsch(Dealer):
    def deal(self, suit: Suit | None) -> tuple[list[Card], list[list[Card]]]:
        if suit is not None:
            raise ValueError("Ramsch has no suit")

        deck = DECK.get_full_deck()
        self.rng.shuffle(deck)

        bad_cards: list[Card] = []

        bad_cards.extend(
            take_matching(
                deck, lambda c: c.rank in [Rank.SIEBEN, Rank.ACHT, Rank.NEUN], 2
            )
        )

        bad_cards.extend(take(deck, 8 - len(bad_cards)))

        other_cards: list[list[Card]] = []

        for _ in range(3):
            other_cards.append(take(deck, 8))

        return bad_cards, other_cards
