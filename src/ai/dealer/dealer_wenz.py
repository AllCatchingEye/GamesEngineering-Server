import random
from ai.dealer.dealer import Dealer
from ai.dealer.deck_manipulation import take, take_matching, take_rank, take_trumps
from logic.gamemodes.gamemode_wenz import GameModeWenz
from state.card import Card
from state.deck import DECK
from state.ranks import Rank
from state.suits import Suit


class DealerWenz(Dealer):
    def deal(self, suit: Suit | None) -> tuple[list[Card], list[list[Card]]]:

        is_farbwenz =  suit is not None

        min_trumps = 3 if is_farbwenz else 2
        num_good_cards = 6 if is_farbwenz else 4

        num_trumps = random.randint(min_trumps, num_good_cards)
        num_other_good_cards = num_good_cards - num_trumps
       
        deck = DECK.get_full_deck()
        self.rng.shuffle(deck)

        good_cards: list[Card] = []

        trumps_farbwenz = GameModeWenz(suit).trumps

        if is_farbwenz:
            good_cards.extend(take_trumps(deck, trumps_farbwenz, num_trumps))
        else:
            good_cards.extend( [take_rank(deck, Rank.UNTER)]*num_trumps)

        other_good_cards = take_matching(
                deck, lambda c: c.rank in [Rank.ASS, Rank.KOENIG, Rank.ZEHN], num_other_good_cards
            )
        good_cards.extend(other_good_cards)
            
        good_cards.extend(take(deck, 8 - len(good_cards)))

        other_cards: list[list[Card]] = []

        for _ in range(3):
            other_cards.append(take(deck, 8))

        return good_cards, other_cards
