from ai.dealer.dealer import Dealer
from ai.dealer.deck_manipulation import take, take_random_good_cards, take_rank, take_trumps
from logic.gamemodes.gamemode_geier import GameModeGeier
from state.card import Card
from state.deck import DECK
from state.ranks import Rank
from state.suits import Suit
import random


class DealerGeier(Dealer):
    def deal(self, suit: Suit | None) -> tuple[list[Card], list[list[Card]]]:
        is_farbgeier = suit is not None

        min_trumps = 3 if is_farbgeier else 2
        num_good_cards = 6 if is_farbgeier else 4

        num_trumps = random.randint(min_trumps, num_good_cards)
        num_other_good_cards = num_good_cards - num_trumps

        trumps = GameModeGeier(suit).trumps

        deck = DECK.get_full_deck()
        self.rng.shuffle(deck)

        good_cards: list[Card] = []

        if is_farbgeier:
            good_cards.extend(take_trumps(deck, trumps, num_trumps))
        else:
            good_cards.extend([take_rank(deck, Rank.OBER)]*num_trumps)
        
        other_good_cards = take_random_good_cards(deck, num_other_good_cards)
        good_cards.extend(other_good_cards)

        good_cards.extend(take(deck, 8 - len(good_cards)))

        other_cards: list[list[Card]] = []

        for _ in range(3):
            other_cards.append(take(deck, 8))

        return good_cards, other_cards
