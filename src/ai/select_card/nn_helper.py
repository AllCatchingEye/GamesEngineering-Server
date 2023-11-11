from state.card import Card
from state.ranks import Rank, get_all_ranks
from state.suits import Suit, get_all_suits

ranks = get_all_ranks()
suits = get_all_suits()

NUM_RANKS = len(ranks)
NUM_SUITS = len(suits)
NUM_CARDS = NUM_RANKS * NUM_SUITS

rank_values = {
    Rank.OBER: 0,
    Rank.UNTER: 1,
    Rank.ASS: 2,
    Rank.ZEHN: 3,
    Rank.KOENIG: 4,
    Rank.NEUN: 5,
    Rank.ACHT: 6,
    Rank.SIEBEN: 7,
}


def card_to_suit_offset(suit: Suit):
    return suit.value


def card_to_rank_value(rank: Rank):
    return rank_values.get(rank)


def card_to_nn_input_values_index(card: Card) -> int:
    suit_offset = card_to_suit_offset(card.get_suit())
    rank_value = card_to_rank_value(card.get_rank())
    return suit_offset * NUM_RANKS + rank_value


def nn_output_code_to_card(code: int) -> Card:
    suit_code = code // NUM_RANKS
    rank_code = code % NUM_SUITS
    return Card(suits[suit_code], ranks[rank_code])


def card_to_nn_input_values(hand_cards: list[Card]) -> list[int]:
    nn_input = [0] * NUM_CARDS
    for card in hand_cards:
        card_index = card_to_nn_input_values_index(card)
        nn_input[card_index] = 1
    return nn_input
