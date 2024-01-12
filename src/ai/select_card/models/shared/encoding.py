from torch import Tensor

from ai.nn_helper import NUM_RANKS, RANK_VALUES, SUITS
from state.card import Card


def get_card_from_one_hot_encoding_index(index: int) -> Card:
    suit_code = index // NUM_RANKS
    rank_code = index % NUM_RANKS
    return Card(SUITS[suit_code], list(RANK_VALUES.keys())[rank_code])


def pick_highest_valid_card(output: Tensor, playable_cards: list[Card]):
    best_card = None
    optimal_q_value: float = float("-inf")
    tensor_list: list[float] = output.tolist()[0]
    for index, q_value in enumerate(tensor_list):
        decoded_card = get_card_from_one_hot_encoding_index(index)
        if q_value > optimal_q_value and decoded_card in playable_cards:
            optimal_q_value = q_value
            best_card = decoded_card

    if best_card is None:
        raise ValueError(
            "No playable card found. Playable cards: %s" % (playable_cards)
        )

    return best_card
