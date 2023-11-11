from state.card import Card
from state.gametypes import Gametype
from state.ranks import Rank, get_all_ranks
from state.suits import Suit, get_all_suits

NUM_RANKS = len(get_all_ranks())
NUM_SUITS = len(get_all_suits())
NUM_CARDS = NUM_RANKS * NUM_SUITS


def card_to_suit_offset(suit: Suit):
    return suit.value


def card_to_rank_value(rank: Rank):
    # See also `train_classifier.ipynb` > Example Execution > `category_mapping`
    match rank:
        case Rank.OBER:
            return 0
        case Rank.UNTER:
            return 1
        case Rank.ASS:
            return 2
        case Rank.ZEHN:
            return 3
        case Rank.KOENIG:
            return 4
        case Rank.NEUN:
            return 5
        case Rank.ACHT:
            return 6
        case Rank.SIEBEN:
            return 7


def card_to_nn_input_values_index(card: Card) -> int:
    suit_offset = card_to_suit_offset(card.get_suit())
    rank_value = card_to_rank_value(card.get_rank())
    # See also `train_classifier.ipynb` > Example Execution > `category_mapping`
    return suit_offset * NUM_RANKS + rank_value


def card_to_nn_input_values(hand_cards: list[Card]) -> list[int]:
    nn_input = [0] * NUM_CARDS
    for card in hand_cards:
        card_index = card_to_nn_input_values_index(card)
        nn_input[card_index] = 1
    return nn_input


def code_to_game_type(game_type_code: int) -> Gametype:
    # See also `train_classifier.ipynb` > Example Execution > `category_mapping`
    if game_type_code <= 3:
        return Gametype.FARBGEIER
    if game_type_code <= 7:
        return Gametype.FARBWENZ
    if game_type_code == 8:
        return Gametype.GEIER
    if game_type_code <= 11:
        return Gametype.SAUSPIEL
    if game_type_code <= 15:
        return Gametype.SOLO
    if game_type_code == 16:
        return Gametype.WENZ
    return Gametype.RAMSCH
