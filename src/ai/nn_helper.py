from state.card import Card
from state.gametypes import Gametype
from state.ranks import Rank, get_all_ranks
from state.suits import Suit, get_all_suits

RANKS = get_all_ranks()
SUITS = get_all_suits()

NUM_RANKS = len(RANKS)
NUM_SUITS = len(SUITS)
NUM_CARDS = NUM_RANKS * NUM_SUITS
NUM_STACK_CARDS = 4
NUM_ROUNDS = 8

RANK_VALUES = {
    Rank.OBER: 0,
    Rank.UNTER: 1,
    Rank.ASS: 2,
    Rank.ZEHN: 3,
    Rank.KOENIG: 4,
    Rank.NEUN: 5,
    Rank.ACHT: 6,
    Rank.SIEBEN: 7,
}


def get_suit_value(suit: Suit) -> int:
    return suit.value


def get_rank_value(rank: Rank) -> int:
    value = RANK_VALUES.get(rank)
    assert value is not None, "The given Rank " + rank.name + " could not be found"
    return value


def one_hot_encode_card(card: Card) -> list[int]:
    result = [0] * NUM_CARDS
    suit_offset = get_suit_value(card.get_suit())
    rank_value = get_rank_value(card.get_rank())
    index = suit_offset * NUM_RANKS + rank_value
    result[index] = 1

    return result


def get_one_hot_encoding_index_from_card(card: Card) -> int:
    return get_suit_value(card.get_suit()) * NUM_RANKS + get_rank_value(card.get_rank())


def one_hot_encode_cards(hand_cards: list[Card]) -> list[int]:
    result = [0] * NUM_CARDS
    for card in hand_cards:
        encoded_card = one_hot_encode_card(card)
        result = [old_val | encoded_card[index] for index, old_val in enumerate(result)]
    return result


def decode_game_type(game_type_code: int) -> tuple[Gametype, Suit | None]:
    match game_type_code:
        case 0:
            return (Gametype.FARBGEIER, Suit.EICHEL)
        case 1:
            return (Gametype.FARBGEIER, Suit.GRAS)
        case 2:
            return (Gametype.FARBGEIER, Suit.HERZ)
        case 3:
            return (Gametype.FARBGEIER, Suit.SCHELLEN)

        case 4:
            return (Gametype.FARBWENZ, Suit.EICHEL)
        case 5:
            return (Gametype.FARBWENZ, Suit.GRAS)
        case 6:
            return (Gametype.FARBWENZ, Suit.HERZ)
        case 7:
            return (Gametype.FARBWENZ, Suit.SCHELLEN)

        case 8:
            return (Gametype.GEIER, None)

        case 9:
            return (Gametype.SAUSPIEL, Suit.EICHEL)
        case 10:
            return (Gametype.SAUSPIEL, Suit.GRAS)
        case 11:
            return (Gametype.SAUSPIEL, Suit.SCHELLEN)

        case 12:
            return (Gametype.SOLO, Suit.EICHEL)
        case 13:
            return (Gametype.SOLO, Suit.GRAS)
        case 14:
            return (Gametype.SOLO, Suit.HERZ)
        case 15:
            return (Gametype.SOLO, Suit.SCHELLEN)

        case 16:
            return (Gametype.WENZ, None)
        case _:
            return (Gametype.RAMSCH, None)
