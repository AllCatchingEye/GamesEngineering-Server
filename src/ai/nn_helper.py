from state.card import Card
from state.gametypes import Gametype
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
    value = rank_values.get(rank)
    assert value is not None, "The given Rank " + rank.name + " could not be found"
    return value


def card_to_nn_input_values_index(card: Card) -> int:
    suit_offset = card_to_suit_offset(card.get_suit())
    rank_value = card_to_rank_value(card.get_rank())
    return suit_offset * NUM_RANKS + rank_value


def nn_output_code_to_card(code: int) -> Card:
    suit_code = code // NUM_RANKS
    rank_code = code % NUM_RANKS
    return Card(suits[suit_code], list(rank_values.keys())[rank_code])


def card_to_nn_input_values(hand_cards: list[Card]) -> list[int]:
    nn_input = [0] * NUM_CARDS
    for card in hand_cards:
        card_index = card_to_nn_input_values_index(card)
        nn_input[card_index] = 1
    return nn_input


def code_to_game_type(game_type_code: int) -> tuple[Gametype, Suit | None]:
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
