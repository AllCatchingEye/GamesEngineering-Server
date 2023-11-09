from state.card import Card
from state.deck import DECK
from state.gametypes import Gametype
from state.hand import Hand
from state.ranks import Rank
from state.suits import Suit, get_all_suits


def get_playable_gametypes(
    hand: Hand, plays_ahead: int
) -> list[tuple[Gametype, Suit | None]]:
    """Returns all playable gametypes with that hand."""
    types: list[tuple[Gametype, Suit | None]] = []
    # Gametypes Solo
    for suit in get_all_suits():
        types.append((Gametype.SOLO, suit))
    # Gametypes Wenz
    types += __get_practical_gametypes_wenz_geier(
        hand, Rank.UNTER, Gametype.FARBWENZ, Gametype.WENZ
    )
    # Gametypes Geier
    types += __get_practical_gametypes_wenz_geier(
        hand, Rank.OBER, Gametype.FARBGEIER, Gametype.GEIER
    )
    # Gametypes Sauspiel
    if plays_ahead == 0:
        sauspiel_suits = get_all_suits()
        sauspiel_suits.remove(Suit.HERZ)
        for suit in sauspiel_suits:
            suit_cards = hand.get_all_cards_for_suit(
                suit,
                DECK.get_cards_by_rank(Rank.OBER) + DECK.get_cards_by_rank(Rank.UNTER),
            )
            if len(suit_cards) > 0 and Card(suit, Rank.ASS) not in suit_cards:
                types.append((Gametype.SAUSPIEL, suit))
    return types


def __get_practical_gametypes_wenz_geier(
    hand: Hand, rank: Rank, game_type_suit: Gametype, game_type_no_suit: Gametype
) -> list[tuple[Gametype, Suit | None]]:
    practical_types: list[tuple[Gametype, Suit | None]] = []
    if len(hand.get_all_trumps_in_deck(DECK.get_cards_by_rank(rank))) > 0:
        practical_types.append((game_type_no_suit, None))
        for suit in get_all_suits():
            if (
                len(
                    hand.get_all_cards_for_suit(suit, DECK.get_cards_by_rank(Rank.OBER))
                )
                > 0
            ):
                practical_types.append((game_type_suit, suit))
    return practical_types
