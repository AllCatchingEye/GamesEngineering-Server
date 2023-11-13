from state.card import Card
from state.deck import DECK
from state.gametypes import Gametype, GameGroup
from state.hand import Hand
from state.ranks import Rank
from state.suits import Suit, get_all_suits


def get_playable_gametypes(
        hand: Hand, available_game_groups: list[GameGroup]
) -> list[tuple[Gametype, Suit | None]]:
    """Returns all playable gametypes with that hand."""
    types: list[tuple[Gametype, Suit | None]] = []

    if GameGroup.HIGH_SOLO in available_game_groups:
        # Gametypes Solo
        for suit in get_all_suits():
            types.append((Gametype.SOLO, suit))

    if GameGroup.MID_SOLO in available_game_groups:
        types.append((Gametype.WENZ, None))
        types.append((Gametype.GEIER, None))

    if GameGroup.LOW_SOLO in available_game_groups:
        # Gametypes Wenz
        for suit in get_all_suits():
            types.append((Gametype.FARBWENZ, suit))
        # Gametypes Geier
        for suit in get_all_suits():
            types.append((Gametype.FARBGEIER, suit))

    if GameGroup.ALL in available_game_groups:
        # Gametypes Sauspiel
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
