from state.hand import Hand
from state.ranks import Rank
from state.suits import Suit, get_all_suits
from state.card import Card
from state.gametypes import Gametype


def get_playable_gametypes(hand: Hand, trumps: list[Card]) -> list[Gametype]:
    """Returns all playable gametype with that hand."""
    types = [Gametype.SOLO]
    sauspiel_suits = get_all_suits()
    sauspiel_suits.remove(Suit.HERZ)
    for suit in sauspiel_suits:
        suit_cards = hand.get_all_cards_for_suit(suit, trumps)
        if len(suit_cards) > 0 and Card(Rank.ASS, suit) not in suit_cards:
            types.append(Gametype.SAUSPIEL)
    # TODO check other gametypes than sauspiel
    return types
