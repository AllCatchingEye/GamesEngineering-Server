from logic.gamemodes.gamemode import GameMode
from state.deck import DECK
from state.ranks import Rank
from state.suits import Suit


class GameModeSolo(GameMode):
    def __init__(self, suit: Suit):
        trumps_init = DECK.get_cards_by_rank(Rank.OBER) + DECK.get_cards_by_rank(
            Rank.UNTER
        )
        for card in DECK.get_cards_by_suit(suit):
            if card not in trumps_init:
                trumps_init.append(card)

        super().__init__(suit, trumps_init)
