from logic.gamemodes.gamemode import GameMode
from state.deck import DECK
from state.ranks import Rank
from state.suits import Suit


class GameModeGeier(GameMode):
    def __init__(self, suit: Suit | None):
        trumps_init = DECK.get_cards_by_rank(Rank.OBER)
        if suit is not None:
            for card in DECK.get_cards_by_suit(suit):
                if card not in trumps_init:
                    trumps_init.append(card)

        super().__init__(suit, trumps_init)
