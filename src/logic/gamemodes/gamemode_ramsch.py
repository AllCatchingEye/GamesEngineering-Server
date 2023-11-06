from logic.gamemodes.gamemode import GameMode
from logic.gamemodes.gamemode_solo import GameModeSolo
from state.card import Card
from state.deck import DECK
from state.hand import Hand
from state.player import Player
from state.ranks import Rank
from state.stack import Stack
from state.suits import Suit


class GameModeRamsch(GameMode):
    def __init__(self):
        trumps_init = DECK.get_cards_by_rank(Rank.OBER) + DECK.get_cards_by_rank(Rank.UNTER)
        for card in DECK.get_cards_by_suit(Suit.HERZ):
            if card not in trumps_init:
                trumps_init.append(card)

        super().__init__(Suit.HERZ, trumps_init)
