from logic.gamemodes.gamemode import GameMode
from state.card import Card
from state.deck import DECK
from state.ranks import Rank
from state.suits import Suit


class GameModeSolo(GameMode):
    def __init__(self, suit: Suit):
        super().__init__(suit)

        self.trumps = DECK.get_cards_by_rank(Rank.OBER) + DECK.get_cards_by_rank(Rank.UNTER)

        for card in DECK.get_cards_by_suit(suit):
            if card not in self.trumps:
                self.trumps.append(card)

    def get_trump_cards(self) -> list[Card]:
        return self.trumps
