from logic.gamemodes.gamemode import GameMode
from state.card import Card
from state.deck import DECK
from state.hand import Hand
from state.ranks import Rank
from state.stack import Stack
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

    def get_playable_cards(self, stack: Stack, hand: Hand) -> list[Card]:
        # TODO: FIXME - this might not be correct (hopefully fixed now - has to be tested)

        if stack.is_empty():
            return hand.get_all_cards()

        # is trump round?
        trump_round = stack.get_first_card() in self.trumps

        if trump_round:
            # Check trumps
            playable_trumps = hand.get_all_trumps_in_deck(self.trumps)

            if len(playable_trumps) > 0:
                return playable_trumps
        else:
            # Same color
            played_suit = stack.get_first_card().get_suit()
            same_suit = hand.get_all_cards_for_suit(played_suit, self.trumps)
            if len(same_suit) > 0:
                return same_suit

        # Any card - free to choose
        return hand.get_all_cards()
