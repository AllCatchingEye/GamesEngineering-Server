from logic.gamemodes.gamemode import GameMode
from state.card import Card
from state.deck import DECK
from state.hand import Hand
from state.player import Player
from state.ranks import Rank
from state.stack import Stack
from state.suits import Suit


class GameModeSolo(GameMode):
    trumps: list[Card]

    def __init__(self, suit: Suit):
        super().__init__(suit)

        self.trumps = []

        for card in DECK.get_full_deck():
            if card.rank == Rank.OBER or card.rank == Rank.UNTER or card.suit == suit:
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

    def determine_stitch_winner(self, stack: Stack) -> Player:
        strongest_played_card = stack.get_played_cards()[0]
        for played_card in stack.get_played_cards()[1:]:
            if self.__card_is_stronger_than(played_card.get_card(), strongest_played_card.get_card()):
                strongest_played_card = played_card
        return strongest_played_card.get_player()

    def __card_is_stronger_than(self, card_one: Card, card_two: Card) -> bool:
        match card_one.rank:
            case Rank.OBER:
                match card_two.rank:
                    case Rank.OBER:
                        return card_one.get_suit().value < card_two.get_suit().value
                    case _:
                        return True
            case Rank.UNTER:
                match card_two.rank:
                    case Rank.OBER:
                        return False
                    case Rank.UNTER:
                        return card_one.get_suit().value < card_two.get_suit().value
                    case _:
                        return True
            case _:
                if [card_one, card_two] in self.trumps or [card_one, card_two] not in self.trumps:
                    return card_one.get_value() > card_two.get_value()
                elif card_one in self.trumps:
                    return True
                else:
                    return False
