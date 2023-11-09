from logic.gamemodes.gamemode import GameMode
from state.card import Card
from state.deck import DECK
from state.hand import Hand
from state.ranks import Rank
from state.stack import Stack
from state.suits import Suit


class GameModeSauspiel(GameMode):
    def __init__(self, suit: Suit):
        if suit == Suit.HERZ:
            raise ValueError("Herz Ass cannot be searching during sauspiel")
        trumps_init = DECK.get_cards_by_rank(Rank.OBER) + DECK.get_cards_by_rank(
            Rank.UNTER
        )
        for card in DECK.get_cards_by_suit(Suit.HERZ):
            if card not in trumps_init:
                trumps_init.append(card)
        super().__init__(suit, trumps_init)

    def get_playable_cards(self, stack: Stack, hand: Hand) -> list[Card]:
        if stack.is_empty():
            return hand.get_all_cards()

        if self.suit is None:
            raise ValueError(
                "Sauspiel suit cannot be None, but was somehow set to None"
            )

        ace: Card | None = hand.get_card_of_rank_and_suit(self.suit, Rank.ASS)

        if ace is not None:
            if (
                stack.get_first_card() not in self.trumps
                and stack.get_first_card().suit == self.suit
            ):
                # played ass is being searched
                return [ace]

            playable_cards = super().get_playable_cards(stack, hand)
            if ace in playable_cards:
                playable_cards.remove(ace)
            return playable_cards

        return super().get_playable_cards(stack, hand)
