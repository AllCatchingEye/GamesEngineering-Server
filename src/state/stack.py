from dataclasses import dataclass

from state.card import Card
from state.player import Player
from state.ranks import Rank, get_value_of
from state.suits import Suit


@dataclass
class PlayedCard:
    card: Card
    player: Player
    """Represents a card played by a player during the game."""

    def __init__(self, card: Card, player: Player) -> None:
        """
        Initialize a PlayedCard instance.

        Args:
            card (Card): The card played by the player.
            player (Player): The player who played the card.
        """
        self.card = card
        self.player = player

    def get_value(self) -> int:
        return self.card.get_value()

    def get_card(self) -> Card:
        return self.card

    def get_suit(self) -> Suit:
        return self.card.get_suit()

    def get_rank(self) -> Rank:
        return self.card.get_rank()

    def get_player(self) -> Player:
        return self.player

    def __str__(self) -> str:
        return f"{self.card} by player {self.player.id}"

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, PlayedCard):
            return False
        return self.card == __value.card and self.player == __value.player


@dataclass
class Stack:
    """Represents a stack of cards played during a round."""

    def __init__(self) -> None:
        """
        Initialize a Stack instance.
        """
        self.played_cards: list[PlayedCard] = []
        self.value = 0

    def add_card(self, card: Card, player: Player) -> None:
        """
        Add a card played by a player to the stack.

        Args:
            card (Card): The card played.
            player (Player): The player who played the card.
        """
        played_card = PlayedCard(card, player)
        self.value += get_value_of(card.get_rank())
        self.played_cards.append(played_card)

    def is_empty(self) -> bool:
        return len(self.played_cards) == 0

    def get_value(self) -> int:
        return self.value

    def get_first_card(self) -> Card:
        top_card: PlayedCard = self.played_cards[0]
        return top_card.get_card()

    def get_played_cards(self) -> list[PlayedCard]:
        return self.played_cards

    def __str__(self) -> str:
        return f"Stack: {self.played_cards}"
