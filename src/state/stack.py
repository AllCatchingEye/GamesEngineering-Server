from state.cards import Card
from state.player import Player
from state.ranks import Rank, get_value_of
from state.suits import Suit


class PlayedCard:
    """Represents a card played by a player during the game."""

    def __init__(self, card: Card, player: Player) -> None:
        """
        Initialize a PlayedCard instance.

        Args:
            card (Card): The card played by the player.
            player (Player): The player who played the card.
        """
        self.card: Card = card
        self.player: Player = player

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


class Stack:
    """Represents a stack of cards played during a round."""

    def __init__(self, suit: Suit) -> None:
        """
        Initialize a Stack instance.

        Args:
            suit (Suit): The suit of the cards in the stack.
        """
        self.suit = suit
        self.played_cards: list[PlayedCard] = []
        self.strongest_card: PlayedCard
        self.value = 0

    def add_card(self, card: Card, player: Player) -> None:
        """
        Add a card played by a player to the stack.

        Args:
            card (Card): The card played.
            player (Player): The player who played the card.
        """
        played_card = PlayedCard(card, player)
        if self.is_empty():
            self.strongest_card = played_card
        else:
            self.__determine_stitch(played_card)

        self.value += get_value_of(card.get_rank())
        self.played_cards.append(played_card)

    def is_empty(self) -> bool:
        return len(self.played_cards) == 0

    def __determine_stitch(self, card: PlayedCard) -> None:
        """
        Determine the strongest card in the stack after a new card is played.

        Args:
            card (PlayedCard): The newly played card.
        """
        if self.__card_is_same_suit(card):
            # Neue Karte hat selbe farbe aber höheren rang
            if self.strongest_card.get_rank().value < card.get_rank().value:
                self.strongest_card = card
        # Neue Karte ist trumpf und momentane karte ist kein trumpf
        elif self.__card_is_trump(card) and not self.__card_is_trump(
            self.strongest_card
        ):
            self.strongest_card = card
        elif self.__card_is_trump(card) and self.__card_is_trump(self.strongest_card):
            # Neue Karte ist trumpf und momentane karte ist trumpf
            if self.strongest_card.get_suit().value < card.get_suit().value:
                self.strongest_card = card
            # Trumpf wird geschlagen von stärkerer karte gleicher Farbe
            if self.strongest_card.get_rank().value < card.get_rank().value:
                self.strongest_card = card
            # Different color but no trump
        elif self.strongest_card.get_suit().value < card.get_suit().value:
            self.strongest_card = card

        print("The strongest card currently is:")
        print(f"{self.strongest_card.get_card()}")
        print(f"It was played by player {self.strongest_card.get_player().get_id()}")

    def __card_is_trump(self, card: PlayedCard) -> bool:
        return card.get_rank() in [Rank.OBER, Rank.UNTER]

    def __card_is_same_suit(self, card: PlayedCard) -> bool:
        return self.strongest_card.get_suit() == card.get_suit()

    def get_value(self) -> int:
        return self.value

    def get_suit(self) -> Suit:
        return self.suit

    def get_top_card(self) -> Card:
        top_card: PlayedCard = self.played_cards[-1]
        return top_card.get_card()

    def get_winner(self) -> Player:
        return self.strongest_card.get_player()

    def get_played_cards(self) -> list[PlayedCard]:
        return self.played_cards
