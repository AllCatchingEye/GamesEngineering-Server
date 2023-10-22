from cards import Card
from player import Player
from suits import Suit
from ranks import Rank
from ranks import get_value_of

class PlayedCard:
    def __init__(self, card: Card, player: Player) -> None:
        self.card: Card = card
        self.player: Player = player

    def get_value(self):
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
    def __init__(self, suit: Suit):
        self.suit = suit
        self.played_cards: list[PlayedCard] = []
        self.strongest_card: PlayedCard 
        self.value = 0

    def add_card(self, card: Card, player: Player):
        played_card = PlayedCard(card, player)
        if self.is_empty():
            self.strongest_card = played_card
        else:
            self.__determine_stitch(played_card)

        self.value += get_value_of(card.get_rank())
        self.played_cards.append(played_card)

    def is_empty(self) -> bool:
        return len(self.played_cards) == 0

    def __determine_stitch(self, card: PlayedCard):
        if self.__same_suit(card):
            # Neue Karte hat selbe farbe aber höheren rang
            if self.strongest_card.get_rank() < card.get_rank():
                self.strongest_card = card
        # Neue Karte ist trumpf und momentane karte ist kein trumpf
        elif self.__is_trump(card) and not self.__is_trump(self.strongest_card):
            self.strongest_card = card
        elif self.__is_trump(card) and self.__is_trump(self.strongest_card):
            # Neue Karte ist trumpf und momentane karte ist trumpf
            if self.strongest_card.get_suit() < card.get_suit():
                self.strongest_card = card
            # Trumpf wird geschlagen von stärkerer karte gleicher Farbe
            if self.strongest_card.get_rank() < card.get_rank():
                self.strongest_card = card
            # Different color but no trump
        elif self.strongest_card.get_suit() < card.get_suit():
            self.strongest_card = card

        print("The strongest card currently is:")
        print(f"{self.strongest_card.get_card()}, by player {self.strongest_card.get_player().get_id()}")

    def __is_trump(self, card: PlayedCard) -> bool:
        return card.get_rank() == Rank.Ober or card.get_rank() == Rank.Unter

    def __same_suit(self,  card: PlayedCard) -> bool:
        return self.strongest_card.get_suit() == card.get_suit()

    # def __same_rank(self,  card: PlayedCard) -> bool:
    #     return self.strongest_card.get_rank() == card.get_rank()

    def get_value(self) -> int:
        return self.value

    def get_suit(self) -> Suit:
        return self.suit

    def get_top(self) -> Card:
        top_card: PlayedCard = self.played_cards[-1]
        return top_card.get_card()

    def get_winner(self) -> Player:
        return self.strongest_card.get_player()

    def get_played_cards(self) -> list[PlayedCard]:
        return self.played_cards
