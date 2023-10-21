from state.cards import Card
from state.player import Player

class PlayedCard:
    def __init__(self, card: Card, player: Player) -> None:
        self.card: Card = card
        self.player: Player = player

    def get_value(self):
        return self.card.get_value()

    def get_card(self) -> Card:
        return self.card

    def get_player(self) -> Player:
        return self.player

class Stack:
    def __init__(self):
        self.played_cards: list[PlayedCard]= []
        self.value = 0

    def add_card(self, card: Card, player: Player):
        played_card = PlayedCard(card, player)
        self.played_cards.append(played_card)
        self.value += card.rank.value

    def get_value(self) -> int:
        return self.value

    def get_top(self) -> Card:
        top_card: PlayedCard = self.played_cards[-1]
        return top_card.get_card()

    def get_winner(self) -> Player:
        best_card: PlayedCard = self.played_cards[0]
        for next_card in self.played_cards[1:]:
            self.__compare_cards(best_card, next_card)

        return best_card.get_player()

    def __compare_cards(self, current_card: PlayedCard, next_card: PlayedCard) -> PlayedCard:
        if current_card.get_value() > next_card.get_value():
            return current_card
        else:
            return next_card

    def get_played_cards(self) -> list[PlayedCard]:
        return self.played_cards
