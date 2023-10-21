from state.cards import Card
from state.stack import Stack

class Player:
    def __init__(self, id: int):
        self.id = id
        self.points = 0
        self.hand = []
        self.played_cards: list[Card] = []

    def set_hand(self, hand: list[Card]):
        self.hand = hand

    def get_card(self, index: int) -> Card:
        return self.hand[index]

    def show_cards(self):
        print(f'Player {self.id}, your cards are:')
        for card in self.hand:
            print(card)

    def add_points(self, points: int):
        self.points += points

    def has_empty_hand(self):
        cards_left = len(self.hand)
        if cards_left > 0:
            return True
        else:
            return False

    def lay_card(self, stack: Stack) -> Card:
        layable_cards = self.__get_layable_cards(stack)
        self.__show_layable_cards(layable_cards)
        chosen_card = self.__ask_for_card()
        self.played_cards.append(chosen_card)
        return chosen_card

    #TODO: Implement method
    def __get_layable_cards(self, stack: Stack) -> list[Card]:
        pass

    def __show_layable_cards(self, layable_cards: list[Card]):
        print(f"Player {self.id}, you can lay the following cards: ")
        for card in layable_cards:
            print(card)

    def __ask_for_card(self) -> Card:
        print("Enter the index of the card you want to play: ")
        index = int(input())
        chosen_card: Card = self.get_card(index)
        return chosen_card

    def plays(self) -> bool:
        print("Do you want to play?")
        answer = input()
        if answer == 'y':
            return True
        else:
            return False
