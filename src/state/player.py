from cards import Card
from hand import Hand

#TODO: Refactor this spaghetti
class Player:
    def __init__(self, id: int):
        self.id = id
        self.points = 0
        self.hand: Hand
        self.played_cards: list[Card] = []

    def get_id(self) -> int:
        return self.id

    def add_points(self, points: int):
        self.points += points

    def get_points(self) -> int:
        return self.points

    def set_hand(self, hand: Hand):
        self.hand: Hand = hand

    def show_hand(self):
        print(f'Player {self.id}, your cards are:')
        self.hand.show()

    def lay_card(self, trump_cards: set[Card]) -> Card:
        layable_cards: set[Card] = self.__get_layable_cards(trump_cards)
        self.show_hand()
        self.__show_layable_cards(layable_cards)
        chosen_card = self.__ask_for_card()
        self.hand.remove_card(chosen_card)
        self.played_cards.append(chosen_card)
        return chosen_card

    def __get_layable_cards(self, trump_cards: set[Card]) -> set[Card]:
        return set(self.hand.get_cards()).intersection(trump_cards)

    def __show_layable_cards(self, layable_cards: set[Card]):
        if len(layable_cards) == 0:
            self.show_hand()
        else:
            print(f"Player {self.id}, you can lay the following cards: ")
            for card in layable_cards:
                print(card)

    def __ask_for_card(self) -> Card:
        print("Enter the index of the card you want to play: ")
        index = int(input())
        chosen_card: Card = self.hand.get_card(index)
        return chosen_card

    def decide_first_card(self) -> Card:
        print(f"Player {self.id}, you are first. Place the first card.")
        self.show_hand()
        first_card = self.__ask_for_card()
        self.hand.remove_card(first_card)
        self.played_cards.append(first_card)
        return first_card

    def plays(self) -> bool:
        print(f"Player {self.id}, do you want to play?")
        print("Enter 'y' for yes, any other key for no.")
        answer = input()
        if answer == 'y':
            return True
        else:
            return False
