from cards import Card
from hand import Hand
from suits import get_all_suits
from suits import Suit

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

    def lay_card(self, suit: Suit, trump_cards: list[Card]) -> Card:
        layable_cards: list[tuple[int, Card]] = self.__get_layable_cards(suit, trump_cards)
        self.show_hand()
        self.__show_layable_cards(layable_cards)
        chosen_card = self.__ask_for_card()
        self.hand.remove_card(chosen_card)
        self.played_cards.append(chosen_card)
        return chosen_card

    def __get_layable_cards(self, suit:Suit, trump_cards: list[Card]) -> list[tuple[int, Card]]:
        available_suits: list[tuple[int, Card]] = self.hand.get_suits(suit)
        if len(available_suits) != 0:
            return available_suits

        available_trumps: list[tuple[int, Card]] = self.hand.get_trumps(trump_cards)
        if len(available_trumps) != 0:
            return available_trumps

        return []
        

    def __show_layable_cards(self, layable_cards: list[tuple[int, Card]]):
        if len(layable_cards) == 0:
            self.show_hand()
        else:
            print(f"Player {self.id}, you can lay the following cards: ")
            for card in layable_cards:
                index = card[0]
                card = card[1]
                print(f"{index}, {card}")

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
        self.show_hand()
        print(f"Player {self.id}, do you want to play?")
        print("Enter 'y' for yes, any other key for no.")
        answer = input()
        if answer == 'y':
            return True
        else:
            return False

    def get_suit_for_game(self) -> Suit:
        print(f"Player {self.id}, which suit do you want to set?")
        self.__print_suits()
        return self.__ask_for_suit()

    def __ask_for_suit(self) -> Suit:
        print("Enter the number of the suit you want to set: ")
        number = int(input())
        if number >= 0 and number <= 3:
            return get_all_suits()[number]
        else:
            self.__ask_for_suit()

    def __print_suits(self):
        all_suits: list[Suit] = get_all_suits()
        for suit in all_suits:
            print(f"{suit.value}: {suit.name}")
