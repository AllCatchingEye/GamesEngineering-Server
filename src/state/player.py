from state.card import Card
from state.hand import Hand
from state.suits import Suit, get_all_suits


class Player:
    def __init__(self, player_id: int) -> None:
        self.id = player_id
        self.points = 0
        self.hand: Hand
        self.played_cards: list[Card] = []

    def get_id(self) -> int:
        return self.id

    def add_points(self, points: int) -> None:
        self.points += points

    def get_points(self) -> int:
        return self.points

    def set_hand(self, hand: Hand) -> None:
        self.hand = hand

    def show_hand(self) -> None:
        """Display the player's hand."""
        print(f"Player {self.id}, your cards are:")
        self.hand.show_all_cards()

    def lay_card(self, suit: Suit, trump_cards: list[Card]) -> Card:
        """Player lays a card on the table."""
        layable_cards: list[tuple[int, Card]] = self.__get_layable_cards(
            suit, trump_cards
        )
        self.show_hand()
        self.__show_layable_cards(layable_cards)
        chosen_card = self.__ask_for_card()
        self.hand.remove_card(chosen_card)
        self.played_cards.append(chosen_card)
        return chosen_card

    def __get_layable_cards(
        self, suit: Suit, trump_cards: list[Card]
    ) -> list[tuple[int, Card]]:
        """Get all layable cards for the given suit or trumps."""
        available_suits: list[tuple[int, Card]] = self.hand.get_all_cards_for_suit(suit)
        if len(available_suits) != 0:
            return available_suits

        available_trumps: list[tuple[int, Card]] = self.hand.get_all_trumps_in_deck(
            trump_cards
        )
        if len(available_trumps) != 0:
            return available_trumps

        return []

    def __show_layable_cards(self, layable_cards: list[tuple[int, Card]]) -> None:
        """Display the cards that can be layed."""
        if len(layable_cards) == 0:
            self.show_hand()
        else:
            print(f"Player {self.id}, you can lay the following cards: ")
            for index, card in layable_cards:
                print(f"{index}, {card}")

    def __ask_for_card(self) -> Card:
        """Ask the player to choose a card."""
        print("Enter the index of the card you want to play: ")
        index = int(input())
        return self.hand.get_card(index)

    def wants_to_play(self) -> bool:
        """Ask if the player wants to play."""
        self.show_hand()
        print(f"Player {self.id}, do you want to play?")
        print("Enter 'y' for yes, any other key for no.")
        return input().lower() == "y"

    def decide_suit_for_game(self) -> Suit:
        """Ask the player to choose a suit."""
        print(f"Player {self.id}, which suit do you want to set?")
        self.__print_suits()
        print("Enter the number of the suit you want to set: ")
        number = int(input())
        if 0 <= number <= 3:
            return get_all_suits()[number]
        return self.decide_suit_for_game()

    def __print_suits(self) -> None:
        """Print all available suits."""
        all_suits = get_all_suits()
        for index, suit in enumerate(all_suits):
            print(f"{index}: {suit.name.capitalize()}")
