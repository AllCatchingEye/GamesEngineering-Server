# from typing import Dict
import random

from state.card import Card
from state.deck import Deck
from state.hand import Hand
from state.player import Player

# from suits import Suit
from state.ranks import Rank
from state.stack import Stack
from state.suits import Suit

HAND_SIZE = 8
ROUNDS = 8
PLAYER_COUNT = 4


class Game:
    def __init__(self) -> None:
        self.players = self.__create_players()
        self.deck: Deck = Deck()
        self.played_cards: list[Card] = []

    def __create_players(self) -> list[Player]:
        """Create a list of players for the game."""
        players: list[Player] = []
        for i in range(PLAYER_COUNT):
            players.append(Player(i))
        return players

    def run(self) -> None:
        """Start the game."""

        while True:
            suit_chosen: Suit = self.determine_gametype()
            self.__new_game(suit_chosen)

    def determine_gametype(self) -> Suit:
        """Determine the game type based on player choices."""
        self.__distribute_cards()
        suit_chosen = self.__call_game()
        if suit_chosen is None:
            print("=============================================================")
            return self.determine_gametype()
        return suit_chosen

    def __distribute_cards(self) -> None:
        """Distribute cards to players."""
        deck: list[Card] = self.deck.get_full_deck()
        random.shuffle(deck)

        for player in self.players:
            deck = self.__distribute_hand(player, deck)

    def __distribute_hand(self, player: Player, deck: list[Card]) -> list[Card]:
        """Distribute cards for a player's hand."""
        hand: Hand = Hand(deck[:HAND_SIZE])
        player.set_hand(hand)

        deck = deck[HAND_SIZE:]
        return deck

    def __call_game(self) -> Suit | None:
        """Call the game type based on player choices."""
        chosen_suit = None
        game_called = False
        for player in self.players:
            game_called = player.wants_to_play()
            if game_called and chosen_suit is None:
                chosen_suit = player.decide_suit_for_game()

        return chosen_suit

    def __new_game(self, suit_chosen: Suit) -> None:
        """Start a new game with the specified suit as the game type."""
        trump_cards = self.__get_trump_cards()
        for _ in range(ROUNDS):
            print(f"The suit for this game is: {suit_chosen.name}")
            self.start_round(suit_chosen, trump_cards)

        game_winner = self.__get_game_winner()
        print(f"The winner of this game is player {game_winner.get_id()}!")
        print(f"He won {game_winner.get_points()} points!")

    def __get_trump_cards(self) -> list[Card]:
        """Get all trump cards for the game."""
        # For basic implementation return always Farbsolo prio
        trump_ober = self.deck.get_cards_by_rank(Rank.OBER)
        trump_unter = self.deck.get_cards_by_rank(Rank.UNTER)

        trump_cards: list[Card] = trump_ober + trump_unter
        return trump_cards

    def start_round(self, suit_chosen: Suit, trump_cards: list[Card]) -> None:
        """Start a new round."""
        stack = self.__play_cards(suit_chosen, trump_cards)
        self.__finish_round(stack)

    def __play_cards(self, suit: Suit, trump_cards: list[Card]) -> Stack:
        """Play cards in the current round."""
        print("=============================================================")
        stack = Stack(suit)
        for player in self.players:
            print("=============================================================")
            card: Card = player.lay_card(suit, trump_cards)
            stack.add_card(card, player)
        return stack

    def __finish_round(self, stack: Stack) -> None:
        """Finish the current round and determine the winner."""
        winner = stack.get_winner()
        stack_value = stack.get_value()
        winner.add_points(stack_value)
        self.__show_winner(winner)
        self.__change_player_order(winner)

    def __show_winner(self, winner: Player) -> None:
        """Display the winner of the round."""
        print(f"Player {winner.get_id()}, you are the winner of this round!")
        print(f"You got {winner.get_points()} points.")

    def __change_player_order(self, winner: Player) -> None:
        """Change the order of players based on the round winner."""
        winner_index = self.__get_winner_index(winner)
        self.__swap_players(winner_index)

    def __get_game_winner(self) -> Player:
        """Determine the winner of the entire game."""
        game_winner = self.players[0]
        for player in self.players[0:]:
            if player.get_points() > game_winner.get_points():
                game_winner = player

        return game_winner

    def __get_winner_index(self, winner: Player) -> int:
        """Find the index of the player who won"""
        winner_index = 0
        for index, player in enumerate(self.players):
            if player.get_id() == winner.get_id():
                winner_index = index
        return winner_index

    def __swap_players(self, winner_index: int) -> None:
        first: list[Player] = self.players[winner_index:]
        last: list[Player] = self.players[:winner_index]
        self.players = first + last
