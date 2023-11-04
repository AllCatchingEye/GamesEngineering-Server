import random

from controller.player_controller import PlayerController
from logic.gamemodes import gamemode_wenz
from logic.gamemodes.gamemode import GameMode
from logic.gamemodes.gamemode_geier import GameModeGeier
from logic.gamemodes.gamemode_ramsch import GameModeRamsch
from logic.gamemodes.gamemode_sauspiel import GameModeSauspiel
from logic.gamemodes.gamemode_solo import GameModeSolo
from logic.gamemodes.gamemode_wenz import GameModeWenz
from state.card import Card
from state.deck import Deck
from state.event import (
    CardPlayedEvent,
    Event,
    GameEndEvent,
    GameStartEvent,
    GametypeDeterminedEvent,
    GametypeWishedEvent,
    PlayDecisionEvent,
    RoundResultEvent,
)
from state.gametypes import Gametype
from state.hand import Hand
from state.player import Player
from state.stack import Stack
from state.suits import Suit

HAND_SIZE = 8
ROUNDS = 8
PLAYER_COUNT = 4


class Game:
    controllers: list[PlayerController]
    gamemode: GameMode

    def __init__(self) -> None:
        self.players = self.__create_players()
        self.deck: Deck = Deck()
        self.played_cards: list[Card] = []
        self.controllers = []

    def __create_players(self) -> list[Player]:
        """Create a list of players for the game."""
        players: list[Player] = []
        for i in range(PLAYER_COUNT):
            players.append(Player(i))
        return players

    def run(self) -> None:
        """Start the game."""

        while True:
            self.determine_gametype()
            self.__new_game()

    def determine_gametype(self) -> Gametype:
        """Determine the game type based on player choices."""
        self.__distribute_cards()
        suit_chosen = self.__call_game()
        if suit_chosen is None:
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
        player.hand = hand

        deck = deck[HAND_SIZE:]

        self.controllers[player.id].on_game_event(GameStartEvent(hand))
        return deck

    def __call_game(self) -> Gametype | None:
        """Call the game type based on player choices."""
        decisions: list[bool | None] = [None, None, None, None]
        for player in self.players:
            i = player.id
            wants_to_play = self.controllers[i].wants_to_play(decisions)
            self.__broadcast(PlayDecisionEvent(player, wants_to_play))
            decisions[i] = wants_to_play

        chosen_types: list[Gametype | None] = [None, None, None, None]
        for i, wants_to_play in enumerate(decisions):
            if wants_to_play is True:
                game_type = self.controllers[i].select_gametype([Gametype.SOLO])
                chosen_types[i] = game_type
                self.__broadcast(GametypeWishedEvent(self.players[i], game_type))

        # TODO: Fix gametype determination
        for i, game_type in enumerate(chosen_types):
            if game_type is not None:
                self.__broadcast(GametypeDeterminedEvent(self.players[i], game_type))
                match (game_type):
                    case Gametype.SOLO:
                        self.gamemode = GameModeSolo(Suit.EICHEL)  # TODO: Fix suit
                    case Gametype.WENZ:
                        self.gamemode = GameModeWenz(None)
                    case Gametype.GEIER:
                        self.gamemode = GameModeGeier(None)
                    case Gametype.FARBWENZ:
                        self.gamemode = GameModeWenz(Suit.EICHEL)
                    case Gametype.FARBGEIER:
                        self.gamemode = GameModeGeier(Suit.EICHEL)
                    case Gametype.SAUSPIEL:
                        self.gamemode = GameModeSauspiel(Suit.EICHEL)
                return game_type

        self.gamemode = GameModeRamsch()

        return Gametype.RAMSCH

    def __new_game(self) -> None:
        """Start a new game with the specified suit as the game type."""
        trump_cards = self.gamemode.get_trump_cards()
        for _ in range(ROUNDS):
            self.start_round(trump_cards)

        self.__get_game_winner()

    def start_round(self, trump_cards: list[Card]) -> None:
        """Start a new round."""
        stack = self.__play_cards(trump_cards)
        self.__finish_round(stack)

    def __play_cards(self, trump_cards: list[Card]) -> Stack:
        """Play cards in the current round."""
        stack = Stack()
        for player in self.players:
            # TODO: Actually determine the playable cards for the player
            card: Card = self.controllers[player.id].play_card(stack, trump_cards)
            stack.add_card(card, player)
            self.__broadcast(CardPlayedEvent(player, card, stack))
        return stack

    def __finish_round(self, stack: Stack) -> None:
        """Finish the current round and determine the winner."""
        winner = self.gamemode.determine_stitch_winner(stack)
        stack_value = stack.get_value()
        winner.points += stack_value
        self.__broadcast(RoundResultEvent(winner, stack_value, stack))
        self.__change_player_order(winner)

    def __change_player_order(self, winner: Player) -> None:
        """Change the order of players based on the round winner."""
        winner_index = self.__get_winner_index(winner)
        self.__swap_players(winner_index)

    def __get_game_winner(self) -> Player:
        """Determine the winner of the entire game."""
        game_winner = self.players[0]
        for player in self.players[0:]:
            if player.points > game_winner.points:
                game_winner = player

        self.__broadcast(GameEndEvent(game_winner, game_winner.points))
        return game_winner

    def __get_winner_index(self, winner: Player) -> int:
        """Find the index of the player who won"""
        winner_index = 0
        for index, player in enumerate(self.players):
            if player.id == winner.id:
                winner_index = index
        return winner_index

    def __swap_players(self, winner_index: int) -> None:
        first: list[Player] = self.players[winner_index:]
        last: list[Player] = self.players[:winner_index]
        self.players = first + last

    def __broadcast(self, event: Event) -> None:
        """Broadcast an event to all players."""
        for controller in self.controllers:
            controller.on_game_event(event)
