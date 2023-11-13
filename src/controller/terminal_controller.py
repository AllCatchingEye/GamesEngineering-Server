from controller.player_controller import PlayerController
from state.card import Card
from state.event import Event
from state.gametypes import Gametype, GameGroup
from state.stack import Stack
from state.suits import Suit
from state.player import Player


class TerminalController(PlayerController):
    def wants_to_play(self, current_player: Player | None, current_lowest_gamegroup: GameGroup | None) -> bool:
        print("Your hand:")
        print(self.player.hand)
        if ( current_player is not None):
            print(f'Player %d is atleast playing %s', current_player.id, current_lowest_gamegroup)
        decision = input("Do you want to play? (y/n) ")
        return decision == "y"

    def select_gametype(
            self, choosable_gametypes: list[tuple[Gametype, Suit | None]]
    ) -> tuple[Gametype, Suit | None]:
        print("Choose a gamemode:")
        for index, gametype in enumerate(choosable_gametypes):
            print(f"{index}: {gametype}")
        gametype_index = int(input())
        return choosable_gametypes[gametype_index]

    def play_card(self, stack: Stack, playable_cards: list[Card]) -> Card:
        print("The stack is:")
        print(stack)
        print("Choose a card to play:")
        for index, card in enumerate(playable_cards):
            print(f"{index}: {card}")
        card_index = int(input())
        return playable_cards[card_index]

    def on_game_event(self, event: Event) -> None:
        print(event)

    def chooseGameGroup(self, available_groups: list[GameGroup]) -> GameGroup:
        print("Choose a gamegroup:")
        for index, gamegroup in enumerate(available_groups):
            print(f"{index}: {gamegroup}")
        gamegroup_index = int(input())
        return available_groups[gamegroup_index]
