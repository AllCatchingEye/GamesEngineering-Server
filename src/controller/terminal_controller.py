from controller.player_controller import PlayerController
from state.card import Card
from state.event import Event
from state.gametypes import Gametype
from state.stack import Stack
from state.suits import Suit


class TerminalController(PlayerController):
    def wants_to_play(self, decisions: list[bool | None]) -> bool:
        print("Your hand:")
        print(self.player.hand)
        print("Decisions before you:")
        print(decisions)
        decision = input("Do you want to play? (y/n) ")
        return decision == "y"

    def select_gametype(self, choosable_gametypes: list[(Gametype, Suit | None)]) -> (Gametype, Suit | None):
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
