from controller.player_controller import PlayerController
from state.card import Card
from state.event import Event
from state.gametypes import GameGroup, Gametype
from state.stack import Stack
from state.suits import Suit


class TerminalController(PlayerController):
    async def wants_to_play(self, current_lowest_gamegroup: GameGroup) -> bool:
        print(f"You have to play atleast {current_lowest_gamegroup}")
        decision = input("Do you want to play? (y/n) ")
        return decision == "y"

    async def select_gametype(
        self, choosable_gametypes: list[tuple[Gametype, Suit | None]]
    ) -> tuple[Gametype, Suit | None]:
        print("Choose a gamemode:")
        for index, gametype in enumerate(choosable_gametypes):
            print(f"{index}: {gametype}")
        gametype_index = int(input())
        return choosable_gametypes[gametype_index]

    async def play_card(self, stack: Stack, playable_cards: list[Card]) -> Card:
        print("The stack is:")
        print(stack)
        print("Choose a card to play:")
        for index, card in enumerate(playable_cards):
            print(f"{index}: {card}")
        card_index = int(input())
        return playable_cards[card_index]

    async def on_game_event(self, event: Event) -> None:
        print(event)

    async def choose_game_group(self, available_groups: list[GameGroup]) -> GameGroup:
        print("Choose a gamegroup:")
        for index, gamegroup in enumerate(available_groups):
            print(f"{index}: {gamegroup}")
        gamegroup_index = int(input())
        return available_groups[gamegroup_index]
