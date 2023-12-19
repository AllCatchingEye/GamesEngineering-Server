from controller.player_controller import PlayerController
from state.card import Card
from state.event import Event, GameStartUpdate
from state.gametypes import GameGroup, Gametype
from state.stack import Stack
from state.suits import Suit


class CombiController(PlayerController):
    def __init__(self, wants_to_play_contoller: PlayerController, play_card_contoller: PlayerController):
        super().__init__()
        self.wants_to_play_contoller = wants_to_play_contoller
        self.play_card_contoller = play_card_contoller

    async def wants_to_play(self, current_lowest_gamegroup: GameGroup) -> bool:
        return await self.wants_to_play_contoller.wants_to_play(current_lowest_gamegroup)

    async def select_gametype(
            self, choosable_gametypes: list[tuple[Gametype, Suit | None]]
    ) -> tuple[Gametype, Suit | None]:
        return await self.wants_to_play_contoller.select_gametype(choosable_gametypes)

    async def play_card(self, stack: Stack, playable_cards: list[Card]) -> Card:
        return await self.play_card_contoller.play_card(stack, playable_cards)

    async def on_game_event(self, event: Event) -> None:
        if isinstance(event, GameStartUpdate):
            await self.wants_to_play_contoller.on_game_event(
                GameStartUpdate(event.player, event.hand.copy(), event.play_order))
            await self.play_card_contoller.on_game_event(
                GameStartUpdate(event.player, event.hand.copy(), event.play_order))
        else:
            await self.wants_to_play_contoller.on_game_event(event)
            await self.play_card_contoller.on_game_event(event)

    async def choose_game_group(self, available_groups: list[GameGroup]) -> GameGroup:
        return await self.wants_to_play_contoller.choose_game_group(available_groups)
