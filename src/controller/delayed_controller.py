import asyncio
import random
from controller.player_controller import PlayerController
from state.card import Card
from state.event import Event
from state.gametypes import GameGroup, Gametype
from state.stack import Stack
from state.suits import Suit


class DelayedController(PlayerController):
    def __init__(
        self,
        actual_controller: PlayerController,
        delay_ms: tuple[int, int] = (500, 3000),
    ):
        self.actual_controller = actual_controller
        self.delay_ms = delay_ms

    async def wants_to_play(self, current_lowest_gamegroup: GameGroup) -> bool:
        wants_to_play = await self.actual_controller.wants_to_play(
            current_lowest_gamegroup
        )
        await asyncio.sleep(random.randint(*self.delay_ms) / 1000)
        return wants_to_play

    async def select_gametype(
        self, choosable_gametypes: list[tuple[Gametype, Suit | None]]
    ) -> tuple[Gametype, Suit | None]:
        result = await self.actual_controller.select_gametype(choosable_gametypes)
        await asyncio.sleep(random.randint(*self.delay_ms) / 1000)
        return result

    async def play_card(self, stack: Stack, playable_cards: list[Card]) -> Card:
        card = await self.actual_controller.play_card(stack, playable_cards)
        await asyncio.sleep(random.randint(*self.delay_ms) / 1000)
        return card

    async def on_game_event(self, event: Event) -> None:
        await self.actual_controller.on_game_event(event)

    async def choose_game_group(self, available_groups: list[GameGroup]) -> GameGroup:
        group = await self.actual_controller.choose_game_group(available_groups)
        await asyncio.sleep(random.randint(*self.delay_ms) / 1000)
        return group
