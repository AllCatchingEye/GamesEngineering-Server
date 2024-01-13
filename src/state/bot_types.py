from enum import Enum
from controller.handcrafted_controller import HandcraftedController
from controller.passive_controller import PassiveController

from controller.player_controller import PlayerController
from controller.random_controller import RandomController


class BotType(Enum):
    """Enumeration of bot types."""

    RANDOM = 0
    PASSIVE = 1
    HANDCRAFTED = 2


def bot_type_to_controller(bot_type: BotType) -> PlayerController:
    """Converts a bot type to a PlayerController."""
    match bot_type:
        case BotType.RANDOM:
            return RandomController()
        case BotType.PASSIVE:
            return PassiveController()
        case BotType.HANDCRAFTED:
            return HandcraftedController()
