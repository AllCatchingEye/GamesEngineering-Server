from enum import Enum

import controller.player_controller as _
class BotType(Enum):
    """Enumeration of bot types."""

    RANDOM = 0
    PASSIVE = 1
    HANDCRAFTED = 2


def bot_type_to_controller(bot_type: BotType) -> "_.PlayerController":
    """Converts a bot type to a PlayerController."""
    match bot_type:
        case BotType.RANDOM:
            from controller.random_controller import RandomController
            return RandomController()
        case BotType.PASSIVE:
            from controller.passive_controller import PassiveController
            return PassiveController()
        case BotType.HANDCRAFTED:
            from controller.handcrafted_controller import HandcraftedController
            return HandcraftedController()
