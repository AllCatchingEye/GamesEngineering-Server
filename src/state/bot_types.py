from enum import Enum

import controller.player_controller as _
class BotType(Enum):
    """Enumeration of bot types."""

    HANDCRAFTED = 0
    RANDOM = 1
    PASSIVE = 2
    AI_ITER_01 = 3
    AI_ITER_02 = 4
    AI_ITER_03 = 5
    AI_ITER_04 = 6


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
        case BotType.AI_ITER_01:
            from controller.ai_controller_iter1 import AiController_1It
            return AiController_1It()
        case BotType.AI_ITER_02:
            from controller.ai_controller_iter2 import AiController_2It
            return AiController_2It()
        case BotType.AI_ITER_03:
            from controller.ai_controller_iter3 import AiController_3It
            return AiController_3It()
        case BotType.AI_ITER_04:
            from controller.ai_controller_iter4 import AiController_4It
            return AiController_4It()

            
