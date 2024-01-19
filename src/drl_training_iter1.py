import asyncio
import logging


from controller.ai_controller_iter1 import AiController_1It
from drl_training_base import start_training
from state.gametypes import Gametype



if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)
    asyncio.run(
        start_training(
            get_ai_controller=lambda net_layers, train, lr, prefix: AiController_1It(train),
            prefix="iter_01",
            net_combinations=[[256, 256, 256, 256]],
            num_epochs=50_000,
            epochs_until_applying_training=250,
            epochs_until_arena_evaluation=500,
            game_types_to_train=[Gametype.SAUSPIEL, Gametype.SOLO]
        )
    )
