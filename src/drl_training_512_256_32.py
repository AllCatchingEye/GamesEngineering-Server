import asyncio
import logging


from controller.ai_controller import AiController
from drl_training_base import start_training
from state.gametypes import Gametype



if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)
    asyncio.run(
        start_training(
            get_ai_controller=AiController,
            net_combinations=[[512, 512, 32, 256]],
            prefix="iter_02",
            num_epochs=50_000,
            epochs_until_applying_training=250,
            epochs_until_arena_evaluation=500,
            game_types_to_train=[Gametype.SAUSPIEL, Gametype.SOLO]
        )
    )
