import asyncio
import logging


from controller.ai_controller_ppo import AiControllerPpo
from drl_training_base import start_training



if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)
    asyncio.run(
        start_training(
            get_ai_controller=AiControllerPpo,
            net_combinations=[[512, 256, 256, 256]],
            prefix="ppo",
            num_epochs=50_000,
            epochs_until_applying_training=250,
            epochs_until_arena_evaluation=500,
        )
    )
