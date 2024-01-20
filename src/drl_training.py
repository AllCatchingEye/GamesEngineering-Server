import asyncio
import logging


from ai.nn_layout_helper import (
    compute_layers_combinations,
)
from controller.ai_controller import AiController
from drl_training_base import ALL_GAME_TYPES, start_training
from state.gametypes import Gametype


# Modify this array to train only certain game types.
GAME_TYPES_TO_TRAIN: list[Gametype] = ALL_GAME_TYPES
NUM_EPOCHS = 15000
EPOCHS_UNTIL_APPLYING_TRAINING = 250
DISPLAY_LOGS = False
DISPLAY_PROGRESS = True
UPDATE_PROGRESS_INTERVAL_MS = 0.1
EPOCHS_UNTIL_ARENA_EVALUATION = 500


# NET_COMBINATIONS = compute_layers(neurons=256, layers=5)
NET_COMBINATIONS = compute_layers_combinations(
    neurons_range=(64, 1024),
    neurons_increment=lambda x: x * 2,
    layers_range=(2, 16),
    layers_increment=lambda x: x * x,
    auto_encoder_threshold=8,
    auto_encoder_neurons_range=(10, 40),
    auto_encoder_neurons_increment=lambda x: x + 30,
)



if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)
    asyncio.run(start_training(
        get_ai_controller=AiController,
        net_combinations=list(NET_COMBINATIONS),
        game_types_to_train=[Gametype.SAUSPIEL],
        num_epochs=50_000,
        epochs_until_applying_training=250,
        epochs_until_arena_evaluation=1000,
    ))
