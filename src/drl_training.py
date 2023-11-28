import asyncio
import random

from controller.ai_controller import AiController
from logic.game import Game

NUM_EPOCHS = 10_000
EPOCHS_UNTIL_APPLYING_TRAINING = 100


def get_new_game():
    new_game = Game(random.Random())
    new_trained_ai_controller = AiController(True)
    new_game.controllers = [
        new_trained_ai_controller,
        AiController(False),
        AiController(False),
        AiController(False),
    ]
    return (new_game, new_trained_ai_controller)


for epoch in range(NUM_EPOCHS):
    game, trained_ai_controller = get_new_game()
    asyncio.run(game.run())

    if epoch % EPOCHS_UNTIL_APPLYING_TRAINING == 0:
        print("%i%%" % (epoch / NUM_EPOCHS * 100))
        trained_ai_controller.persist_training_results()
        updated_game, updated_trained_ai_controller = get_new_game()
        game = updated_game
        trained_ai_controller = updated_trained_ai_controller
