import asyncio
import logging
import random
import sys
from time import gmtime, strftime, time

from controller.ai_controller import AiController
from logic.game import Game

NUM_EPOCHS = 10_000
EPOCHS_UNTIL_APPLYING_TRAINING = 100
DISPLAY_LOGS = False
DISPLAY_PROGRESS = True
UPDATE_PROGRESS_INTERVAL_MS = 0.1


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


def main():
    if not DISPLAY_PROGRESS:
        logging.basicConfig(level=logging.DEBUG)

    print("Training started.")

    start = time()
    last_update = start
    for epoch in range(NUM_EPOCHS):
        game, trained_ai_controller = get_new_game()
        asyncio.run(game.run())

        end = time()
        if DISPLAY_PROGRESS and end - last_update > UPDATE_PROGRESS_INTERVAL_MS:
            last_update = end
            duration = end - start
            remaining_time = duration / (epoch + 1) * NUM_EPOCHS
            sys.stdout.write(
                "\rTraining ongoing: %i%% (%s of ~%s)"
                % (
                    (epoch + 1) / NUM_EPOCHS * 100,
                    strftime("%M:%S", gmtime(duration)),
                    strftime("%M:%S", gmtime(remaining_time)),
                )
            )
            sys.stdout.flush()

        if epoch % EPOCHS_UNTIL_APPLYING_TRAINING == 0:
            trained_ai_controller.persist_training_results()
            updated_game, updated_trained_ai_controller = get_new_game()
            game = updated_game
            trained_ai_controller = updated_trained_ai_controller

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
