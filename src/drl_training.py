import asyncio
import datetime
import logging
import random
import sys
from time import time

from matplotlib import pyplot as plt

from ai.select_card.rl_agent_trainer import RLBaseAgentTrainer
from controller.ai_controller import AiController
from controller.random_controller import RandomController
from logic.game import Game

NUM_EPOCHS = 10_000_000
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
        # AiController(False),
        RandomController(),
    ]
    return (new_game, new_trained_ai_controller)


def evaluate(rewards: list[float]):
    fig = plt.figure(figsize=(10, 10))

    ax = fig.add_subplot(111)
    ax.plot(rewards)
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    plt.show()


async def main():
    if not DISPLAY_PROGRESS:
        logging.basicConfig(level=logging.DEBUG)

    print("> Training started.")

    mean_rewards: list[float] = []

    start = time()
    last_update = start
    for epoch in range(NUM_EPOCHS):
        game, trained_ai_controller = get_new_game()
        await game.run()
        if isinstance(trained_ai_controller.play_game_agent, RLBaseAgentTrainer):
            rewards = trained_ai_controller.play_game_agent.rewards
            mean_rewards.append(sum(rewards) / len(rewards))
        trained_ai_controller.play_game_agent.reset()

        end = time()
        if DISPLAY_PROGRESS and end - last_update > UPDATE_PROGRESS_INTERVAL_MS:
            last_update = end
            duration = end - start
            remaining_time = duration / (epoch + 1) * NUM_EPOCHS
            sys.stdout.write(
                "> Training ongoing: %i%% (%s of ~%s)%s\r"
                % (
                    (epoch + 1) / NUM_EPOCHS * 100,
                    str(datetime.timedelta(seconds=duration))[:-7],
                    str(datetime.timedelta(seconds=remaining_time))[:-7],
                    " " * 20,
                )
            )
            sys.stdout.flush()

        if epoch % EPOCHS_UNTIL_APPLYING_TRAINING == 0:
            trained_ai_controller.persist_training_results()
            updated_game, updated_trained_ai_controller = get_new_game()
            game = updated_game
            trained_ai_controller = updated_trained_ai_controller

    print("\n> Training complete.")

    evaluate(mean_rewards)


if __name__ == "__main__":
    asyncio.run(main())
