import asyncio
import logging
import random

from matplotlib import pyplot as plt
from tqdm import tqdm

from ai.dealer.dealer_factory import DealerFactory
from ai.select_card.rl_agent_trainer import RLBaseAgentTrainer
from controller.ai_controller import AiController
from logic.card_game import CardGame
from state.gametypes import Gametype
from state.suits import Suit

TRAIN_GAME_TYPE = Gametype.SAUSPIEL
TRAIN_WITH_TRUMP_SUITS = [Suit.EICHEL, Suit.GRAS, Suit.SCHELLEN]

NUM_EPOCHS = 10_000_000
EPOCHS_UNTIL_APPLYING_TRAINING = 100
DISPLAY_LOGS = False
DISPLAY_PROGRESS = True
UPDATE_PROGRESS_INTERVAL_MS = 0.1


async def get_new_game():
    suit = random.sample(TRAIN_WITH_TRUMP_SUITS, 1)[0]
    dealer = DealerFactory.get_dealer(TRAIN_GAME_TYPE)
    good_cards, other_cards = dealer.deal(suit)

    hands = [good_cards]
    hands.extend(other_cards)

    card_game = CardGame()
    new_trained_ai_controller = AiController(True)
    card_game.game.controllers = [
        new_trained_ai_controller,
        AiController(False),
        AiController(False),
        AiController(False),
    ]

    main_player_id = card_game.game.players[0].id
    card_game.set_player_hands(main_player_id, hands)
    await card_game.game.announce_hands()
    await card_game.set_game_type(TRAIN_GAME_TYPE, suit)

    return (card_game.game, new_trained_ai_controller)


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

    print("Training started.")

    mean_rewards: list[float] = []

    for epoch in tqdm(
        range(NUM_EPOCHS),
        "Performing training within %i epochs" % (NUM_EPOCHS),
        unit="epochs",
    ):
        game, trained_ai_controller = await get_new_game()
        await game.run()
        if isinstance(trained_ai_controller.play_game_agent, RLBaseAgentTrainer):
            rewards = trained_ai_controller.play_game_agent.rewards
            mean_rewards.append(sum(rewards) / len(rewards))
        trained_ai_controller.play_game_agent.reset()

        if epoch % EPOCHS_UNTIL_APPLYING_TRAINING == 0:
            trained_ai_controller.persist_training_results()
            updated_game, updated_trained_ai_controller = await get_new_game()
            game = updated_game
            trained_ai_controller = updated_trained_ai_controller

    print("Training complete.")

    evaluate(mean_rewards)


if __name__ == "__main__":
    asyncio.run(main())
