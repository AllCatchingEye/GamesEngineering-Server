import asyncio
import logging
import os
import random
from typing import Sequence

from tqdm import tqdm

from ai.dealer.dealer_factory import DealerFactory
from ai.nn_layout_helper import (
    compute_layers_combinations,
    get_combination_details,
    print_combinations,
)
from ai.nn_meta_training_helper import store_winrates_money_in_csv
from ai.select_card.drl_agent_trainer import DRLAgentTrainer
from arena import Arena
from controller.ai_controller import AiController
from controller.passive_controller import PassiveController
from controller.player_controller import PlayerController
from logic.card_game import CardGame
from state.gametypes import Gametype
from state.suits import Suit

ALL_SUITS = [Suit.EICHEL, Suit.GRAS, Suit.HERZ, Suit.SCHELLEN]
ALL_GAME_TYPES = [
    Gametype.SAUSPIEL,
    Gametype.RAMSCH,
    Gametype.SOLO,
    Gametype.WENZ,
    Gametype.FARBWENZ,
    Gametype.GEIER,
    Gametype.FARBGEIER,
]
TRAINING_CONFIG = {
    Gametype.SAUSPIEL.name: [Suit.EICHEL, Suit.GRAS, Suit.SCHELLEN],
    Gametype.RAMSCH.name: None,
    Gametype.SOLO.name: ALL_SUITS,
    Gametype.WENZ.name: None,
    Gametype.FARBWENZ.name: ALL_SUITS,
    Gametype.GEIER.name: None,
    Gametype.FARBGEIER.name: ALL_SUITS,
}

# Modify this array to train only certain game types.
GAME_TYPES_TO_TRAIN: list[Gametype] = ALL_GAME_TYPES

NUM_EPOCHS = 100_000
EPOCHS_UNTIL_APPLYING_TRAINING = 250
DISPLAY_LOGS = False
DISPLAY_PROGRESS = True
UPDATE_PROGRESS_INTERVAL_MS = 0.1
EPOCHS_UNTIL_ARENA_EVALUATION = 1_000


# NET_COMBINATIONS = compute_layers(neurons=256, layers=5)
NET_COMBINATIONS = compute_layers_combinations(
    neurons_range=(64, 256),
    neurons_increment=lambda x: x * 2,
    layers_range=(5, 20),
    layers_increment=lambda x: x + 5,
    auto_encoder_threshold=8,
    auto_encoder_neurons_range=(10, 40),
    auto_encoder_neurons_increment=lambda x: x + 30,
)

logger = logging.getLogger("DRL-Trainings-Loop")


async def get_new_game(
    game_type: Gametype,
    playable_suits: list[Suit] | None,
    net_layers: list[int],
    reused_controllers: tuple[AiController, list[PlayerController]] | None = None,
):
    suit = None if playable_suits is None else random.sample(playable_suits, 1)[0]
    dealer = DealerFactory.get_dealer(game_type)
    good_cards, other_cards = dealer.deal(suit)

    hands = [good_cards]
    hands.extend(other_cards)

    card_game = CardGame()
    if reused_controllers is not None:
        card_game.game.controllers = [reused_controllers[0]] + reused_controllers[1]
        trained_controller = reused_controllers[0]
    else:
        trained_controller = AiController(net_layers, True)
        new_controllers: list[PlayerController] = [
            trained_controller,
            AiController(net_layers, False),
            AiController(net_layers, False),
            AiController(net_layers, False),
        ]
        card_game.game.controllers = new_controllers

    main_player_id = card_game.game.players[0].id
    card_game.set_player_hands(main_player_id, hands)
    await card_game.game.announce_hands()
    await card_game.set_game_type(game_type, suit)

    return (card_game.game, trained_controller, card_game.game.controllers[1:])


async def main():
    if not DISPLAY_PROGRESS:
        logging.basicConfig(level=logging.DEBUG)

    logger.info("Training started.")
    logger.info("Trained game types: %s", GAME_TYPES_TO_TRAIN)
    logger.info("Trained combinations:")
    print_combinations(NET_COMBINATIONS, logger)

    for net_layers in NET_COMBINATIONS:
        for game_type in GAME_TYPES_TO_TRAIN:
            playable_suits = TRAINING_CONFIG[game_type.name]
            last_controllers: tuple[AiController, list[PlayerController]] | None = None

            logger.info("Performing training for one combination.")
            (layers, neurons, ae_neurons) = get_combination_details(net_layers)
            print_combinations([net_layers], logger)

            for epoch in tqdm(
                range(NUM_EPOCHS),
                "%s | %ix%i%s"
                % (
                    str(game_type).replace("Gametype.", ""),
                    layers,
                    neurons,
                    "+%i" % ae_neurons if ae_neurons is not None else "",
                ),
                unit="epochs",
            ):
                if (
                    epoch % EPOCHS_UNTIL_APPLYING_TRAINING == 0
                    or last_controllers is None
                ):
                    if last_controllers is not None:
                        last_controllers[0].persist_training_results()
                    game, trained_ai_controller, other_controllers = await get_new_game(
                        game_type, playable_suits, list(net_layers)
                    )
                else:
                    game, trained_ai_controller, other_controllers = await get_new_game(
                        game_type, playable_suits, list(net_layers), last_controllers
                    )

                last_controllers = (trained_ai_controller, other_controllers)
                agent = trained_ai_controller.play_game_agent
                if isinstance(agent, DRLAgentTrainer):
                    agent.steps_done = epoch

                await game.run()

                trained_ai_controller.play_game_agent.reset()

                if epoch % EPOCHS_UNTIL_ARENA_EVALUATION == 0:
                    if isinstance(agent, DRLAgentTrainer):
                        logger.debug(
                            "EPS_DECAY: %i", agent.get_eps_threshold(agent.steps_done)
                        )

                    await run_arena(game_type, net_layers, epoch)

                if epoch % 1_000 == 0:
                    if isinstance(agent, DRLAgentTrainer):
                        logger.debug("\nFirst Parameter: %s", agent.agent.model.get_raw_model().parameters().__next__().tolist()[0][0])

            if last_controllers is not None:
                last_controllers[0].persist_training_results()

    logger.info("Training complete.")



async def run_arena(game_type: Gametype, layers: Sequence[int], epoch: int):
    def get_ai_ctrl() -> AiController:
        return AiController(list(layers))
    logger.info("Arena Evaluation: ")
    arena = Arena()
    arena.add_bot(get_ai_ctrl)
    arena.add_bot(PassiveController)
    arena.add_bot(PassiveController)
    arena.add_bot(PassiveController)
    arena.config.games = 1000
    arena.config.rounds_per_game = 1
    await arena.run_game_type(game_type)
    df = arena.results_overview()
    winrates = df["Winrate"]
    money = df["Money"]

    net_combination = "x".join([str(it) for it in layers])

    from_here = os.path.dirname(os.path.abspath(__file__))
    store_winrates_money_in_csv(
        os.path.join(from_here, "meta_logs_%s_%s.csv" % (net_combination, game_type)),
        epoch,
        str(winrates[0]),
        str(money[0]),
    )
    logger.info(
        "\nEpoch: %i | Net: %s | Winrates: %s | Money: %s",
        epoch,
        net_combination,
        winrates[0],
        money[0],
    )


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
