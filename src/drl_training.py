import asyncio
import logging
import random

from tqdm import tqdm

from ai.dealer.dealer_factory import DealerFactory
from controller.ai_controller import AiController
from controller.player_controller import PlayerController
from logic.card_game import CardGame
from state.gametypes import Gametype
from state.suits import Suit

ALL_SUITS = [Suit.EICHEL, Suit.GRAS, Suit.HERZ, Suit.SCHELLEN]
ALL_GAME_TYPES = [Gametype.SAUSPIEL, Gametype.RAMSCH, Gametype.SOLO, Gametype.WENZ, Gametype.FARBWENZ, Gametype.GEIER, Gametype.FARBGEIER]
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

NUM_EPOCHS = 10_000_000
EPOCHS_UNTIL_APPLYING_TRAINING = 100
DISPLAY_LOGS = False
DISPLAY_PROGRESS = True
UPDATE_PROGRESS_INTERVAL_MS = 0.1


async def get_new_game(
    game_type: Gametype,
    playable_suits: list[Suit] | None,
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
        trained_controller = AiController(True)
        new_controllers: list[PlayerController] = [
            trained_controller,
            AiController(False),
            AiController(False),
            AiController(False),
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

    print("Training started. Trained game types: %s"%(GAME_TYPES_TO_TRAIN))

    for game_type in GAME_TYPES_TO_TRAIN:
        playable_suits = TRAINING_CONFIG[game_type.name]
        last_controllers: tuple[AiController, list[PlayerController]] | None = None

        for epoch in tqdm(
            range(NUM_EPOCHS),
            "Performing training for %s within %i epochs" % (game_type, NUM_EPOCHS),
            unit="epochs",
        ):
            if epoch % EPOCHS_UNTIL_APPLYING_TRAINING == 0 or last_controllers is None:
                if last_controllers is not None:
                    last_controllers[0].persist_training_results()
                game, trained_ai_controller, other_controllers = await get_new_game(
                    game_type, playable_suits
                )
            else:
                game, trained_ai_controller, other_controllers = await get_new_game(
                    game_type, playable_suits, last_controllers
                )

            last_controllers = (trained_ai_controller, other_controllers)
            await game.run()

            trained_ai_controller.play_game_agent.reset()
    
        if last_controllers is not None:
            last_controllers[0].persist_training_results()
            
    print("Training complete.")


if __name__ == "__main__":
    # logging.basicConfig(level=logging.DEBUG)
    asyncio.run(main())
