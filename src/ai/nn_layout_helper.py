import logging
from collections import deque
from typing import Callable, Sequence


def times_two(val: int):
    return val * 2


def plus_five(val: int):
    return val + 5


def plus_ten(val: int):
    return val + 10


def compute_layers(
    neurons: int, layers: int, auto_encoder_neurons: int | None = None
) -> Sequence[Sequence[int]]:
    all_combinations = list(
        compute_layers_combinations(
            neurons_range=(neurons, neurons),
            neurons_increment=plus_five,
            layers_range=(layers, layers),
            layers_increment=plus_five,
            auto_encoder_threshold=0
            if auto_encoder_neurons is not None
            else layers + 1,
            auto_encoder_neurons_range=(
                auto_encoder_neurons or 0,
                auto_encoder_neurons or 0,
            ),
            auto_encoder_neurons_increment=plus_five,
        )
    )
    return all_combinations[-1:]


def compute_layers_combinations(
    neurons_range: tuple[int, int],
    neurons_increment: Callable[[int], int],
    layers_range: tuple[int, int],
    layers_increment: Callable[[int], int],
    auto_encoder_threshold: int,
    auto_encoder_neurons_range: tuple[int, int],
    auto_encoder_neurons_increment: Callable[[int], int],
):
    combinations: Sequence[list[int]] = deque()

    layers = layers_range[0]
    while layers <= layers_range[1]:
        neurons = neurons_range[0]
        while neurons <= neurons_range[1]:
            layers_list = [neurons] * layers
            combinations.append(layers_list)

            if layers >= auto_encoder_threshold:
                ae_neurons = auto_encoder_neurons_range[0]
                while ae_neurons <= auto_encoder_neurons_range[1]:
                    ae_layers_list = [neurons] * (layers // 2)
                    ae_layers_list.append(ae_neurons)
                    ae_layers_list.extend([neurons] * (layers // 2 + layers % 2))
                    combinations.append(ae_layers_list)
                    ae_neurons = max(
                        ae_neurons + 1,
                        min(
                            auto_encoder_neurons_increment(ae_neurons),
                            auto_encoder_neurons_range[1],
                        ),
                    )

            neurons = max(
                neurons + 1, min(neurons_increment(neurons), neurons_range[1])
            )
        layers = max(layers + 1, min(layers_increment(layers), layers_range[1]))

    return combinations


def get_combination_details(layers: Sequence[int]):
    num_layers = len(layers)
    neurons = layers[0]
    for layer in layers:
        ae_neurons = layers[0]
        if layer < ae_neurons:
            ae_neurons = layer
            return (num_layers, neurons, ae_neurons)
    return (num_layers, neurons, None)


def print_combinations(combinations: Sequence[Sequence[int]], logger: logging.Logger):
    for combination in combinations:
        (layers, neurons, ae_neurons) = get_combination_details(combination)
        if ae_neurons is not None:
            logger.info(
                "🪗 layers %i, ⚛ neurons %i, 🗜️ auto encoder neurons in between %i",
                layers,
                neurons,
                ae_neurons,
            )
        else:
            logger.info("🪗 layers %i, ⚛ neurons %i", layers, neurons)
