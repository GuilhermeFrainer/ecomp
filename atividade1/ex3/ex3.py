# Arquivos externos
import numpy as np
import math

# Arquivos internos
from market import Market
from scenario import Scenario


# Condições iniciais
POSITIONS = [
    (100, 100),
    (900, 100),
    (500, 100 + math.sqrt(800 ** 2) - math.sqrt(400 ** 2))
]

WEIGHTS_1 = [1, 1, 1]
WEIGHTS_2 = [1.5, 1, 1]
WEIGHTS_3 = [1, 1.7, 1]

SCENARIOS = [WEIGHTS_1, WEIGHTS_2, WEIGHTS_3]

MAX_ITERATIONS = 1000
MAX_ERROR = 10 ** -5
INITIAL_POSITION = (200, 200)


def main():
    scenarios = [Scenario(POSITIONS, scenario) for scenario in SCENARIOS]
    for scenario in scenarios:
        scenario.find_optimal_location(INITIAL_POSITION, MAX_ERROR, MAX_ITERATIONS)
        print(scenario)
        scenario.find_minimal_cost()
        print(f"Custo mínimo: {scenario.minimal_cost}")

if __name__ == "__main__":
    main()

