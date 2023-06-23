# Arquivos externos
import numpy as np
import matplotlib.pyplot as plt
import math

# Arquivos internos
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
    # Salva os cenários (i), (ii) e (iii)
    scenarios = [Scenario(POSITIONS, scenario) for scenario in SCENARIOS]
    for scenario in scenarios:
        scenario.find_optimal_location(INITIAL_POSITION, MAX_ERROR, MAX_ITERATIONS)
        print(scenario)
        scenario.find_minimal_cost()
        print(f"Custo mínimo: {scenario.minimal_cost}")

    figure = plt.figure("Exercício 3a")
    axis = figure.add_subplot(111)

    # Desenha triângulo locacional de Weber
    x_pos = [pos[0] for pos in POSITIONS]
    y_pos = [pos[1] for pos in POSITIONS]
    axis.fill(x_pos, y_pos, fill=False, color="#0000ff")

    # Coloca legenda indicando mercados e fornecedor
    for (i, (x, y)) in enumerate(zip(x_pos, y_pos)):
        axis.plot(x, y, marker='o', color="#0000ff")
        axis.annotate(f"$x_{i + 1}$", xy=(x, y), xytext=(x + 10, y))

    figure.savefig("Exercício 3a.png")


if __name__ == "__main__":
    main()

