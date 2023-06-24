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
    (500, 100 + math.sqrt(800 ** 2 - 400 ** 2))
]

WEIGHTS_1 = [1, 1, 1]
WEIGHTS_2 = [1.5, 1, 1]
WEIGHTS_3 = [1, 1.7, 1]

SCENARIOS = [WEIGHTS_1, WEIGHTS_2, WEIGHTS_3]

MAX_ITERATIONS = 1000
MAX_ERROR = 10 ** -5
INITIAL_POSITION = (200, 200)

ISOTIMS = [
    100,
    500,
    1000
]


def main():
    # Reposta (a)
    # Salva os cenários (i), (ii) e (iii)
    scenarios = [Scenario(POSITIONS, scenario) for scenario in SCENARIOS]
    for (i, scenario) in enumerate(scenarios):
        scenario.find_optimal_location(INITIAL_POSITION, MAX_ERROR, MAX_ITERATIONS)
        print(f"({'i' * (i + 1)}): {scenario}")

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

    # Coloca localizações ótimas no gráfico
    for (i, scenario) in enumerate(scenarios):
        x, y = scenario.optimal_location[0], scenario.optimal_location[1]
        axis.plot(x, y, marker='o', color="#ff0000")
        axis.annotate(f"({'i' * (i + 1)})", xy=(x, y), xytext=(x + 10, y), color="#ff0000")

    figure.savefig("Exercício 3a.png")

    # Resposta (b)
    # Calcula e printa os custos mínimos
    for (i, scenario) in enumerate(scenarios):
        print(f"Custo mínimo ({'i' * (i + 1)}): {scenario.find_minimal_cost()}")

    # Resposta (c)
    # TODO -> procurar equivalente de 'fminsearch()' no Python

    # Resposta (d)
    figure = plt.figure("Exercício 3d")
    axis = figure.add_subplot(111)
    X = np.linspace(-300, 1250, 100)
    Y = np.linspace(-400, 1150, 100)

    Z = np.array([[find_minimal_cost(x, y, scenarios[0]) for x in X] for y in Y])

    # THIS WORKS!!!!
    levels = [scenarios[0].minimal_cost + isotim for isotim in ISOTIMS]
    axis.contour(X, Y, Z, levels=levels)

    figure.savefig("Exercício 3d.png")


# Encontra custo mínimo a partir de x e y
def find_minimal_cost(x: float, y: float, scenario: Scenario) -> float:
    point = np.array([x, y])
    cost = 0
    for market in scenario.markets:
        cost += market.weight * np.linalg.norm(point - market.coord)
    return cost


if __name__ == "__main__":
    main()

