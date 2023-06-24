# Arquivos externos
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.optimize as opt
from tabulate import tabulate

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

# Isodapanas
ISOTIMS = [
    100,
    500,
    1000
]
ISOTIM_LABELS = [r"$C_{min}$" + f"$ + {value}$" for value in ISOTIMS]

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
    # Como equivalente de 'fminsearch()', usou-se scipy.optmizie.fmin
    # Ambas usam o mesmo método (Nelder-Mead) para encontrar o ponto de ótimo
    opt_results = []
    for scenario in scenarios:
        cost_function = lambda x: sum([market.weight * np.linalg.norm(market.coord - x) for market in scenario.markets])
        opt_results.append(opt.minimize(cost_function, np.array(INITIAL_POSITION)))

    # Extrai localizações das otimizações
    opt_loc = [result.x.tolist() for result in opt_results]
    estimated_locations = [scenario.optimal_location.tolist() for scenario in scenarios]

    # Calcula erro relativo percentual
    errors = []
    for (loc, estimate) in zip(opt_loc, estimated_locations):
        errors.append([get_relative_error(estimate[0], loc[0]), get_relative_error(estimate[1], loc[1])])

    print("Compração localização")
    print(tabulate({
            "Cenário": ["(i)", "(ii)", "(iii)"],
            "Estimativa localização ótima": estimated_locations,
            "Resultado scipy": opt_loc,
            "Erro (%)": errors
        },
        headers="keys"
    ))

    # Resposta (d)
    figure = plt.figure("Exercício 3d")
    X = np.linspace(-300, 1250, 100)
    Y = np.linspace(-400, 1150, 100)

    for (i, scenario) in enumerate(scenarios):
        axis = figure.add_subplot(310 + 1 + i)
        Z = np.array([[find_minimal_cost(x, y, scenario) for x in X] for y in Y])
        
        levels = [scenario.minimal_cost + isotim for isotim in ISOTIMS]
        contours = axis.contour(X, Y, Z, levels=levels)

        # Coloca legendas para isodapanas
        fmt = {}
        for (i, contour) in enumerate(contours.levels):
            fmt[contour] = ISOTIM_LABELS[i]
        axis.clabel(contours, levels=levels, fmt=fmt)

        # Desenha triâgulo locacional e identifica localização ótima
        plot_triangle(axis)
        axis.plot(
            scenario.optimal_location[0],
            scenario.optimal_location[1],
            marker='o',
            color="#ff0000",
            label="Localização ótima"
        )
        axis.legend()

    # Aumenta a figura
    figure.set_figheight(15)
    figure.set_figwidth(8)
    figure.savefig("Exercício 3d.png")


# Desenha o triângulo locacional de Weber
def plot_triangle(axis: plt.Axes):
    x_pos = [pos[0] for pos in POSITIONS]
    y_pos = [pos[1] for pos in POSITIONS]
    axis.fill(x_pos, y_pos, fill=False, color="#0000ff")


# Encontra custo mínimo a partir de x e y
def find_minimal_cost(x: float, y: float, scenario: Scenario) -> float:
    point = np.array([x, y])
    cost = 0
    for market in scenario.markets:
        cost += market.weight * np.linalg.norm(point - market.coord)
    return cost


# Calcula o erro relativo percentual
def get_relative_error(estimate: float, real_value: float) -> float:
    return (real_value - estimate) / real_value * 100


if __name__ == "__main__":
    main()

