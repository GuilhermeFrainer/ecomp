# Bibliotecas externas
import numpy as np
import matplotlib.pyplot as plt

# Bibliotecas internas
import math
from good import Good
from firm import Firm
from particle import Particle
from typing import Callable
import random


ALPHA_1 = 0.4
ALPHA_2 = 0.4
GAMMA = 1 - ALPHA_1 - ALPHA_2
A = 1
K = GAMMA * math.pow(A * math.pow(ALPHA_1, ALPHA_1) * math.pow(ALPHA_2, ALPHA_2), 1 / GAMMA)

# Preços FOB
GOOD_PRICE = 400
INPUT_1_PRICE = 100
INPUT_2_PRICE = 100

# Custos de transporte
GOOD_TRANSPORT_COST = 0.15
INPUT_1_TRANSPORT_COST = 0.25
INPUT_2_TRANSPORT_COST = 0.15

# Localizações
GOOD_POS = [500, 100 + math.sqrt(800 ** 2 - 400 ** 2)]
INPUT_1_POS = [100, 100]
INPUT_2_POS = [900, 100]

# Constantes do PSO
SWARM_SIZE = 30
MAX_ITERATIONS = 1000
INDIVIDUAL_LEARNING = 2
GROUP_LEARNING = 2
MAX_VELOCITY = 5
MAX_VARIANCE = math.pow(10, -4) # Critério de parada
# Mesmas delimitações que o exercício anterior
LOWER_BOUND = [100, 100]
UPPER_BOUND = [900, 100 + math.sqrt(800 ** 2 - 400 ** 2)]


def main():
    # Inicializa bens e firma
    product = Good(GOOD_PRICE, GOOD_POS, GOOD_TRANSPORT_COST)
    input1 = Good(INPUT_1_PRICE, INPUT_1_POS, INPUT_1_TRANSPORT_COST)
    input2 = Good(INPUT_2_PRICE, INPUT_2_POS, INPUT_2_TRANSPORT_COST)
    firm = Firm(product, [input1, input2])

    f = lambda x, firm: -moses_profit(x, firm) # Maximiza encontrando o mínimo de -f(x)
    g_best, g_best_values = solve_by_pso(f, LOWER_BOUND, UPPER_BOUND, [firm])
    print(f"Localização ótima: {g_best}\nLucro ótimo: {g_best_values[-1]}\nIterações: {len(g_best_values)}")

    # Custo de transporte total: como?

    # Gráfico da função lucro
    # Dúvida: gráfico em função de que?

    # Mapa de contorno
    X = np.linspace(-300, 1250, 100)
    Y = np.linspace(-400, 1150, 100)
    Z = np.array([[moses_profit(np.array([x, y]), firm) for x in X] for y in Y])

    fig, ax = plt.subplots()
    ax.contour(X, Y, Z)
    plt.show()
    fig.savefig("Mapa de contorno lucro.png")

    # Gráfico de convergência
    fig, ax = plt.subplots()
    ax.plot(g_best_values, 'ob')
    plt.show()
    fig.savefig("Gráfico de convergência.png")


# Função de lucro no modelo de Moses.
# Retorna o lucro dada uma posição para a firma.
def moses_profit(pos: np.ndarray, firm: Firm) -> float:
    denominator = math.pow(firm.inputs[0].get_price_for_input(pos), ALPHA_1)
    denominator *= math.pow(firm.inputs[1].get_price_for_input(pos), ALPHA_2)
    return K * math.pow(firm.product.get_revenue_from_good(pos) / denominator, 1 / GAMMA)


# Encontra o mínimo de func pelo método do PSO.
# Não foi necessário adaptar o método do exercício 2.
def solve_by_pso(func: Callable, lower_bound: list[float], upper_bound: list[float], args=()) -> tuple[np.ndarray, list[float], int]:
    Particle.max_velocity = MAX_VELOCITY
    g_best_values = [] # Guarda histórico do valor em g_best

    # Inicializa enxame
    particles: list[Particle] = []
    for _ in range(SWARM_SIZE):
        # Inicializa no retângulo entre LOWER_BOUND e UPPER_BOUND
        x = float(random.randrange(int(lower_bound[0]), int(upper_bound[0])))
        y = float(random.randrange(int(lower_bound[1]), int(upper_bound[1])))
        particles.append(Particle([x, y]))
    
    # Inicializa g_best        
    g_best = particles[0].best.copy()

    # Roda o algoritmo
    for _ in range(MAX_ITERATIONS):
        # Encontra novo g_best
        for p in particles:
            if func(p.best, *args) < func(g_best, *args):
                g_best = p.best.copy()
        
        for p in particles:
            p.update_velocity(g_best, INDIVIDUAL_LEARNING, GROUP_LEARNING)
            p.pos += p.vel # Move a partícula
            # Escolhe novo p_best
            if func(p.pos, *args) < func(p.best, *args):
                p.best = p.pos.copy()

        # Salva valor em g_best
        g_best_values.append(func(g_best, *args))

        # Critério de parada
        if np.var([func(p.best, *args) for p in particles]) < MAX_VARIANCE:
            break
    
    return g_best, g_best_values


if __name__ == "__main__":
    main()