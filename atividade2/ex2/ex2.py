# Bibliotecas externas
import numpy as np
import matplotlib.pyplot as plt

# Bibliotecas internas
from typing import Callable
from particle import Particle
import random
import math
from market import Market

SIGNIFICANT_DIGITS = 4

# Constantes do método PSO
SWARM_SIZE = 30
MAX_ITERATIONS = 1000
INDIVIDUAL_LEARNING = 2
GROUP_LEARNING = 2
MAX_VELOCITY = 5
MAX_VARIANCE = math.pow(10, -4) # Critério de parada


# Constantes do modelo de Weber especificado
MARKET_POSITIONS = [
    [100, 100],
    [900, 100],
    [500, 100 + math.sqrt(800 ** 2 - 400 ** 2)]
]
SCENARIOS = [
    [1, 1, 1],
    [1.5, 1, 1],
    [1, 1.7, 1]
]
LOWER_BOUND = [100, 100]
UPPER_BOUND = [900, 100 + math.sqrt(800 ** 2 - 400 ** 2)]


def main():
    # Guarda valores de custo para responder questão b
    optimal_costs = []
    
    # (a)
    for (j, s) in enumerate(SCENARIOS):
        markets = [Market(MARKET_POSITIONS[i], w) for (i, w) in enumerate(s)]
        g_best, g_best_values = solve_by_pso(cost_function, LOWER_BOUND, UPPER_BOUND, [markets])
        optimal_costs.append(g_best_values)
        print(f"Cenário: {j + 1}")
        print(f"x: {format_array(g_best, SIGNIFICANT_DIGITS)}\nf: {g_best_values[-1]:.4g}")
    
    # (b)
    fig, axs = plt.subplots(3)
    for (i, cost) in enumerate(optimal_costs):
        axs[i].plot(cost, 'ob')
        axs[i].set_title(f'({(i + 1) * "i"})')

    plt.tight_layout()
    plt.savefig('ex2.png')
    plt.show()

    return


# Função de custos do modelo de Weber
def cost_function(input: np.ndarray, markets: list[Market]) -> float:
    sum = 0
    for m in markets:
        sum += m.weight * np.linalg.norm(m.pos - input)
    return sum


# Função de Rosenbrock para duas variáveis.
# Usada apenas para testes.
def rosenbrock(x: np.ndarray):
    return math.pow(1 - x[0], 2) + 100 * math.pow(x[1] - math.pow(x[0], 2), 2)


# Encontra o mínimo de func pelo método do PSO
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


# Usado para formatar conteúdo de array para N algarismos significativos.
def format_array(array: np.ndarray, significant_digits: int) -> list[str]:
    return [f"{x:.{significant_digits}g}" for x in array]


# Usado para formatar conteúdo de array para N algarismos significativos.
def format_array(array: np.ndarray, significant_digits: int) -> list[str]:
    return [f"{x:.{significant_digits}g}" for x in array]


if __name__ == "__main__":
    main()

