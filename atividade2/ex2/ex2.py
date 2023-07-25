# Bibliotecas externas
import numpy as np

# Bibliotecas internas
from typing import Callable
from particle import Particle
import random
import math


SWARM_SIZE = 30
#LOWER_BOUND = [100, 100]
#UPPER_BOUND = [900, 100 + math.sqrt(800 ** 2 - 400 ** 2)]
LOWER_BOUND = [-35, -35]
UPPER_BOUND = [35, 35]
MAX_ITERATIONS = 1000
INDIVIDUAL_LEARNING = 2
GROUP_LEARNING = 2
MAX_VELOCITY = 5


def main():
    g_best, g_best_value = solve_by_pso(rosenbrock)
    print(f"x: {g_best}\nf: {g_best_value}")
    return


# Função de Rosenbrock para duas variáveis
def rosenbrock(x: np.ndarray):
    return math.pow(1 - x[0], 2) + 100 * math.pow(x[1] - math.pow(x[0], 2), 2)


# Encontra o mínimo de func pelo método do PSO
def solve_by_pso(func: Callable) -> tuple[np.ndarray, float]:
    Particle.max_velocity = MAX_VELOCITY
    
    # Inicializa enxame
    particles: list[Particle] = []
    for _ in range(SWARM_SIZE):
        x = float(random.randrange(LOWER_BOUND[0], UPPER_BOUND[0]))
        y = float(random.randrange(LOWER_BOUND[1], UPPER_BOUND[1]))
        particles.append(Particle([x, y]))
    
    # Inicializa g_best        
    g_best = particles[0].best.copy()

    # Roda o algoritmo
    for _ in range(MAX_ITERATIONS):
        # Encontra novo g_best
        for p in particles:
            if func(p.best) < func(g_best):
                g_best = p.best.copy()
        
        for p in particles:
            p.update_velocity(g_best, INDIVIDUAL_LEARNING, GROUP_LEARNING)
            p.pos += p.vel # Move a partícula
            # Escolhe novo p_best
            if func(p.pos) < func(p.best):
                p.best = p.pos.copy()
    
    return g_best, func(g_best)


if __name__ == "__main__":
    main()

