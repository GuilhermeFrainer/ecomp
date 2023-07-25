import random as rand
import numpy as np


MAX_PARTICLES = 20
MAX_ITERATIONS = 1000
MAX_VELOCITY = 5
VARIABLES = 2
INDIVIDUAL_WEIGHT = 2
SOCIAL_WEIGHT = 2
L = 35 # Delimita espaço das partículas


class Particle:
    pos: np.ndarray
    vel: float

    def __init__(self, pos: list):    
        self.pos = np.array(pos)
        self.vel = 0

    def __repr__(self) -> str:
        return f"{self.pos}"


def main():
    particles = [Particle([rand.randrange(-L, L) for _ in range(VARIABLES)]) for _ in range(MAX_PARTICLES)]


# Otimiza função 'func' por PSO
def pso(func, particles: list[Particle]):
    g_best = 1000
    for i in range(MAX_ITERATIONS):
        for (j, p) in enumerate(particles):
            # PSEUDOCÓDIGO
            # f = func(p)
            # Determinar g_best
            if g_best > func(p):
                g_best = func(p)


if __name__ == "__main__":
    main()

    