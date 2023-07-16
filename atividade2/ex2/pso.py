# Particle Swarm Optimization para uma função de duas variáveis
import random
from typing import Callable
import numpy as np


class Particle:
    velocity: np.ndarray
    p_best: np.ndarray
    p_best_value: float
    current_position: np.ndarray
    current_position_value: float

    # Variáveis de classe
    individual_learning: float
    group_learning: float
    # Por padrão, começam iguais a 2
    # Seguindo Rao, capítulo 13
    individual_learning = 2
    group_learning = 2

    # Inicializa a partícula
    # 'func' recebe 'np.ndarray' como argumento    
    def __init__(self, func: Callable, lower_bound: np.ndarray, upper_bound: np.ndarray):
        # Escolhe posição inicial aleatória para a partícula
        # Limitada ao retângulo com o canto inferior esquerdo em 'lower_bound'
        # E canto superior direito em 'upper_bound'
        x = random.randrange(lower_bound[0], upper_bound[0])
        y = random.randrange(lower_bound[1], upper_bound[1])

        self.velocity = 0 # Velocidade inicial = 0
        self.p_best = np.array([x, y])
        self.current_position = np.array([x, y])
        self.p_best_value = func(self.p_best)
        self.current_position_value = self.p_best_value

    # Muda variável de classe "individual_learning"
    # Isto é, a taxa de aprendizado de cada partícula individual
    @classmethod
    def change_individual_learning(cls, new_learning_rate: float):
        Particle.individual_learning = new_learning_rate
    
    # Muda variável de classe "group_learning"
    # Isto é, a taxa de aprendizado do grupo
    @classmethod
    def change_group_learning(cls, new_learning_rate: float):
        Particle.group_learning = new_learning_rate

    # Encontra a velocidade da partícula
    def find_velocity(self, g_best: np.ndarray):
        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)
        self.velocity += Particle.individual_learning * r1 * (self.p_best - self.current_position)
        self.velocity += Particle.group_learning * r2 * (g_best - self.current_position)


class Swarm:
    particles: list[Particle]
    g_best: np.ndarray
    g_best_value: float
    func: Callable
    max_error: float
    current_error: float

    def __init__(self, n: int, max_error: float, func: Callable, lower_bound: np.ndarray, upper_bound: np.ndarray):
        self.particles = []
        for i in range(n):
            self.particles.append(Particle(func, lower_bound, upper_bound))
            if self.particles[i].p_best_value > self.g_best_value:
                self.g_best = self.particles[i].p_best
                self.g_best_value = self.particles[i].p_best_value
        self.func = func
        self.max_error = max_error
        self.current_error = 1.0 # Inicializa erro para 1
        