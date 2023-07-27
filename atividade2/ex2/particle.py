# Representa partícula para resolver função de 2 variáveis com PSO
import numpy as np

class Particle:
    pos: np.ndarray # Posição
    vel: np.ndarray # Velocidade
    best: np.ndarray # Posição que gera o melhor valor da função já atingido pela partícula

    max_velocity = 5 # Variável de classe. Cinco por padrão.

    def __init__(self, pos: list):
        self.pos = np.array(pos)
        self.best = self.pos.copy()
        self.vel = np.array([0.0, 0.0])


    # Usado apenas para testes.
    def __repr__(self):
        return f"{np.round(self.pos, decimals=3)}"

    
    # Atualiza velocidade da partícula
    def update_velocity(self, g_best: np.ndarray, individual_learning: float, group_learning: float):
        r1 = np.random.uniform()
        r2 = np.random.uniform()
        self.vel += individual_learning * r1 * (self.best - self.pos)
        self.vel += group_learning * r2 * (g_best - self.pos)
        if np.linalg.norm(self.vel) > Particle.max_velocity:
            self.vel = (self.vel / np.linalg.norm(self.vel)) * Particle.max_velocity

