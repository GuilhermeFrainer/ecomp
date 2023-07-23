# Representa partícula para resolver função de 2 variáveis com PSO
import numpy as np


class Particle:
    pos: np.ndarray # Posição
    vel: np.ndarray # Velocidade
    best: np.ndarray # Posição que gera o melhor valor da função já atingido pela partícula
    individual_learning_rate: float # Taxa de aprendizado individual
    group_learning_rate: float # Taxa de aprendizado de grupo

    def __init__(self, pos: list, individual: float, group: float):
        self.pos = np.array(pos)
        self.vel = np.array([0.0, 0.0])
        self.best = self.pos
        self.individual_learning_rate = individual
        self.group_learning_rate = group

    def __repr__(self):
        return f"{self.pos}"
    
    # Atualiza velocidade da partícula
    def update_velocity(self, g_best: np.ndarray):
        r1 = np.random.uniform(0, 1)
        r2 = np.random.uniform(0, 1)
        self.vel += self.individual_learning_rate * r1 * (self.best - self.pos)
        self.vel += self.group_learning_rate * r2 * (g_best - self.pos)

