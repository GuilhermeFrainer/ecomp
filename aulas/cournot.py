# 03/07/2023
# Calcula a quantidade ótima no duopólio de Cournot dinâmico

#TODO Alternar custo marginal em loop

import numpy as np
import matplotlib.pyplot as plt
import math
import sys


INITIAL_CONDITION = (0.5, 0.1)
MARGINAL_COSTS = (1, 8)
MAX_ITERATIONS = 50


# Classe que representa firma
class Firm:
    marginal_cost: float
    response_vector: list[float]

    def __init__(self, marginal_cost, initial_quantity):
        self.marginal_cost = marginal_cost
        self.response_vector = [initial_quantity]

    # Calcula a resposta para uma iteração
    def calculate_response(self, rival_last_response: float):
        division = rival_last_response / self.marginal_cost
        next_response = math.sqrt(division) - rival_last_response
        if next_response < 0:
            self.response_vector.append(0)
        else:
            self.response_vector.append(next_response)
        
        #try:
        #    self.response_vector.append(math.sqrt(division) - rival_last_response)
        #except ValueError:
        #    self.response_vector.append(0)


def main():
    # Inicializa as firmas
    firm1 = Firm(MARGINAL_COSTS[0], INITIAL_CONDITION[0])
    firm2 = Firm(MARGINAL_COSTS[1], INITIAL_CONDITION[1])

    iterations = [0]    
    for i in range(MAX_ITERATIONS):
        firm1.calculate_response(firm2.response_vector[i])
        firm2.calculate_response(firm1.response_vector[i])
        iterations.append(i + 1)

    print(f"Quantidades de equilíbrio:\nFirma 1: {firm1.response_vector[-1]}\nFirma 2: {firm2.response_vector[-1]}")

    plot_quantity_chart(firm1, firm2, iterations)


# Desenha gráfico com quantidades no eixo y e iterações no eixo x
def plot_quantity_chart(firm1: Firm, firm2: Firm, iterations: list[int]):
    figure = plt.figure(1)
    axis = figure.add_subplot(111)

    axis.plot(iterations, firm1.response_vector, 's')
    axis.plot(iterations, firm2.response_vector, 's')
    axis.set_xlabel("t")
    axis.set_ylabel("q")

    plt.show()


if __name__ == "__main__":
    main()

