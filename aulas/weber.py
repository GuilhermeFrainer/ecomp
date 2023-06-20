# Implementa o modelo de Weber para n firmas

import numpy as np


MAX_ERROR = 10**-4
MAX_ITERATIONS = 1000
START_CONDITION = (5, 5)


class Market:
    weight: int
    coord: np.ndarray

    def __init__(self, position: list[int], weight: int):
        self.weight = weight
        self.coord = np.array(position)

    def __repr__(self):
        return f"Position: {self.coord}\nWeight: {self.weight}"


def main():
    # Inicializa mercados
    markets: list[Market] = []
    markets.append(Market([0, 0], 1)) 
    markets.append(Market([10, 2], 1))
    markets.append(Market([1, 15], 1))

    solutions = find_solution(markets)

    cost = get_cost(solutions[-1], markets)

    print(f"Solução: {solutions[-1]}")
    print(f"Custo no ponto de ótimo: {cost}")
    for (i, market) in enumerate(markets):
        distance = np.linalg.norm(solutions[-1] - market.coord)
        print(f"Distance to market {i + 1} is {distance}")

    print(f"Custo no ponto 5, 5: {get_cost(np.array([5,5]), markets)}")


# Encontra a solução do problema
def find_solution(markets: list[Market]) -> list[np.ndarray]:
    # Inicializa lista com soluções
    # Usa a condição inicial como primeira solução
    solutions: list[np.ndarray] = [np.array(START_CONDITION).astype('float')]

    error = 1
    i = 1
    while error > MAX_ERROR and i < MAX_ITERATIONS:
        curr_solution_numerator = np.array([0, 0]).astype('float')
        curr_solution_denominator = np.array([0, 0]).astype('float')
        
        # Calcula Sk (solução atual)
        for market in markets:
            numerator = market.weight * market.coord
            denominator = np.linalg.norm(solutions[i - 1] - market.coord)
            curr_solution_numerator += numerator / denominator
            curr_solution_denominator += market.weight / np.linalg.norm(solutions[i - 1] - market.coord)

        solutions.append(curr_solution_numerator / curr_solution_denominator)
        # Calcula erro atual
        error = np.linalg.norm(solutions[i] - solutions[i - 1]) / np.linalg.norm(solutions[i - 1])
        i += 1


# Encontra custo no ponto "point"
def get_cost(point: np.ndarray, markets: list[Market]) -> float:
    cost = 0
    for market in markets:
        cost += market.weight * np.linalg.norm(point - market.coord)
    return cost


if __name__ == "__main__":
    main()

