from market import Market
import numpy as np


class Scenario:
    markets: list[Market]
    solutions: list[np.ndarray] # Lista que contém aproximações para encontrar solução
    optimal_location: np.ndarray
    minimal_cost: float

    def __init__(self, coord_list: list[tuple[float]], weight_list: list[float]):
        self.markets = []
        for coord, weight in zip(coord_list, weight_list):
            self.markets.append(Market(coord, weight))

    def __repr__(self):
        return f"Localização ótima: {self.optimal_location[0], self.optimal_location[1]}"

    # Encontra a localização ótima para a firma
    def find_optimal_location(self, initial_condition: tuple[float], max_error: float, max_iterations: int):
        self.solutions = [np.array([initial_condition], dtype=np.float64)]

        error = 10 * max_error
        i = 1
        while error > max_error and i < max_iterations:
            curr_solution_numerator = np.array([0, 0]).astype('float')
            curr_solution_denominator = np.array([0, 0]).astype('float')
            
            # Calcula Sk (solução atual)
            for market in self.markets:
                curr_solution_numerator += market.weight * market.coord / np.linalg.norm(self.solutions[i - 1] - market.coord)
                curr_solution_denominator += market.weight / np.linalg.norm(self.solutions[i - 1] - market.coord)

            self.solutions.append(curr_solution_numerator / curr_solution_denominator)
            # Calcula erro atual
            error = np.linalg.norm(self.solutions[i] - self.solutions[i - 1]) / np.linalg.norm(self.solutions[i - 1])
            i += 1
        self.optimal_location = self.solutions[-1]

    # Encontra custo mínimo na localização ótima
    def find_minimal_cost(self):
        self.minimal_cost = 0
        for market in self.markets:
            self.minimal_cost += market.weight * np.linalg.norm(self.optimal_location - market.coord)

