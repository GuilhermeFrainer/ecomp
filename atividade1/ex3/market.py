import numpy as np


class Market:
    weight: float
    coord: np.ndarray

    # Inicializa mercado
    def __init__(self, position: tuple[float], weight: float):
        self.weight = weight
        self.coord = np.array(position)

    def __repr__(self):
        return f"Posição: {self.coord}\nPeso: {self.weight}"

