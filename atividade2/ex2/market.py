# Representa um mercado ou fornecedor no modelo de Weber
import numpy as np

class Market:
    weight: float
    pos: np.ndarray

    def __init__(self, pos: list[float], weight: float):
        self.pos = np.array(pos)
        self.weight = weight