import numpy as np


# Representa tanto insumos quanto produto
class Good:
    fob: float # Preço FOB.
    pos: np.ndarray # Origem do insumo ou destino do produto
    transport_cost: float # Custo de transporte por unidade de distância


    def __init__(self, fob: float, pos: list[float], transport_cost: float):
        self.fob = fob
        self.pos = np.array(pos)
        self.transport_cost = transport_cost

    
    # Retorna receita recebida pela firma por unidade de produto.
    def get_revenue_from_good(self, origin: np.ndarray) -> float:
        return self.fob - self.transport_cost * np.linalg.norm(self.pos - origin)
    

    # Preço pago pela firma por unidade de insumo.
    def get_price_for_input(self, destination: np.ndarray) -> float:
        return self.fob + self.transport_cost * np.linalg.norm(self.pos - destination)
    
