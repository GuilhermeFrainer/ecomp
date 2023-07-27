import numpy as np
from good import Good


class Firm:
    product: Good # Produto produzido pela firma
    inputs: list[Good] # insumos

    def __init__(self, product: Good, inputs: list[Good]):
        self.product = product
        self.inputs = inputs

