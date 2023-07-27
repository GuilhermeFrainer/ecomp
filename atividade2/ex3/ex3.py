# Bibliotecas externas
import numpy as np

# Bibliotecas internas
import math
from good import Good
from firm import Firm


ALPHA_1 = 0.4
ALPHA_2 = 0.4
GAMMA = 1 - ALPHA_1 - ALPHA_2
A = 1
K = GAMMA * math.pow(A * math.pow(ALPHA_1, ALPHA_1) * math.pow(ALPHA_2, ALPHA_2), 1 / GAMMA)

# Preços FOB
GOOD_PRICE = 400
INPUT_1_PRICE = 100
INPUT_2_PRICE = 100

# Custos de transporte
GOOD_TRANSPORT_COST = 0.15
INPUT_1_TRANSPORT_COST = 0.25
INPUT_2_TRANSPORT_COST = 0.15

# Localizações
GOOD_POS = [500, 100 + math.sqrt(800 ** 2 - 400 ** 2)]
INPUT_1_POS = [100, 100]
INPUT_2_POS = [900, 100]


def main():
    # Inicializa bens e firma
    product = Good(GOOD_PRICE, GOOD_POS, GOOD_TRANSPORT_COST)
    input1 = Good(INPUT_1_PRICE, INPUT_1_POS, INPUT_1_TRANSPORT_COST)
    input2 = Good(INPUT_2_PRICE, INPUT_2_POS, INPUT_2_TRANSPORT_COST)
    firm = Firm(product, [input1, input2])


# Função de lucro no modelo de Moses.
# Retorna o lucro dada uma posição para a firma.
def moses_profit(pos: np.ndarray, firm: Firm) -> float:
    denominator = math.pow(firm.inputs[0].get_price_for_input(pos), ALPHA_1)
    denominator *= math.pow(firm.inputs[1].get_price_for_input(pos), ALPHA_2)
    return K * math.pow(firm.product.get_revenue_from_good(pos) / denominator, 1 / GAMMA)


if __name__ == "__main__":
    main()