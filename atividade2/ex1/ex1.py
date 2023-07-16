from typing import Callable
import math
import matplotlib.pyplot as plt
import numpy as np

ALPHA = 0.1
B = 5
MAX_N = 0.8
MAX_ITERATIONS = 100
STARTING_N = 0.1

def main():
    x_axis = np.linspace(0, 2, 1000)
    
    # (i)
    f1 = lambda x: B / (1 - min(ALPHA * x, MAX_N))
    f1_series = pred_function(f1)

    # (ii)
    f2 = lambda x: B / (1 - min(ALPHA * math.sqrt(x), MAX_N))
    f2_series = pred_function(f2)
    # (iii)
    f3 = lambda x: B / (1 - min(ALPHA * math.pow(x, 2), MAX_N))
    f3_series = pred_function(f3)


    


# Calcula o modelo de Pred
def pred_function(func: Callable) -> list[float]:
    pred_list = [B / (1 - STARTING_N)]
    
    for i in range(MAX_ITERATIONS):
        pred_list.append(func(pred_list[i]))

    return pred_list


if __name__=="__main__":
    main()

