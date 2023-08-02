# Resolve PVI do modelo de Solow pelo método de Euler

# (i) Resolver pelo método de Euler até que o capital per capita (k) atinja o equilíbrio
# Fazer gráficos para K(t), L(t) e k(t) e comparar solução com solução de biblioteca

# (ii) Variar s entre 0 e 1 e fazer gráfico do k de equilíbrio em função de s

import matplotlib.pyplot as plt
import scipy.integrate as scint
import numpy as np

import math

# Constantes do modelo
# Taxas são anuais
S = 0.2 # Taxa marginal de poupança
ALPHA = 1 / 3
BETA = 0.02
DELTA = 0.05

# Condições iniciais
K0 = 0.001
L0 = 0.001

# Constantes do método

MAX_ITERATIONS = 1000
MAX_ERROR = math.pow(10, -5)
T = 1 / 10 # Delta t escolhido para o método

def main():
    global S
    
    capital_history, pop_history, per_capita_capital, i = euler_method(S)

    # Printa soluções no equilíbrio
    print(f"Equilíbrio atingido!\nIterações: {i}\nCapital: {capital_history[-1]}")
    print(f"População: {pop_history[-1]}\nCapital per capita: {capital_history[-1] / pop_history[-1]}")
    
    # Gráficos

    # K(t)
    fig, ax = plt.subplots()
    ax.plot(capital_history)
    ax.set_title("K(t)")
    #plt.show()

    # L(t)
    fig, ax = plt.subplots()
    ax.plot(pop_history)
    ax.set_title("L(t)")
    #plt.show()

    # k(t)
    fig, ax = plt.subplots()
    ax.plot(per_capita_capital)
    ax.set_title("k(t)")
    plt.show()

    # Solução scipy
    y0 = [K0, L0]
    t = np.linspace(0, 100, 1000)
    sol = scint.odeint(solow_system, y0, t, args=(ALPHA, BETA, DELTA, S))
    #print(sol)
    print(sol[-1, 0] / sol[-1, 1])

    # Calcula estados de equilíbrio em função de S
    var_s = np.linspace(0, 1, 100)
    steady_states = []
    for s in var_s:
        capital, pop, per_capita, i = euler_method(s)
        steady_states.append(per_capita[-1])

    fig, ax = plt.subplots()
    ax.plot(var_s, steady_states)
    ax.set_title("$k_\infty (t)$")
    plt.show()

    return

# Resolve modelo de Solow pelo método de Euler
def euler_method(s: float) -> tuple[list[float], list[float], list[float], int]:
    capital_history = [K0]
    pop_history = [L0]
    per_capita_capital = [K0 / L0]
    current_error = MAX_ERROR + 1
    i = 1

    while current_error > MAX_ERROR and i < MAX_ITERATIONS:
        capital_history.append(capital_history[i - 1] + T * capital_accumulation(s, capital_history[i - 1], pop_history[i - 1]))
        pop_history.append(pop_history[i - 1] + T * pop_growth(pop_history[i - 1]))
        per_capita_capital.append(capital_history[i] / pop_history[i])
        current_error = get_per_capita_capital_error(per_capita_capital, i)
        i += 1
    return capital_history, pop_history, per_capita_capital, i


# Função de acumulação de capital
def capital_accumulation(s: float, K: float, L: float) -> float:
    return s * math.pow(K, ALPHA) * math.pow(L, 1 - ALPHA) - DELTA * K


# Função de crescimento populacional
def pop_growth(L: float) -> float:
    return L * BETA


# Calcula erro usado como critério de parada
def get_per_capita_capital_error(capital_history: list[float], i: int) -> float:
    return abs(capital_history[i] - capital_history[i - 1]) / abs(capital_history[i])


# Converte taxa de um período para novo período.
# Novo período deve ser em função do antigo.
def convert_rates(old_rate: float, new_period: float) -> float:
    return math.pow(1 + old_rate, new_period) - 1


# Define sistema de Solow para poder ser usado pelo odeint
def solow_system(y, t, a, b, d, s):
    capital, pop = y
    dydt = [
        s * np.power(capital, a) * np.power(pop, 1 - a) - d * capital,
        b * pop
    ]
    return dydt


if __name__ == "__main__":
    main()

