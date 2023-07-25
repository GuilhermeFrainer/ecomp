# Otimização das funções de Ackley e Rosenbrock
# Resolução do caso de duas variáveis
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt


MAX_ERROR = 10 ** -6
MAX_ITERATIONS = 1000
DERIVATIVE_FACTOR= 10 ** -5 # Fator utilizado para estimar vetor gradiente


def main():
    # Cria gráfico para função de Ackley
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    y, x = np.meshgrid(y, x)

    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    surf = ax1.plot_trisurf(x.flat, y.flat, ackley(np.array([x, y])).flat, cmap='coolwarm')

    # Cria gráfico para função de Rosenbrock
    x = np.linspace(-35, 35, 70)
    y = np.linspace(-35, 35, 70)
    y, x = np.meshgrid(y, x)

    fig, ax = plt.subplots(21, subplot_kw={'projection': '3d'})
    surf = ax2.plot_trisurf(x.flat, y.flat, rosenbrock(np.array([x, y])).flat, cmap='coolwarm')
    
    plt.show()
    
    # Encontra ponto de mínimo na função de Ackley pelo Scipy
    initial_guess = [5, 5]
    ac_opt_result = opt.minimize(ackley, initial_guess)
    print("Resultado Scipy função de Ackley")
    print(f"x: {ac_opt_result.x}\nf: {ac_opt_result.fun}")


    # Encontra ponto de mínimo na função de Rosenbrock pelo Scipy
    initial_guess = [10, 10]
    rb_opt_result = opt.minimize(rosenbrock, initial_guess)
    print("Resultado Scipy função de Rosenbrock")
    print(f"x: {rb_opt_result.x}\nf: {rb_opt_result.fun}")
    return


# Função de ackley para duas variáveis.
def ackley(x: np.ndarray):
    return_value = -20 * np.exp(-0.02 * np.sqrt(0.5 * (np.power(x[0], 2)) + np.power(x[1], 2)))
    return_value -= np.exp(0.5 * (np.cos(2 * np.pi * x[0]) + np.cos(2 * np.pi * x[1])))
    return_value += 20 + np.e
    return return_value


# Função de Rosenbrock para n variáveis
def rosenbrock(x: np.ndarray):
    sum = 0
    for i in range(len(x) - 1):
        sum += 100 * np.power((x[i + 1] - np.power(x[i], 2)), 2) + np.power((x[i] - 1), 2)
    return sum


# Minimiza a função pelo método do vetor gradiente.
# Recebe a função e array com estimativa dos argumentos.
def minimize_by_gradient(func, x0: np.ndarray) -> np.ndarray:
    error = 1
    i = 0
    x = [x0]
    while error > MAX_ERROR and i < MAX_ITERATIONS:
        min_func = lambda alpha: ...
        
        i += 1


# Retorna vetor gradiente no ponto x para a função
def get_gradient(func, x: np.ndarray) -> np.ndarray:
    func_base_value = func(x)
    gradient = []
    for i in range(len(x)):
        temp = x.copy()
        temp[i] += DERIVATIVE_FACTOR
        func_temp_value = func(temp)
        derivative = (func_temp_value - func_base_value) / DERIVATIVE_FACTOR
        gradient.append(derivative)

    return np.array(gradient)


if __name__ == "__main__":
    main()

