# Otimização das funções de Ackley e Rosenbrock
# Resolução do caso de duas variáveis
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt


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
    

if __name__ == "__main__":
    main()

