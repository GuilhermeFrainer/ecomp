# Arquivos externos
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np
import math
from tabulate import tabulate
import decimal # Usado para garantir precisão

# Arquivos internos
import pi_formatter
import helper_sin as hs


X = math.pi / 4
N_TEST_CASES = [5, 10, 100]
TARGET_VALUE = decimal.Decimal(math.sqrt(2))

PRECISION = 100 # Precisão usada para calcular erro


def main():
    # Determina precisão em PRECISION casas decimais
    decimal.setcontext(decimal.Context(prec=PRECISION))
    
    results = [2 * taylor_series_sin_x(decimal.Decimal(X), test_case) for test_case in N_TEST_CASES]
    errors = [get_relative_error(result, TARGET_VALUE) for result in results]
    sqrt_of_two = [math.sqrt(2) for _ in range(3)]
    
    # Resposta (a)
    print("Resposta (a) com 10 algarismos significativos de precisão")
    print(tabulate({
        "n": N_TEST_CASES,
        "Valor": sqrt_of_two,
        "Aproximação": results,
        "Erro (%)": errors
        },
        headers="keys",
        floatfmt=".10g"
    ))
    
    # Resposta (b)
    x_axis = np.arange(-np.pi, np.pi, 0.01)
    x_axis = np.array([decimal.Decimal(x) for x in x_axis])

    vec_sin = np.vectorize(hs.sin)

    # Popula arrays com aproximações feitas pelo polinômio de Taylor
    p5_values = np.array([taylor_series_sin_x(x, N_TEST_CASES[0]) for x in x_axis], dtype=decimal.Decimal)
    p10_values = np.array([taylor_series_sin_x(x, N_TEST_CASES[1]) for x in x_axis], dtype=decimal.Decimal)
    p100_values = np.array([taylor_series_sin_x(x, N_TEST_CASES[2]) for x in x_axis], dtype=decimal.Decimal)
    
    figure = plt.figure("Exercício 1b")
    axis = figure.add_subplot(111)

    axis.plot(x_axis, vec_sin(x_axis), '#000000', label="$sen(x)$")
    axis.plot(x_axis, p5_values, '#ff0000', label="$P_{5}(x)$")
    axis.plot(x_axis, p10_values, '#00ff00', label="$P_{10}(x)$")
    axis.plot(x_axis, p100_values, '#0000ff', label="$P_{100}(x)$")

    # Dá título aos eixos e ao gráfico
    axis.set_title("Polinômio de Taylor que aproxima sen(x)")
    axis.set_xlabel("x")
    axis.set_ylabel("f (x)")

    axis.xaxis.set_major_locator(tck.MultipleLocator(np.pi / 2))
    axis.xaxis.set_minor_locator(tck.MultipleLocator(np.pi / 4))
    axis.xaxis.set_major_formatter(plt.FuncFormatter(pi_formatter.multiple_formatter()))

    # Coloca legenda
    axis.legend()

    figure.savefig("Exercício 1b.png")

    # Resposta (c)

    figure = plt.figure("Exercício 1c")
    axis = figure.add_subplot(111)

    # Popula array com valores de seno
    sin_of_x = np.array([decimal.Decimal(hs.sin(x)) for x in x_axis])

    axis.plot(
        x_axis,
        abs(sin_of_x - p5_values),
        "#ff0000",
        label="$E_{5}$"
    )
    axis.plot(
        x_axis,
        abs(sin_of_x - p10_values),
        "#00ff00",
        label="$E_{10}$"
    )
    axis.plot(
        x_axis,
        abs(sin_of_x - p100_values),
        "#0000ff",
        label="$E_{100}$"
    )

    # Dá título aos eixos e ao gráfico
    axis.set_title("Erros nas estimativas de sen(x)")
    axis.set_xlabel("x")
    axis.set_ylabel("E (x)")

    axis.xaxis.set_major_locator(tck.MultipleLocator(np.pi / 2))
    axis.xaxis.set_minor_locator(tck.MultipleLocator(np.pi / 4))
    axis.xaxis.set_major_formatter(plt.FuncFormatter(pi_formatter.multiple_formatter()))

    # Coloca eixo y em escala logarítmica
    axis.set_yscale("log")
    
    # Coloca legenda
    axis.legend()

    figure.savefig("Exercício 1c.png")


# Calcula o polinômio de Taylor que aproxima sen(x) de grau n no ponto x
def taylor_series_sin_x(x: decimal.Decimal, n: int) -> decimal.Decimal:
    return sum([decimal.Decimal(math.pow(-1, k) * math.pow(x, 2 * k + 1)) / math.factorial(2 * k + 1) for k in range(n + 1)])


# Calcula o erro relativo percentual
def get_relative_error(estimate: decimal.Decimal, real_value: decimal.Decimal) -> decimal.Decimal:
    return (real_value - estimate) / real_value * 100


if __name__ == "__main__":
    main()

