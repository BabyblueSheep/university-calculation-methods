import csv

import numpy, math, matplotlib.pyplot as plot, matplotlib.colors

from typing import TextIO


def newton_basis_polynomial(x: float, x_values: list[float], order: int) -> float:
    """Повертає (x - x0) * (x - x1) * (x - x2) * ... * (x - xn) де order є n."""
    end_value = 1
    for i in range(order):
        end_value *= (x - x_values[i])
    return end_value

def newton_divided_differences(x_values: list[float], y_values: list[float], order: int) -> float:
    """Повертає значення розділеної різниці довільного порядку order."""
    end_value = 0
    for i in range(order + 1):
        denominator = 1
        for j in range(order + 1):
            if i != j:
                denominator *= (x_values[i] - x_values[j])
        end_value += y_values[i] / denominator
    return end_value

def newton_divided_differences_coefficients(x_values: list[float], y_values: list[float]) -> list[float]:
    """Повертає коефіцієнти для многочлена Ньютона"""
    coefficients = [0] * len(y_values)
    for i in range(len(y_values)):
        coefficients[i] = newton_divided_differences(x_values, y_values, i)
    return coefficients

def newton_interpolation_polynomial(x: float, x_values: list[float], y_values: list[float]) -> float:
    """Повертає значення інтерполяційного многочлена Ньютона."""
    end_value = y_values[0]
    for i in range(1, len(x_values)):
        end_value += newton_basis_polynomial(x, x_values, i) * newton_divided_differences(x_values, y_values, i)
    return end_value

def newton_interpolation_polynomial_coefficients(x: float, x_values: list[float], coefficients: list[float]) -> float:
    """Повертає значення інтерполяційного многочлена Ньютона."""
    end_value = coefficients[0]
    for i in range(1, len(x_values)):
        end_value += newton_basis_polynomial(x, x_values, i) * coefficients[i]
    return end_value



def finite_difference(y_values: list[float], i: int, order: int):
    """Повертає результат правого різницевого оператора із степенем."""
    #if order == 0:
    #    return y_values[i]
    #return finite_difference(y_values, i + 1, order - 1) - finite_difference(y_values, i, order - 1)
    values = y_values[i:i + order + 1]

    for current_order in range(order):
        values = [values[j + 1] - values[j] for j in range(len(values) - 1)]

    return values[0]

def falling_factorial(t: float, order: int) -> float:
    """Повертає результат спадаючого факторіала."""
    end_value = 1
    for i in range(order):
        end_value *= (t - i)
    return end_value

def factorial_polynomial(t: float, y_values: list[float], n: int) -> float:
    """Повертає результат факторіального многочлена."""
    end_value = 0
    for k in range(n):
        end_value += finite_difference(y_values, 0, k) / math.factorial(k) * falling_factorial(t, k)
    return end_value




def read_key_value_pair_from_csv(file: TextIO, key: str, value: str) -> tuple[list[float], list[float]]:
    """Читає файл CSV, що складається із двох стовпчиків із назвами key та value, і повертає масиви із значеннями у стовпчиків."""
    keys = []
    values = []

    reader = csv.DictReader(file)
    for row in reader:
        keys.append(float(row[key]))
        values.append(float(row[value]))

    return keys, values

with open("input.csv", 'r', newline='') as file:
    requests_per_second, cpu_power = read_key_value_pair_from_csv(file, "rps", "cpu")

for i in range(len(requests_per_second)):
    print(f"Відсоток використання CPU при RPS {requests_per_second[i]}: {cpu_power[i]}%")

print(f"Відсоток використання CPU при RPS {600} (многочлен Ньютона): {newton_interpolation_polynomial(600, requests_per_second, cpu_power)}%")
print(f"Відсоток використання CPU при RPS {600} (факторіальний многочлен): {factorial_polynomial(3.5, cpu_power, len(cpu_power))}%")

figure, axes = plot.subplots(nrows=3, ncols=2, tight_layout=True)

for i in range(3):
    for j in range(2):
        if i == 0:
            axes[i, j].set_xlabel("Запити за секунду")
        else:
            axes[i, j].set_xlabel("Запити за секунду (нормалізовані)")
        axes[i, j].set_ylabel("Використання CPU")

requests_per_second_granular = numpy.linspace(requests_per_second[0], requests_per_second[-1], 128)


newton_coefficients = newton_divided_differences_coefficients(requests_per_second, cpu_power)
cpu_power_newton_polynomial = [newton_interpolation_polynomial_coefficients(x, requests_per_second, newton_coefficients) for x in requests_per_second_granular]

axes[0, 0].plot(requests_per_second_granular, cpu_power_newton_polynomial, color="red", linestyle="-")[0].set_label("Інтерполяційний многочлен Ньютона")
axes[0, 0].plot(requests_per_second, cpu_power, color="blue", linestyle="", marker="o")[0].set_label("Дані точки")

axes[0, 0].set_title("Інтерполяційний многочлен Ньютона")
axes[0, 0].legend()



cpu_power_newton_factorial = [factorial_polynomial(x, cpu_power, len(cpu_power)) for x in numpy.linspace(0, len(cpu_power), 128)]

axes[0, 1].plot(requests_per_second, cpu_power, color="black", linestyle="-")
axes[0, 1].plot(requests_per_second, cpu_power, color="black", linestyle="", marker="o")[0].set_label("Дані точки")

for order in range(len(cpu_power)):
    cpu_power_factorial = [factorial_polynomial(i, cpu_power, order + 1) for i in range(len(cpu_power))]

    color = matplotlib.colors.hsv_to_rgb((order / 8, 1, 1))

    axes[0, 1].plot(requests_per_second, cpu_power_factorial, color=color, linestyle="--")
    axes[0, 1].plot(requests_per_second, cpu_power_factorial, color=color, linestyle="", marker="o")[0].set_label(f"Інтерпольовані точки (порядок {order})")

axes[0, 1].set_title("Факторіальні многочлени")
axes[0, 1].legend()



x_values_granular = numpy.linspace(0, 1, 128)
y_values_real = numpy.array([numpy.sin(x * numpy.pi * 5) for x in x_values_granular])

axes[1, 0].plot(x_values_granular, y_values_real, color="black", linestyle="-")[0].set_label("Дані точки")
axes[2, 0].plot(x_values_granular, y_values_real, color="black", linestyle="-")[0].set_label("Дані точки")

def plot_points(point_amount: int):
    color = matplotlib.colors.hsv_to_rgb((point_amount / 100, 1, 1))

    x_values = numpy.linspace(0, 1, point_amount)
    y_values = numpy.sin(numpy.linspace(0, 1, point_amount) * numpy.pi * 5)

    point_newton_coefficients = newton_divided_differences_coefficients(x_values.tolist(), y_values.tolist())
    y_values_polynomial = [newton_interpolation_polynomial_coefficients(x, x_values.tolist(), point_newton_coefficients) for x in x_values_granular]

    axes[1, 0].plot(x_values_granular, y_values_polynomial, color=color, linestyle="--")
    axes[1, 0].plot(x_values, y_values, color=color, linestyle="", marker="o")[0].set_label(f"Точки ({point_amount} точок)")

    axes[1, 1].plot(x_values_granular, numpy.abs(y_values_real - numpy.array(y_values_polynomial)), color=color, linestyle="-")[0].set_label(f"Похибка ({point_amount} точок)")

    y_values_factorial = [factorial_polynomial(i, y_values.tolist(), len(y_values)) for i in numpy.linspace(0, len(y_values) - 1, 128)]

    axes[2, 0].plot(x_values_granular, y_values_factorial, color=color, linestyle="--")
    axes[2, 0].plot(x_values, y_values, color=color, linestyle="", marker="o")[0].set_label(f"Точки ({point_amount} точок)")

    axes[2, 1].plot(x_values_granular, numpy.abs(y_values_real - numpy.array(y_values_factorial)), color=color, linestyle="-")[0].set_label(f"Похибка ({point_amount} точок)")

plot_points(20)
plot_points(10)
plot_points(5)

axes[1, 0].set_title("Інтерполяційні многочлени Ньютона із різними к-стями точок")
axes[1, 0].legend()
axes[1, 1].set_title("Похибки інтерполяційних многочленів Ньютона із різними к-стями точок")
axes[1, 1].legend()
axes[2, 0].set_title("Факторіальні многочлени із різними к-стями точок")
axes[2, 0].legend()
axes[2, 1].set_title("Похибки факторіальних многочленів із різними к-стями точок")
axes[2, 1].legend()

plot.show()
