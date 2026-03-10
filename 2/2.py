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

figure, axes = plot.subplots(nrows=1, ncols=2, tight_layout=True)

for i in range(2):
    axes[i].set_xlabel("Запити за секунду")
    axes[i].set_ylabel("Використання CPU")

requests_per_second_granular = numpy.linspace(requests_per_second[0], requests_per_second[-1], 128)


newton_coefficients = newton_divided_differences_coefficients(requests_per_second, cpu_power)
cpu_power_newton_polynomial = [newton_interpolation_polynomial_coefficients(x, requests_per_second, newton_coefficients) for x in requests_per_second_granular]

axes[0].plot(requests_per_second_granular, cpu_power_newton_polynomial, color="red", linestyle="-")[0].set_label("Інтерполяційний многочлен Ньютона")
axes[0].plot(requests_per_second, cpu_power, color="blue", linestyle="", marker="o")[0].set_label("Дані точки")

axes[0].set_title("Інтерполяційний многочлен Ньютона")
axes[0].legend()



cpu_power_newton_factorial = [factorial_polynomial(x, cpu_power, len(cpu_power)) for x in numpy.linspace(0, len(cpu_power), 128)]

axes[1].plot(requests_per_second, cpu_power, color="black", linestyle="-")
axes[1].plot(requests_per_second, cpu_power, color="black", linestyle="", marker="o")[0].set_label("Дані точки")

for order in range(len(cpu_power)):
    cpu_power_factorial = [factorial_polynomial(i, cpu_power, order + 1) for i in range(len(cpu_power))]

    color = matplotlib.colors.hsv_to_rgb((order / 8, 1, 1))

    axes[1].plot(requests_per_second, cpu_power_factorial, color=color, linestyle="--")
    axes[1].plot(requests_per_second, cpu_power_factorial, color=color, linestyle="", marker="o")[0].set_label(f"Інтерпольовані точки (порядок {order})")

axes[1].set_title("Факторіальні многочлени")
axes[1].legend()

plot.show()


figure, axes = plot.subplots(nrows=2, ncols=2, tight_layout=True)

for i in range(2):
    for j in range(2):
        if i == 0:
            axes[i, j].set_xlabel("Запити за секунду")
        else:
            axes[i, j].set_xlabel("Запити за секунду (нормалізовані)")
        axes[i, j].set_ylabel("Використання CPU")


def plot_points_newton(x_values: list[float], y_values: list[float], axis:int):
    points_color = matplotlib.colors.hsv_to_rgb((len(x_values) / 100, 1, 1))

    x_values_granular = []
    y_values_granular = []

    for i in range(len(x_values) - 1):
        current_x = x_values[i]
        next_x = x_values[i + 1]

        current_y = y_values[i]
        next_y = y_values[i + 1]

        for step in range(20):
            percentage = step / 20
            x_values_granular.append(current_x * (1 - percentage) + next_x * percentage)
            y_values_granular.append(current_y * (1 - percentage) + next_y * percentage)
    x_values_granular.append(x_values[-1])
    y_values_granular.append(y_values[-1])

    points_newton_coefficients = newton_divided_differences_coefficients(x_values, y_values)
    interpolated_y_values = [
        newton_interpolation_polynomial_coefficients(x, x_values, points_newton_coefficients)
        for x in x_values_granular
    ]

    axes[axis, 0].plot(x_values_granular, interpolated_y_values, color=points_color, linestyle="--")[0].set_label(f"{len(x_values)} точок")
    axes[axis, 0].plot(x_values, y_values, color=points_color, linestyle="", marker="o")

    axes[axis, 1].plot(x_values_granular, numpy.abs(numpy.array(interpolated_y_values) - numpy.array(y_values_granular)), color=points_color, linestyle="--")[0].set_label(f"{len(x_values)} точок")

requests_per_second_10_points = []
cpu_power_10_points = []
for i in range(len(requests_per_second) - 1):
    requests_per_second_10_points.append(requests_per_second[i])
    requests_per_second_10_points.append((requests_per_second[i] + requests_per_second[i + 1]) / 2)
    cpu_power_10_points.append(cpu_power[i])
    cpu_power_10_points.append((cpu_power[i] + cpu_power[i + 1]) / 2)
requests_per_second_10_points.append(requests_per_second[-1])
cpu_power_10_points.append(cpu_power[-1])

requests_per_second_20_points = []
cpu_power_20_points = []
for i in range(len(requests_per_second_10_points) - 1):
    requests_per_second_20_points.append(requests_per_second_10_points[i])
    requests_per_second_20_points.append((requests_per_second_10_points[i] + requests_per_second_10_points[i + 1]) / 2)
    cpu_power_20_points.append(cpu_power_10_points[i])
    cpu_power_20_points.append((cpu_power_10_points[i] + cpu_power_10_points[i + 1]) / 2)
requests_per_second_20_points.append(requests_per_second_10_points[-1])
cpu_power_20_points.append(cpu_power_10_points[-1])

plot_points_newton(x_values=requests_per_second_20_points, y_values=cpu_power_20_points, axis=1)
plot_points_newton(x_values=requests_per_second, y_values=cpu_power, axis=1)
plot_points_newton(x_values=requests_per_second_10_points, y_values=cpu_power_10_points, axis=0)
plot_points_newton(x_values=requests_per_second, y_values=cpu_power, axis=0)


axes[0, 0].set_title("Інтерполяційні многочлени Ньютона із різними к-стями точок (5, 10)")
axes[0, 0].legend()
axes[0, 1].set_title("Похибки інтерполяційних многочленів Ньютона із різними к-стями точок (5, 10)")
axes[0, 1].legend()
axes[1, 0].set_title("Інтерполяційні многочлени Ньютона із різними к-стями точок (5, 20)")
axes[1, 0].legend()
axes[1, 1].set_title("Похибки інтерполяційних многочленів Ньютона із різними к-стями точок (5, 20)")
axes[1, 1].legend()

plot.show()


figure, axes = plot.subplots(nrows=2, ncols=2, tight_layout=True)

for i in range(2):
    for j in range(2):
        if i == 0:
            axes[i, j].set_xlabel("Запити за секунду")
        else:
            axes[i, j].set_xlabel("Запити за секунду (нормалізовані)")
        axes[i, j].set_ylabel("Використання CPU")


def plot_points_factorial(x_values: list[float], y_values: list[float], axis:int):
    points_color = matplotlib.colors.hsv_to_rgb((len(x_values) / 100, 1, 1))

    x_values_granular = []
    y_values_granular = []

    for i in range(len(x_values) - 1):
        current_x = x_values[i]
        next_x = x_values[i + 1]

        current_y = y_values[i]
        next_y = y_values[i + 1]

        for step in range(20):
            percentage = step / 20
            x_values_granular.append(current_x * (1 - percentage) + next_x * percentage)
            y_values_granular.append(current_y * (1 - percentage) + next_y * percentage)
    x_values_granular.append(x_values[-1])
    y_values_granular.append(y_values[-1])

    interpolated_y_values = [
        factorial_polynomial(x, y_values, len(y_values))
        for x in numpy.linspace(0, len(y_values) - 1, len(y_values_granular))
    ]

    axes[axis, 0].plot(x_values_granular, interpolated_y_values, color=points_color, linestyle="--")[0].set_label(f"{len(x_values)} точок")
    axes[axis, 0].plot(x_values, y_values, color=points_color, linestyle="", marker="o")

    axes[axis, 1].plot(x_values_granular, numpy.abs(numpy.array(interpolated_y_values) - numpy.array(y_values_granular)), color=points_color, linestyle="--")[0].set_label(f"{len(x_values)} точок")


plot_points_factorial(x_values=requests_per_second_20_points, y_values=cpu_power_20_points, axis=1)
plot_points_factorial(x_values=requests_per_second, y_values=cpu_power, axis=1)
plot_points_factorial(x_values=requests_per_second_10_points, y_values=cpu_power_10_points, axis=0)
plot_points_factorial(x_values=requests_per_second, y_values=cpu_power, axis=0)


axes[0, 0].set_title("Інтерполяційні многочлени Ньютона із різними к-стями точок (5, 10)")
axes[0, 0].legend()
axes[0, 1].set_title("Похибки інтерполяційних многочленів Ньютона із різними к-стями точок (5, 10)")
axes[0, 1].legend()
axes[1, 0].set_title("Інтерполяційні многочлени Ньютона із різними к-стями точок (5, 20)")
axes[1, 0].legend()
axes[1, 1].set_title("Похибки інтерполяційних многочленів Ньютона із різними к-стями точок (5, 20)")
axes[1, 1].legend()

plot.show()