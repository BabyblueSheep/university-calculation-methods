import csv

import numpy, math, matplotlib.pyplot as plot

from typing import TextIO


def newton_basis_polynomial(x_values: list[float], x: float, n: int) -> float:
    end_value = 1
    for i in range(n):
        end_value *= (x - x_values[i])
    return end_value

def newton_divided_differences(x_values: list[float], y_values: list[float], n: int) -> float:
    end_value = 0
    for i in range(n + 1):
        denominator = 1
        for j in range(n + 1):
            if i != j:
                denominator *= (x_values[i] - x_values[j])
        end_value += y_values[i] / denominator
    return end_value

def newton_interpolation_polynomial(x_values: list[float], y_values: list[float], x: float) -> float:
    end_value = y_values[0]
    for i in range(1, len(x_values)):
        end_value += newton_basis_polynomial(x_values, x, i) * newton_divided_differences(x_values, y_values, i)
    return end_value



def finite_difference(y_values: list[float], i: int, order: int):
    if order == 0:
        return y_values[i]
    return finite_difference(y_values, i + 1, order - 1) - finite_difference(y_values, i, order - 1)

def falling_factorial(t: float, order: int) -> float:
    end_value = 1
    for i in range(order):
        end_value *= (t - i)
    return end_value

def factorial_polynomial(t: float, y_values: list[float], n: int) -> float:
    end_value = 0
    for k in range(n):
        end_value += finite_difference(y_values, 0, k) / math.factorial(k) * falling_factorial(t, k)
    return end_value




def read_key_value_pair_from_csv(file: TextIO, key: str, value: str) -> tuple[list[float], list[float]]:
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

print(f"Відсоток використання CPU при RPS {600}: {newton_interpolation_polynomial(requests_per_second, cpu_power, 600)}%")

cpu_power_factorial = [factorial_polynomial(i, cpu_power, 5) for i in range(len(cpu_power))]

figure, axes = plot.subplots(nrows=1, ncols=2, tight_layout=True)

axes[0].plot(requests_per_second, cpu_power, color="blue", linestyle="", marker="o")
axes[1].plot(requests_per_second, cpu_power, color="blue", linestyle="", marker="o")

interpolated_x = numpy.linspace(requests_per_second[0], requests_per_second[-1], 128)
interpolated_y = [newton_interpolation_polynomial(requests_per_second, cpu_power, x) for x in interpolated_x]

axes[0].plot(interpolated_x, interpolated_y, color="red")

axes[1].plot(requests_per_second, cpu_power_factorial, color="red")

plot.show()