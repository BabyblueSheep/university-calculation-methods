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



def get_t(x_values: list[float], i: int) -> float:
    return (x_values[i] - x_values[0]) / (x_values[1] - x_values[0])

def finite_difference(y_values: list[float], i: int, order: int):
    end_value = 0
    for k in range(order + 1):
        end_value += math.pow(-1, order - k) * (math.factorial(order) /(math.factorial(k) * math.factorial(order - k))) * y_values[i + k]
    return end_value

def falling_factorial(t: float, order: int) -> float:
    end_value = 1
    for i in range(order):
        end_value *= (t - i)
    return end_value

def factorial_polynomial(t: float, y_values: list[float]) -> float:
    end_value = 0
    for k in range(len(y_values)):
        end_value += y_values[0] * finite_difference(y_values, i, k) / math.factorial(k) * falling_factorial(t, k)
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

requests_per_second_20_points = [-1] * 21
cpu_power_20_points = [-1] * 21

for i in range(21):
    requests_per_second_20_points[i] = requests_per_second[0] + (requests_per_second[-1] - requests_per_second[0]) * i / 20
#for t in numpy.linspace(0, len(cpu_power), 20):
    #cpu_power_20_points[i] = factorial_polynomial(t, cpu_power)

print(requests_per_second)
print(requests_per_second_20_points)

figure, axes = plot.subplots(nrows=1, ncols=2, tight_layout=True)

axes[0].plot(requests_per_second, cpu_power, color="blue", linestyle="", marker="o")

interpolated_x = numpy.linspace(requests_per_second[0], requests_per_second[-1], 128)
interpolated_y = [newton_interpolation_polynomial(requests_per_second, cpu_power, x) for x in interpolated_x]

axes[0].plot(interpolated_x, interpolated_y, color="red")

plot.show()