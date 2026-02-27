import csv

import numpy

from typing import TextIO


def newton_polynomial(x_values: list[float], x: float, n: int, i_exclude: int | None = None):
    end_value = 1
    n = numpy.min(n, len(x_values))
    for i in range(n):
        if i_exclude is None or i != i_exclude:
            end_value *= (x - x_values[i])
    return end_value

def newton_divided_differences(x_values: list[float], y_values: list[float], n: int):
    end_value = 0
    n = numpy.min(n, len(x_values))
    for i in range(n):
        end_value += y_values[i] / newton_polynomial(x_values, x_values[i], n, i)

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

