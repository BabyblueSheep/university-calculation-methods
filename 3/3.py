import csv

import numpy, matplotlib.pyplot as plot, matplotlib.colors

from typing import TextIO


def form_matrix(x_values: numpy.ndarray, m: int) -> numpy.ndarray:
    matrix = numpy.zeros((m + 1, m + 1))
    for i in range(m + 1):
        for j in range(m + 1):
            matrix[i, j] = numpy.sum(numpy.pow(x_values, i + j))
    return matrix

def form_vector(x_values: numpy.ndarray, y_values: numpy.ndarray, m: int) -> numpy.ndarray:
    vector = numpy.zeros(m + 1)
    for i in range(m + 1):
        vector[i] = numpy.sum(y_values * numpy.pow(x_values, i))
    return vector

def gaussian_method(a_matrix: numpy.ndarray, b_vector: numpy.ndarray) -> numpy.ndarray:
    max_columns = a_matrix.shape[1]
    max_rows = a_matrix.shape[0]

    #triangular_matrix = numpy.zeros(a_matrix.shape)
    #triangular_matrix_vector = numpy.zeros(max_columns)

    for pivot_column in range(max_columns):
        maximum_row = pivot_column
        for pivot_row in range(pivot_column + 1, max_rows):
            if numpy.abs(a_matrix[pivot_row, pivot_column]) > numpy.abs(a_matrix[maximum_row, pivot_column]):
                maximum_row = pivot_row

        for replacing_column in range(max_columns):
            a_matrix[pivot_column, replacing_column], a_matrix[maximum_row, replacing_column] = a_matrix[maximum_row, replacing_column], a_matrix[pivot_column, replacing_column]
        b_vector[pivot_column], b_vector[maximum_row] = b_vector[maximum_row], b_vector[pivot_column]

        for pivot_row in range(pivot_column + 1, max_rows):
            #triangular_matrix[pivot_row, pivot_column] = a_matrix[pivot_column, pivot_column]
            #triangular_matrix_vector[pivot_row] = b_vector[pivot_row]
            factor = a_matrix[pivot_row, pivot_column] / a_matrix[pivot_column, pivot_column]
            a_matrix[pivot_row, pivot_column:] = a_matrix[pivot_row, pivot_column:] - factor * a_matrix[pivot_column, pivot_column:]
            b_vector[pivot_row] = b_vector[pivot_row] - factor * b_vector[pivot_column]

    x_values = numpy.zeros(max_columns)

    for i in range(max_columns - 1, -1, -1):
        summed_values = 0
        for j in range(i + 1, max_columns):
            summed_values += a_matrix[i, j] * x_values[j]
        x_values[i] = (b_vector[i] - summed_values) / a_matrix[i, i]

    return x_values

def calculate_polynomial(x_values: numpy.ndarray, coefficients: numpy.ndarray) -> numpy.ndarray:
    y_values = numpy.zeros(x_values.shape)

    for i in range(len(coefficients)):
        y_values += coefficients[i] * numpy.pow(x_values, i)

    return y_values

def get_variance(true_y_values: numpy.ndarray, approximate_y_values: numpy.ndarray) -> float:
    return numpy.average(numpy.power(true_y_values - approximate_y_values, 2))



def read_key_value_pair_from_csv(file: TextIO, key: str, value: str) -> tuple[list[float], list[float]]:
    keys = []
    values = []

    reader = csv.DictReader(file)
    for row in reader:
        keys.append(float(row[key]))
        values.append(float(row[value]))

    return keys, values



with open("input.csv", 'r', newline='') as file:
    months, temperatures = read_key_value_pair_from_csv(file, "Month", "Temp")

months = numpy.array(months)
temperatures = numpy.array(temperatures)

max_degree = 10
variances = []
for degree in range(max_degree):
    a_matrix = form_matrix(months, degree)
    b_vector = form_vector(months, temperatures, degree)
    coefficients = gaussian_method(a_matrix, b_vector)
    approximate_y_values = calculate_polynomial(months, coefficients)
    variance = get_variance(temperatures, approximate_y_values)
    variances.append(variance)
optimal_m = numpy.argmin(numpy.array(variances)) + 1
optimal_m = optimal_m.astype(int)

figure, axes = plot.subplots(nrows=2, ncols=2, tight_layout=True)

def plot_polynomial_approximate(degree: int, axis: int, color: tuple[float, float, float] | None = None):
    a_matrix = form_matrix(months, degree)
    b_vector = form_vector(months, temperatures, degree)
    coefficients = gaussian_method(a_matrix, b_vector)
    approximate_temperatures = calculate_polynomial(months, coefficients)

    future_months = numpy.array([25, 26, 27])
    future_temperatures = calculate_polynomial(future_months, coefficients)

    error = temperatures - approximate_temperatures

    if color is None:
        color = matplotlib.colors.hsv_to_rgb(((degree / 12) % 1, 1, 1))

    axes[axis, 0].plot(months, approximate_temperatures, color=color, linestyle="--")[0].set_label(f"Приблизні температури (при степені {degree})")

    axes[axis, 1].plot(months, numpy.abs(error), color=color, linestyle="--")[0].set_label(f"Похибка (при степені {degree})")





axes[0, 0].plot(months, temperatures, color="black", linestyle="-", linewidth=2)[0].set_label("Оригінальні температури")
axes[1, 0].plot(months, temperatures, color="black", linestyle="-", linewidth=2)[0].set_label("Оригінальні температури")

plot_polynomial_approximate(optimal_m, 0, (1, 0, 0))
for i in range(1, 10 + 1):
    plot_polynomial_approximate(i, 1)

axes[0, 0].set_xlabel("Місяць (к-сть)")
axes[0, 0].set_ylabel("Температура (°C)")
axes[0, 0].set_title("Оригінальні дані + многочлен з оптимальним степенем")
axes[0, 0].legend()

axes[1, 0].set_xlabel("Місяць (к-сть)")
axes[1, 0].set_ylabel("Температура (°C)")
axes[1, 0].set_title("Оригінальні дані + різні степені")
axes[1, 0].legend()

axes[0, 1].set_xlabel("Місяць (к-сть)")
axes[0, 1].set_ylabel("Температура (°C)")
axes[0, 1].set_title("Похибка многочлена з оптимальним степенем")
axes[0, 1].legend()

axes[1, 1].set_xlabel("Місяць (к-сть)")
axes[1, 1].set_ylabel("Температура (°C)")
axes[1, 1].set_title("Похибки многочлен з різними степенями")
axes[1, 1].legend()

figure.suptitle("Візуалізація я алгебраїчних многочленів найкращого квадратичного наближення методом найменших квадратів")

plot.show()