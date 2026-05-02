import numpy
import math
import matplotlib.pyplot as plot

def f(x, coefficients):
    result = 0
    for i in range(len(coefficients)):
        result += coefficients[i] * x**i
    return result


def horners_approximation(x_start: float, coefficients: list[float], epsilon: float):
    x = x_start
    k = 0
    for k in range(1000):
        x_old = x

        b_coeff = [-1.0 for _ in range(len(coefficients))]
        b_coeff[-1] = coefficients[-1]
        for i in range(len(coefficients) - 2, -1, -1):
            item_to_add = coefficients[i] + x_old * b_coeff[i + 1]
            b_coeff[i] = item_to_add

        c_coeff = [-1.0 for _ in range(len(coefficients))]
        c_coeff[-1] = b_coeff[-1]
        for i in range(len(b_coeff) - 2, -1, -1):
            item_to_add = b_coeff[i] + x_old * c_coeff[i + 1]
            c_coeff[i] = item_to_add

        x -= b_coeff[0] / c_coeff[1]

        if abs(f(x, coefficients)) < epsilon:
            break
        if abs(x - x_old) < epsilon:
            break

    return x, k

def lins_approximation(r_start: float, i_start: float, coefficients: list[float], epsilon: float):
    r = r_start
    i = i_start
    k = 0
    for k in range(1000):
        r_old = r
        i_old = i

        p_old = -2 * r_old
        q_old = r_old * r_old + i_old * i_old

        b_coeff = [-1.0 for _ in range(len(coefficients))]
        b_coeff[-1] = coefficients[-1]
        b_coeff[-2] = coefficients[-2] - p_old * b_coeff[-1]
        for x in range(len(coefficients) - 3, -1, -1):
            item_to_add = coefficients[x] - p_old * b_coeff[x + 1] - q_old * b_coeff[x + 2]
            b_coeff[x] = item_to_add

        q = coefficients[0] / b_coeff[2]
        p = (coefficients[1] * b_coeff[2] - coefficients[0] * b_coeff[3]) / (b_coeff[2] * b_coeff[2])

        r = -p / 2
        i = math.sqrt(abs(q - r*r))

        if abs(r - r_old) < epsilon:
            break
        if abs(i - i_old) < epsilon:
            break

    return r, i, k



starting_coefficients = [-6, 4, -3, 2]

tabulation_start = -5
tabulation_end = 5
step = 0.1

points = []
for x in numpy.arange(tabulation_start, tabulation_end, step):
    points.append((x, f(x, starting_coefficients)))

plot.plot([item[0] for item in points], [item[1] for item in points])
plot.plot([tabulation_start, tabulation_end], [0, 0])
plot.show()

horners_result = horners_approximation(2, starting_coefficients, 10**(-10))
print(f"Approximated y=0, got x={horners_result[0]} in {horners_result[1]} steps; Horner's")

lins_result = lins_approximation(20, 20, starting_coefficients, 10**(-10))
print(f"Approximated y=0, got x={lins_result[0]} +- {lins_result[1]} i in {lins_result[2]} steps; Lin's")