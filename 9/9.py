import matplotlib
import matplotlib.pyplot as plot

import numpy

def f(x1, x2, coefficients):
    return (coefficients[0] * x1*x1 + coefficients[1] * x1 +
            coefficients[2] * x2*x2 + coefficients[3] * x2 +
            coefficients[4] * (x2 - x1) + coefficients[5] * (x2 - x1)*(x2 - x1) +
            coefficients[6]
            )

def rosenbrock_target(x1, x2):
    return 100 * ((x1**2 - x2)**2) + (x1 - 1)**2


def hooke_jeeves_method(x_start):
    def func(x):
        return rosenbrock_target(x[0], x[1])

    q = 2
    p = 2
    e_1 = 10**(-10)
    e_2 = 10**(-10)

    x_first = x_start
    x_step = numpy.array([0.5, 0.5])

    x_final = x_start

    k = 0
    for k in range(1000):

        i = 0
        while True:
            while True:
                if x_step[i] < e_1:
                    break

                x_first[i] += x_start[i] * x_step[i]

                if func(x_first) >= func(x_start):
                    x_first[i] -= x_start[i] * x_step[i]

                    if func(x_first) >= func(x_start):
                        x_step[i] /= q
                        if numpy.max(x_step) < e_1:
                            x_first[i] = x_start[i]
                        else:
                            continue

            i += 1
            if i >= len(x_start):
                break

        if numpy.linalg.norm(x_step) < e_1 and numpy.abs(func(x_first) - func(x_start)) < e_2:
            x_final = x_first
            break

        x_second = x_first + p * (x_first - x_start)
        x_step = numpy.array([0.5, 0.5])

        while True:
            while True:
                x_second += x_start * x_step

                if func(x_second) >= func(x_first):
                    x_second -= x_start * x_step

                    if func(x_second) >= func(x_first):
                        x_second = x_start

            if func(x_second) < func(x_first):
                x_start = x_first
                x_first = x_second
            else:
                x_start = x_first
                break

    return x_final, k




# []x1^2 + []x1 + []x2^2 + []x2 + [](x2 - x1) + [](x2 - x1)^2 + []
coefficients = [
    [4, 9, 0, 0, -10, 4, 5],
    [5, 0, -6, -1, 3, 0, 12]
]

x1_values = numpy.linspace(-2, 2, 50)
x2_values = numpy.linspace(-3, 3, 50)
f_values = numpy.array([[rosenbrock_target(x1, x2) for x1 in x1_values] for x2 in x2_values])

#plot.contourf(x1_values, x2_values, f_values, levels=50)
#plot.colorbar()

#plot.show()


test_x, test_k = hooke_jeeves_method(numpy.array([-1.2, 0]))
print(test_x, test_k)