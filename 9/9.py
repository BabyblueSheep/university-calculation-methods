import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plot

import numpy

def f(x1, x2, coefficients):
    return (coefficients[0] * x1*x1 + coefficients[1] * x1 +
            coefficients[2] * x2*x2 + coefficients[3] * x2 +
            coefficients[4] * (x2 - x1) + coefficients[5] * (x2 - x1)*(x2 - x1) +
            coefficients[6]
            )

def rosenbrock_target(x1, x2):
    return 100 * (x2 - x1**2)**2 + (x1 - 1)**2


def hooke_jeeves_method(function_target, x_start):
    q = 2
    p = 2
    e_1 = 10**(-10)
    e_2 = 10**(-10)

    x_step = numpy.array([0.5, 0.5])
    x_first = x_start.copy()

    k = 0
    for k in range(1000):
        x_first = x_start.copy()

        for i in range(len(x_start)):
            while True:
                x_first[i] = x_start[i] + x_step[i]
                if function_target(x_first) < function_target(x_start):
                    break

                x_first[i] = x_start[i] - x_step[i]
                if function_target(x_first) < function_target(x_start):
                    break

                x_step[i] /= q
                if x_step[i] < e_1:
                    x_first[i] = x_start[i]
                    break

        if numpy.max(numpy.abs(x_first - x_start)) < e_2:
            break

        x_second = x_first.copy()
        x_first = x_start.copy()

        while True:
            x_second_start = x_second + p * (x_second - x_first)
            x_second_probe = x_second_start.copy()

            for i in range(len(x_second_probe)):
                while True:
                    x_second_probe[i] = x_second_start[i] + x_step[i]
                    if function_target(x_second_probe) < function_target(x_second_start):
                        break

                    x_second_probe[i] = x_second_start[i] - x_step[i]
                    if function_target(x_second_probe) < function_target(x_second_start):
                        break

                    x_second_probe[i] = x_second_start[i]
                    break

            if numpy.max(numpy.abs(x_second_probe - x_second_start)) < e_2:
                break
            if function_target(x_second_probe) >= function_target(x_second):
                break

            x_first = x_second.copy()
            x_second = x_second_probe.copy()

        if function_target(x_second) < function_target(x_first):
            x_start = x_second.copy()
        else:
            x_start = x_first.copy()

    return x_start, k




# []x1^2 + []x1 + []x2^2 + []x2 + [](x2 - x1) + [](x2 - x1)^2 + []
coefficients = [
    [-4, -4, 0, -5, 0, 5, 3],
    [0, 9, -5, 4, -5, 0, 1]
]

x1_values = numpy.linspace(-2, 2, 50)
x2_values = numpy.linspace(-3, 3, 50)
f_values = numpy.array([[rosenbrock_target(x1, x2) for x1 in x1_values] for x2 in x2_values])

plot.contourf(x1_values, x2_values, f_values, locator=matplotlib.ticker.LogLocator())
plot.colorbar()

plot.show()

test_x, test_k = hooke_jeeves_method(lambda x: rosenbrock_target(x[0], x[1]), numpy.array([-1.2, 0.0]))
print(f"Given starting point {[-1.2, 0.0]} found point at {test_x} in {test_k} iterations (Rosenberg function)")

x1_values = numpy.linspace(-2, 2, 50)
x2_values = numpy.linspace(-2, 2, 50)
f_values = numpy.array([[f(x1, x2, coefficients[0])**2 + f(x1, x2, coefficients[1])**2 for x1 in x1_values] for x2 in x2_values])

plot.contourf(x1_values, x2_values, f_values, locator=matplotlib.ticker.LogLocator())
plot.colorbar()

plot.show()

found_x, found_k = hooke_jeeves_method(lambda x: f(x[0], x[1], coefficients[0])**2 + f(x[0], x[1], coefficients[1])**2, numpy.array([0.0, 0.0]))
print(f"Given starting point {[0.0, 0.0]} found point at {found_x} in {found_k} iterations (system of equations)")