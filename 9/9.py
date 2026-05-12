import matplotlib
import matplotlib.ticker
import matplotlib.pyplot as plot

import numpy

# []x1^2 + []x1 + []x2^2 + []x2 + [] = 0
def f(x1, x2, coefficients):
    return (coefficients[0] * x1**2 + coefficients[1] * x1 +
            coefficients[2] * x2**2 + coefficients[3] * x2 +
            coefficients[4]
            )

# []y^2 + []y + ([]x^2 + []x + []) = 0
def f_y(x, coefficients):
    a = coefficients[2]
    b = coefficients[3]
    c = x**2 * coefficients[0] + x * coefficients[1] + coefficients[4]

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return None, None

    y_one = (-b + numpy.sqrt(discriminant)) / (2 * a)
    y_two = (-b - numpy.sqrt(discriminant)) / (2 * a)
    return y_one, y_two

def rosenbrock_target(x1, x2):
    return 100 * (x2 - x1**2)**2 + (x1 - 1)**2


def hooke_jeeves_method(function_target, x_start):
    q = 2
    p = 2
    e_1 = 10**(-10)
    e_2 = 10**(-10)

    all_x_points = []

    x_0 = x_start.copy()
    x_1 = x_0.copy()

    x_step = numpy.array([0.1, 0.1])

    k = 0
    for k in range(10000):
        x_1 = x_0.copy()
        #x_step = numpy.array([0.1, 0.1])

        for i in range(len(x_1)):
            while True:
                x_1[i] = x_0[i] + x_step[i]
                if function_target(x_1) < function_target(x_0):
                    break

                x_1[i] = x_0[i] - x_step[i]
                if function_target(x_1) < function_target(x_0):
                    break

                x_step[i] /= q
                if x_step[i] < e_1:
                    x_1[i] = x_0[i]
                    break

        all_x_points.append(x_1)
        if numpy.max(numpy.abs(x_1 - x_0)) < e_2:
            break

        while True:
            x_2_initial = x_1 + p * (x_1 - x_0)
            x_2 = x_2_initial.copy()

            for i in range(len(x_2)):
                while True:
                    x_2[i] = x_2_initial[i] + x_step[i]
                    if function_target(x_2) < function_target(x_2_initial):
                        break

                    x_2[i] = x_2_initial[i] - x_step[i]
                    if function_target(x_2) < function_target(x_2_initial):
                        break

                    x_2[i] = x_2_initial[i]
                    break

            x_0 = x_1.copy()
            if function_target(x_2) < function_target(x_1):
                x_1 = x_2.copy()
                continue
            else:
                break

    all_x_points.append(x_1)
    return x_1, k, all_x_points




# []x1^2 + []x1 + []x2^2 + []x2 + [] = 0
coefficients = [
    [-0.3, -5.0, 0.4, -2.5, -1.0],
    [-0.5, -2.1, 0.8, 3.3, 0.5]
]

x1_values = numpy.linspace(-2, 2, 50)
x2_values = numpy.linspace(-3, 3, 50)
f_values = numpy.array([[rosenbrock_target(x1, x2) for x1 in x1_values] for x2 in x2_values])

plot.contourf(x1_values, x2_values, f_values, locator=matplotlib.ticker.LogLocator())
plot.colorbar()

test_x, test_k, test_x_all = hooke_jeeves_method(lambda x: rosenbrock_target(x[0], x[1]), numpy.array([-1.2, 0.0]))
print(f"Given starting point {[-1.2, 0.0]} found point at {test_x} in {test_k} iterations (Rosenberg function)")

plot.plot([x[0] for x in test_x_all], [x[1] for x in test_x_all], marker="o", color="red")

plot.show()

x1_values = numpy.linspace(-2, 2, 50)
x2_values = numpy.linspace(-2, 2, 50)
f_values = numpy.array([[f(x1, x2, coefficients[0])**2 + f(x1, x2, coefficients[1])**2 for x1 in x1_values] for x2 in x2_values])

plot.contourf(x1_values, x2_values, f_values, locator=matplotlib.ticker.LogLocator())
plot.colorbar()

found_x, found_k, found_x_all = hooke_jeeves_method(lambda x: f(x[0], x[1], coefficients[0])**2 + f(x[0], x[1], coefficients[1])**2, numpy.array([-2.0, 1.0]))
print(f"Given starting point {[-2.0, 1.0]} found point at {found_x_all[-1]} in {found_k} iterations (system of equations)")

plot.plot([x[0] for x in found_x_all], [x[1] for x in found_x_all], marker="o", color="red")

plot.show()



x_values = numpy.linspace(-2, 2, 1000)
x_values_one = []
x_values_two = []
y_values_one = []
y_values_two = []
for i in range(len(x_values)):
    x = x_values[i]
    y_one, y_two = f_y(x, coefficients[0])
    if y_one is not None:
        x_values_one.append(x)
        y_values_one.append(y_one)
    if y_two is not None:
        x_values_two.append(x)
        y_values_two.append(y_two)

plot.plot(x_values_one, y_values_one, color="blue")
plot.plot(x_values_two, y_values_two, color="blue")

x_values_one = []
x_values_two = []
y_values_one = []
y_values_two = []
for i in range(len(x_values)):
    x = x_values[i]
    y_one, y_two = f_y(x, coefficients[1])
    if y_one is not None:
        x_values_one.append(x)
        y_values_one.append(y_one)
    if y_two is not None:
        x_values_two.append(x)
        y_values_two.append(y_two)

plot.plot(x_values_one, y_values_one, color="green")
plot.plot(x_values_two, y_values_two, color="green")

plot.plot([x[0] for x in found_x_all], [x[1] for x in found_x_all], marker="o", markersize=4, linewidth=0.5, color="maroon")
plot.plot(found_x_all[-1][0], found_x_all[-1][1], marker="o", color="red")

plot.axhline(0, color="black")
plot.axvline(0, color="black")

plot.show()