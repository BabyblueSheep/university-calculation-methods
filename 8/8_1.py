import numpy
import math
import matplotlib.pyplot as plot

def f(x):
    return math.exp(x) * math.sin(x) + math.log(1 + x**2) - 2

def fd(x):
    return 2 * x / (1 + x**2) + math.exp(x) * math.sin(x) + math.exp(x) * math.cos(x)

def fdd(x):
    return 2 * (-x**2 + math.exp(x) * (1 + x**2)**2 * math.cos(x) + 1) / ((1 + x**2)**2)


def inverse_slope(x_current, x_previous):
    return (f(x_current) - f(x_previous)) / (x_current - x_previous)

def convergence_condition(x, x_old, epsilon):
    return (abs(f(x)) < epsilon) and (abs(x - x_old)) < epsilon


def simple_approximation(x_start: float, epsilon: float):
    x = x_start
    tau = -0.5 / fd(x_start)
    k = 0
    for k in range(1000):
        x_old = x
        x += tau * f(x_old)

        if convergence_condition(x, x_old, epsilon):
            break
    return x, k

def newtons_approximation(x_start: float, epsilon: float):
    x = x_start
    k = 0
    for k in range(1000):
        x_old = x
        x -= f(x_old) / fd(x_old)

        if convergence_condition(x, x_old, epsilon):
            break
    return x, k

def chebyshev_approximation(x_start: float, epsilon: float):
    x = x_start
    k = 0
    for k in range(1000):
        x_old = x
        x -= f(x_old) / fd(x_old) + 0.5 * f(x_old)**2 * fdd(x_old) / (fd(x_old)**3)

        if convergence_condition(x, x_old, epsilon):
            break
    return x, k

def secant_approximation(x_start_first: float, x_start_second: float, epsilon: float):
    x_old = x_start_second
    x = x_start_first
    k = 0
    for k in range(1000):
        x_older = x_old
        x_old = x

        if abs(f(x_old) - f(x_older)) < epsilon:
            break

        x -= f(x_old) / inverse_slope(x_old, x_older)

        if convergence_condition(x, x_old, epsilon):
            break
    return x, k

def parabolic_approximation(x_start_first: float, x_start_second: float, x_start_third: float, epsilon: float):
    x_older = x_start_third
    x_old = x_start_second
    x = x_start_first
    k = 0
    for k in range(1000):
        x_oldest = x_older
        x_older = x_old
        x_old = x

        if abs(f(x_old) - f(x_older)) < epsilon:
            break
        if abs(f(x_older) - f(x_oldest)) < epsilon:
            break
        if abs(f(x_old) - f(x_oldest)) < epsilon:
            break

        f_co = inverse_slope(x_old, x_older)
        f_coo = (inverse_slope(x_old, x_older) - inverse_slope(x_older, x_oldest)) / (x_old - x_oldest)

        if abs(f_co) < epsilon:
            break
        if abs(f_coo) < epsilon:
            break

        first_value = (x_old - x_older) * f_coo + f_co
        second_value = first_value**2 - 4 * f_coo * f(x_old)

        if second_value < 0:
            break

        delta_plus = 1 / (2 * f_coo) * (-first_value + math.sqrt(second_value))
        delta_minus = 1 / (2 * f_coo) * (-first_value - math.sqrt(second_value))

        if abs(delta_plus) < abs(delta_minus):
            delta = delta_plus
        else:
            delta = delta_minus

        x += delta

        if convergence_condition(x, x_old, epsilon):
            break

    return x, k

def inverse_interpolation_approximation(x_start_first: float, x_start_second: float, epsilon: float):
    x_old = x_start_first
    x = x_start_second
    k = 0
    for k in range(1000):
        x_older = x_old
        x_old = x

        if abs(f(x_older) - f(x_old)) < epsilon:
            break

        x = -f(x_old) / (f(x_older) - f(x_old)) * x_older - f(x_older) / (f(x_old) - f(x_older)) * x_old

        if convergence_condition(x, x_old, epsilon):
            break
    return x, k




tabulation_start = -5
tabulation_end = 5
step = 0.1

points = []
for x in numpy.arange(tabulation_start, tabulation_end, step):
    points.append((x, f(x)))

rising_starting_approximation_x = None
falling_starting_approximation_x = None

previous_item = points[0]
for item in range(1, len(points)):
    current_item = points[item]

    if numpy.sign(current_item[1]) != numpy.sign(previous_item[1]):
        smaller_y = numpy.min([current_item[1], previous_item[1]])
        bigger_y = numpy.max([current_item[1], previous_item[1]])

        intersection_x = numpy.interp(0, [smaller_y, bigger_y], [previous_item[0], current_item[0]])

        if numpy.sign(current_item[1]) > 0:
            print(f"Intersection at 0 found: {intersection_x}, rising")

            if rising_starting_approximation_x is None:
                rising_starting_approximation_x = intersection_x
        else:
            print(f"Intersection at 0 found: {intersection_x}, falling")

            if falling_starting_approximation_x is None:
                falling_starting_approximation_x = intersection_x

    previous_item = current_item

print("---")

plot.plot([item[0] for item in points], [item[1] for item in points])
plot.plot([tabulation_start, tabulation_end], [0, 0])
plot.show()

epsilon = 10**(-14)

def print_information_about_approximation(item, extra):
    print(f"Approximated y=0, got x={item[0]} in {item[1]} steps; {extra}")

print_information_about_approximation(simple_approximation(rising_starting_approximation_x, epsilon), "rising, simple")
print_information_about_approximation(simple_approximation(falling_starting_approximation_x, epsilon), "falling, simple")
print("---")

print_information_about_approximation(newtons_approximation(rising_starting_approximation_x, epsilon), "rising, Newton's")
print_information_about_approximation(newtons_approximation(falling_starting_approximation_x, epsilon), "falling, Newton's")
print("---")

print_information_about_approximation(chebyshev_approximation(rising_starting_approximation_x, epsilon), "rising, Chebyshev's")
print_information_about_approximation(chebyshev_approximation(falling_starting_approximation_x, epsilon), "falling, Chebyshevs's")
print("---")

print_information_about_approximation(secant_approximation(rising_starting_approximation_x, rising_starting_approximation_x - 0.1, epsilon), "rising, secant")
print_information_about_approximation(secant_approximation(falling_starting_approximation_x, falling_starting_approximation_x + 0.1, epsilon), "falling, secant")
print("---")

print_information_about_approximation(parabolic_approximation(rising_starting_approximation_x, rising_starting_approximation_x - 0.1, rising_starting_approximation_x + 0.1, epsilon), "rising, parabolic")
print_information_about_approximation(parabolic_approximation(falling_starting_approximation_x, falling_starting_approximation_x + 0.1, falling_starting_approximation_x - 0.1, epsilon), "falling, parabolic")
print("---")

print_information_about_approximation(inverse_interpolation_approximation(rising_starting_approximation_x, rising_starting_approximation_x - 0.1, epsilon), "rising, inverse interpolation")
print_information_about_approximation(inverse_interpolation_approximation(falling_starting_approximation_x, falling_starting_approximation_x + 0.1, epsilon), "falling, inverse interpolation")
