import math
import matplotlib.pyplot as plot

import numpy

def soil_moisture(t):
    return 50 * math.exp(-0.1 * t) + 5 * math.sin(t)

def soil_moisture_derivative_true(t):
    return -5 * math.exp(-0.1 * t) + 5 * math.cos(t)

def soil_moisture_derivative_central_difference(t, difference):
    return ( soil_moisture(t + difference) - soil_moisture(t - difference) ) / (2 * difference)

def soil_moisture_derivative_central_difference_error(t, difference):
    return abs(soil_moisture_derivative_central_difference(t, difference) - soil_moisture_derivative_true(t))

def soil_moisture_derivative_rombergs(t, difference):
    derivative_double = soil_moisture_derivative_central_difference(t, difference * 2)
    derivative_base = soil_moisture_derivative_central_difference(t, difference)

    return derivative_base + (derivative_base - derivative_double) / 3

def soil_moisture_derivative_rombergs_error(t, difference):
    return abs(soil_moisture_derivative_rombergs(t, difference) - soil_moisture_derivative_true(t))

def soil_moisture_derivative_aitkens(t, difference):
    derivative_quadruple = soil_moisture_derivative_central_difference(t, difference * 4)
    derivative_double = soil_moisture_derivative_central_difference(t, difference * 2)
    derivative_base = soil_moisture_derivative_central_difference(t, difference)

    return derivative_quadruple - pow(derivative_double - derivative_quadruple, 2) / (derivative_base - 2 * derivative_double + derivative_quadruple)

def soil_moisture_derivative_aitkens_accuracy_order(t, difference):
    derivative_quadruple = soil_moisture_derivative_central_difference(t, difference * 4)
    derivative_double = soil_moisture_derivative_central_difference(t, difference * 2)
    derivative_base = soil_moisture_derivative_central_difference(t, difference)

    return 1 / math.log(2) * math.log((derivative_base - derivative_double) / (derivative_double - derivative_quadruple))

def soil_moisture_derivative_aitkens_error(t, difference):
    return abs(soil_moisture_derivative_aitkens(t, difference) - soil_moisture_derivative_true(t))

smallest_error = 99999
step_size_with_smallest_error = -1
for difference_step_size in range(-20, 3 + 1):
    error = soil_moisture_derivative_central_difference_error(1, pow(10, difference_step_size))

    print(f"Помилка із кроком 1e{difference_step_size}: {error}")

    if error < smallest_error:
        smallest_error = error
        step_size_with_smallest_error = difference_step_size

print(f"Крок із найменшою помилкою: 10e{step_size_with_smallest_error}")



step = pow(10, -3)

figure, axes = plot.subplots(nrows=1, ncols=1, tight_layout=True)

axes.plot(
    numpy.arange(0, 20, 0.01),
    [soil_moisture(i) for i in numpy.arange(0, 20, 0.01)],
    color="blue", linestyle="-"
)

axes.set_xlabel("Час (t)")
axes.set_ylabel("Вологість (M(t))")

figure.suptitle("Вологість ґрунту")

plot.show()

def plot_derivative(function_to_use, error_function_to_use, algorithm_name):
    axes[0].plot(
        numpy.arange(0, 20, 0.001),
        [soil_moisture_derivative_true(i) for i in numpy.arange(0, 20, 0.001)],
        color="blue", linestyle="-"
    )[0].set_label("Справжня видкість")

    axes[0].plot(
        numpy.arange(0, 20, step),
        [function_to_use(i, step) for i in numpy.arange(0, 20, step)],
        color="red", linestyle="--"
    )[0].set_label(f"Апроксимація швидкості ({algorithm_name})")

    axes[1].plot(
        numpy.arange(0, 20, step),
        [error_function_to_use(i, step) for i in numpy.arange(0, 20, step)],
        color="red", linestyle="--"
    )[0].set_label(f"Апроксимація швидкості ({algorithm_name})")

    axes[0].set_xlabel("Час (t)")
    axes[1].set_xlabel("Час (t)")
    axes[0].set_ylabel("Швидкість зміни (M'(t))")
    axes[1].set_ylabel("Швидкість зміни (M'(t))")
    axes[0].legend()
    axes[1].legend()

    figure.suptitle("Швидкість зміни вологості ґрунту")

    print(f"Помилка на f'(1): {error_function_to_use(1, step)} ({algorithm_name})")

figure, axes = plot.subplots(nrows=1, ncols=2, tight_layout=True)
plot_derivative(soil_moisture_derivative_central_difference, soil_moisture_derivative_central_difference_error, "центральна різниця")
plot.show()

figure, axes = plot.subplots(nrows=1, ncols=2, tight_layout=True)
plot_derivative(soil_moisture_derivative_rombergs, soil_moisture_derivative_rombergs_error, "метод Рунге-Ромберга")
plot.show()

figure, axes = plot.subplots(nrows=1, ncols=2, tight_layout=True)
plot_derivative(soil_moisture_derivative_aitkens, soil_moisture_derivative_aitkens_error, "метод Ейткена")

print(f"Оцінка порядку точності для f'(x): {soil_moisture_derivative_aitkens_accuracy_order(1, step)} (метод Ейткена)")

plot.show()