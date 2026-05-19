import numpy

import matplotlib
import matplotlib.pyplot as plot

def f(x):
    return 1 / x

def fd(x):
    return -1 / x**2

def differential_equation(x, y):
    return -1 * y**2

def runge_kutta(initial_x: float, initial_y: float, step: float, n: int) -> float:
    x = initial_x
    y = initial_y

    if n == 0:
        return y

    y_pred = y

    for i in range(n):
        x_curr = initial_x + step * i

        k_1 = differential_equation(x_curr, y_pred)
        k_2 = differential_equation(x_curr + step / 2, y_pred + step * k_1 / 2)
        k_3 = differential_equation(x_curr + step / 2, y_pred + step * k_2 / 2)
        k_4 = differential_equation(x_curr + step, y_pred + step * k_3)

        y_pred_new = y_pred + 1/6 * step * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
        y_pred = y_pred_new

    return y_pred



initial_x = 1.0
step = 0.01
n = 400

figure, (axis_estimated, axis_corrected) = plot.subplots(nrows=1, ncols=2, tight_layout=True)

x_values = [initial_x]
for i in range(n):
    x_values.append(x_values[-1] + step)

axis_estimated.plot(
    x_values[2:],
    [numpy.abs(f(x_values[i]) - runge_kutta(initial_x, f(initial_x), step, i)) for i in range(2, len(x_values))]
)

axis_corrected.plot(
    x_values[2:],
    [16 / 15 * numpy.abs(runge_kutta(initial_x, f(initial_x), step / 2, i * 2) - runge_kutta(initial_x, f(initial_x), step, i)) for i in range(2, len(x_values))]
)

axis_estimated.set_xlabel("x")
axis_estimated.set_ylabel("Помилка (f(t))")
axis_estimated.set_title("Різниця від справжньої функції")

axis_corrected.set_xlabel("x")
axis_corrected.set_ylabel("Помилка (f(t))")
axis_corrected.set_title("Різниця від прогнозу")

figure.suptitle("Помилка метода Рунге-Кутта")

plot.show()

figure, axis_optimal_step = plot.subplots(nrows=1, ncols=1, tight_layout=True)

initial_x = 0.1

step_sizes = []
for step_size in range(4):
    step_sizes.append(10**step_size)
    step_sizes.append( (10**step_size + 10**(step_size+1) // 4))
    step_sizes.append( (10**step_size + 10**(step_size+1) // 3))
    step_sizes.append( (10**step_size + 10**(step_size+1) // 2))

target_x_ranges = []
smallest_errors_for_steps = []

for target_x in numpy.arange(0.2, 2 + 0.1, 0.1):
    smallest_error = 1000
    step_with_smallest_error = -1

    difference = target_x - initial_x
    target_n = int(10 * difference)

    for step_size in step_sizes:
        error = (numpy.abs
            (
                f(target_x) - runge_kutta(
                    initial_x,
                    f(initial_x),
                    0.01 / step_size,
                    target_n * step_size
                )
            ))

        if error < smallest_error:
            smallest_error = error
            step_with_smallest_error = 0.1 / step_size

    target_x_ranges.append(target_x)
    smallest_errors_for_steps.append(step_with_smallest_error)

axis_optimal_step.plot(
    target_x_ranges,
    smallest_errors_for_steps
)

axis_optimal_step.set_xlabel("x")
axis_optimal_step.set_ylabel("Крок")

figure.suptitle("Оптимальний крок для кожного X методом Рунге-Кутта")

plot.show()

figure, axis_error_relation = plot.subplots(nrows=1, ncols=1, tight_layout=True)

axis_error_relation.plot(
    [0.1 / step_size for step_size in step_sizes],
    [
        numpy.abs
        (
            f(1) - runge_kutta(
                initial_x,
                f(initial_x),
                0.01 / step_size,
                90 * step_size
            )
        )
        for step_size in step_sizes
    ]
)

axis_error_relation.set_xlabel("Крок")
axis_error_relation.set_ylabel("Помилка (f(t))")

figure.suptitle("Залежність помилка метода Рунге-Кутта від розміру крока")

plot.show()