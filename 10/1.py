import numpy

import matplotlib
import matplotlib.pyplot as plot

def f(x):
    return 1 / x

def fd(x):
    return -1 / x**2

def differential_equation(x, y):
    return -1 * y**2

def adams_bashforth(initial_x: float, initial_y: float, step: float, n: int) -> tuple[float, float]:
    x_0 = initial_x
    y_0 = initial_y

    if n == 0:
        return y_0, y_0

    x_1 = x_0 + step
    y_1_pred = y_0 + step * differential_equation(x_0, y_0)
    y_1_corr = y_0 + 1/2 * step * ( differential_equation(x_1, y_1_pred) + differential_equation(x_0, y_0) )

    if n == 1:
        return y_1_pred, y_1_corr

    y_pred = y_1_pred
    y_corr_prev = y_1_corr
    y_corr = y_1_corr

    for i in range(2, n + 1):
        x_prev = initial_x + (i - 1) * step
        x_curr = initial_x + i * step

        y_pred = y_corr + 1/2 * step * ( 3*differential_equation(x_prev, y_corr) - differential_equation(x_prev - step, y_corr_prev) )
        y_corr_new = y_corr + 1/2 * step * ( differential_equation(x_curr, y_pred) + differential_equation(x_prev, y_corr) )

        y_corr_prev = y_corr
        y_corr = y_corr_new

    return y_pred, y_corr



initial_x = 1.0
step = 0.01
n = 400

figure, (axis_estimated, axis_corrected) = plot.subplots(nrows=1, ncols=2, tight_layout=True)

x_values = [initial_x]
for i in range(n):
    x_values.append(x_values[-1] + step)

axis_estimated.plot(
    x_values[2:],
    [numpy.abs(f(x_values[i]) - adams_bashforth(initial_x, f(initial_x), step, i)[1]) for i in range(2, len(x_values))]
)

axis_corrected.plot(
    x_values[2:],
    [numpy.abs(adams_bashforth(initial_x, f(initial_x), step, i)[1] - adams_bashforth(initial_x, f(initial_x), step, i)[0]) for i in range(2, len(x_values))]
)

axis_estimated.set_xlabel("x")
axis_estimated.set_ylabel("Помилка (f(t))")
axis_estimated.set_title("Різниця від справжньої функції")

axis_corrected.set_xlabel("x")
axis_corrected.set_ylabel("Помилка (f(t))")
axis_corrected.set_title("Різниця від прогнозу")

figure.suptitle("Помилка метода Адамса")

plot.show()

figure, axis_optimal_step = plot.subplots(nrows=1, ncols=1, tight_layout=True)

target_x = 3.0
initial_x = 1.0
step_sizes = []
for step_size in range(6):
    step_sizes.append(10**step_size)
    step_sizes.append( (10**step_size + 10**(step_size+1) // 4))
    step_sizes.append( (10**step_size + 10**(step_size+1) // 3))
    step_sizes.append( (10**step_size + 10**(step_size+1) // 2))

smallest_error = 1000
step_with_smallest_error = -1
for step_size in step_sizes:
    error = (numpy.abs
        (
            f(target_x) - adams_bashforth(
                initial_x,
                f(initial_x),
                0.1 / step_size,
                20 * step_size
            )[1]
        ))

    if error < smallest_error:
        smallest_error = error
        step_with_smallest_error = 0.1 / step_size

print(f"Оптимальний крок: {step_with_smallest_error} (помилка {smallest_error})")

axis_optimal_step.plot(
    [0.1 / step_size for step_size in step_sizes],
    [
        numpy.abs
        (
            f(target_x) - adams_bashforth(
                initial_x,
                f(initial_x),
                0.1 / step_size,
                20 * step_size
            )[1]
        )
        for step_size in step_sizes
    ]
)

axis_optimal_step.set_xlabel("Крок")
axis_optimal_step.set_ylabel("Помилка (f(t))")

figure.suptitle("Залежність помилка метода Адамса від розміру крока")

plot.show()