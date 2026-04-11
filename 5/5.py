from numpy import sin, cos, exp, pi, sqrt
import numpy
from scipy.special import erf
import matplotlib.pyplot as plot

def f(x):
    return (
            50 +
            20 * sin(pi * x / 12) +
            5 * exp( -0.2 * (x - 12)**2 )
           )

def f_antiderivative(x):
    return (
            50 * x -
            240 * cos(pi * x / 12) / pi +
            5 * sqrt(5 * pi) / 2 * erf( sqrt(0.2) * (x - 12) )
           )

def f_integral(a, b):
    return f_antiderivative(b) - f_antiderivative(a)

def f_simpsons(n, a, b):
    h = (b - a) / n

    def f_i(i):
        return f(a + i * h)

    even_sum = 0
    odd_sum = 0

    for i in range(1, n, 2):
        odd_sum += f_i(i)
    for i in range(2, n - 1, 2):
        even_sum += f_i(i)

    return h / 3 * ( f_i(0) + 4 * odd_sum + 2 * even_sum + f_i(n) )

def f_simpsons_error(n, a, b):
    return numpy.abs(f_simpsons(n, a, b) - f_integral(a, b))

def f_rombergs(n, a, b):
    f = f_simpsons(n, a, b)
    f_half = f_simpsons(n // 2, a, b)

    return f + (f - f_half) / 15

def f_rombergs_error(n, a, b):
    return abs(f_rombergs(n, a, b) - f_integral(a, b))

def f_aitkens(n, a, b):
    f = f_simpsons(n, a, b)
    f_half = f_simpsons(n // 2, a, b)
    f_quart = f_simpsons(n // 4, a, b)

    return (f_half * f_half - f * f_quart) / (2 * f_half - f - f_quart)

def f_aitkens_accuracy_order(n, a, b):
    f = f_simpsons(n, a, b)
    f_half = f_simpsons(n // 2, a, b)
    f_quart = f_simpsons(n // 4, a, b)

    return 1 / numpy.log(2) * numpy.abs(numpy.log((f_quart - f_half) / (f_half - f)))

def f_aitkens_error(n, a, b):
    return numpy.abs(f_aitkens(n, a, b) - f_integral(a, b))

def f_adaptive(a, b, delta):
    h = (b - a)
    def f_i(i):
        return f(a + i * h)

    i_1 = h / 6 * (f_i(0) + 4 * f_i(0.5) + f_i(1))
    i_2 = h / 12 * (f_i(0) + 4 * f_i(0.25) + 2 * f_i(0.5) + 4 * f_i(0.75) + f_i(1))

    if numpy.abs(i_1 - i_2) <= delta:
        return i_2
    return f_adaptive(a, (a + b) * 0.5, delta) + f_adaptive((a + b) * 0.5, b, delta)



time_start = 0
time_end = 25

total_integral = f_integral(time_start, time_end)
print(f"Інтеграл функції від a={time_start} до b={time_end}: {total_integral}")



figure, (axis_original, axis_integral) = plot.subplots(nrows=1, ncols=2, tight_layout=True)

axis_original.plot(
    numpy.arange(time_start, time_end, 0.01),
    [f(i) for i in numpy.arange(time_start, time_end, 0.01)],
    color="blue", linestyle="-"
)
axis_original.set_xlabel("Час (t)")
axis_original.set_ylabel("Навантаження (f(t))")

axis_integral.plot(
    numpy.arange(time_start, time_end, 0.01),
    [f_antiderivative(i) for i in numpy.arange(time_start, time_end, 0.01)],
    color="blue", linestyle="-"
)
axis_integral.set_xlabel("Час (t)")
axis_integral.set_ylabel("Навантаження (F(t))")

figure.suptitle("Зміна інтенсивності навантаження на сервер")

plot.show()





figure, axis_difference = plot.subplots(nrows=1, ncols=1, tight_layout=True)

axis_difference.plot(
    range(100, 1000 + 10, 10),
    [f_simpsons_error(n, time_start, time_end) for n in range(100, 1000 + 10, 10)],
    color="blue", linestyle="-"
)

axis_difference.set_xlabel("Крок (N)")
axis_difference.set_ylabel("Помилка (F(t))")

figure.suptitle("Залежність помилка метода Сімпсона від розміру крока")

n_optimal = 100
current_error = f_simpsons_error(n_optimal, time_start, time_end)
optimal_error = 10**(-12)
while current_error >= optimal_error:
    n_optimal = n_optimal + 25
    current_error = f_simpsons_error(n_optimal, time_start, time_end)

print(f"Помилка метода Сімпсона при N {n_optimal}: {current_error}")

plot.show()





n_o = n_optimal / 10
n_o = int(n_o // 8 * 8)

print(f"Помилка методом Сімпсона при N {n_o}: {f_simpsons_error(n_o, time_start, time_end)}")
print(f"Помилка методом Сімпсона & Ромберга при N {n_o}: {f_rombergs_error(n_o, time_start, time_end)}")
print(f"Помилка методом Сімпсона & Ейткена при N {n_o}: {f_aitkens_error(n_o, time_start, time_end)}")

figure, axis_difference = plot.subplots(nrows=1, ncols=1, tight_layout=True)

axis_difference.plot(
    range(128, 1000 + 8, 8),
    [f_simpsons_error(n, time_start, time_end) for n in range(128, 1000 + 8, 8)],
    color="blue", linestyle="--"
)[0].set_label("Метод Сімпсона")
axis_difference.plot(
    range(128, 1000 + 8, 8),
    [f_rombergs_error(n, time_start, time_end) for n in range(128, 1000 + 8, 8)],
    color="red", linestyle="--"
)[0].set_label("Метод Сімпсона + Ромберга")
axis_difference.plot(
    range(128, 1000 + 8, 8),
    [f_aitkens_error(n, time_start, time_end) for n in range(128, 1000 + 8, 8)],
    color="green", linestyle="--"
)[0].set_label("Метод Сімпсона + Ейткена")

axis_difference.set_xlabel("Крок (N)")
axis_difference.set_ylabel("Помилка (F(t))")
axis_difference.legend()

figure.suptitle("Залежність помилка метода Сімпсона від розміру крока")

plot.show()





figure, axis_adaptive = plot.subplots(nrows=1, ncols=1, tight_layout=True)

axis_adaptive.plot(
    range(-10, -1 + 1),
    [numpy.abs(f_adaptive(time_start, time_end, 10**delta_power) - f_integral(time_start, time_end)) for delta_power in range(-10, -1 + 1)],
    color="blue", linestyle="-"
)

axis_adaptive.set_xlabel("Степінь епсилона")
axis_adaptive.set_ylabel("Помилка (F(t))")

figure.suptitle("Залежність помилка адаптивного метода від епсилона")

plot.show()