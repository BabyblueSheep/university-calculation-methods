import math
import matplotlib.pyplot as plot

import numpy

def soil_moisture(t):
    return 50 * math.exp(-0.1 * t) + 5 * math.sin(t)

def soil_moisture_derivative_true(t):
    return -5 * math.exp(-0.1 * t) + 5 * math.cos(t)

def soil_moisture_derivative_central_difference(t, difference):
    return ( soil_moisture(t + difference) - soil_moisture(t - difference) )/ (2 * difference)

def soil_moisture_derivative_central_difference_error(t, difference):
    return abs(soil_moisture_derivative_central_difference(t, difference) - soil_moisture_derivative_true(t))

def soil_moisture_derivative_rombergs(t, difference):
    return (soil_moisture_derivative_central_difference_error(t, difference) +
            (soil_moisture_derivative_central_difference_error(t, difference) - soil_moisture_derivative_central_difference_error(t, difference * 2)) / 3)

def soil_moisture_derivative_rombergs_error(t, difference):
    return abs(soil_moisture_derivative_rombergs(t, difference) - soil_moisture_derivative_true(t))



for difference_step_size in range(-20, 3 + 1):
    print(f"10e{difference_step_size}: {soil_moisture_derivative_central_difference_error(0, pow(10, difference_step_size))}")


figure, axes = plot.subplots(nrows=1, ncols=1, tight_layout=True)

axes.plot(
    numpy.arange(0, 20, 1),
    [soil_moisture_derivative_central_difference(i, 1) for i in numpy.arange(0, 20, 1)],
    color="blue", linestyle="-"
)
axes.plot(
    numpy.arange(0, 20, 1),
    [soil_moisture_derivative_true(i) for i in numpy.arange(0, 20, 1)],
    color="red", linestyle="--"
)

plot.show()