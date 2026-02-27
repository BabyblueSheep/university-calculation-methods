from __future__ import annotations

import numpy, matplotlib.pyplot as plot
import requests

from typing import Final

from matplotlib.lines import Line2D
from matplotlib.pyplot import tight_layout


class CubicSpline:
    """Represents a cubic spline composed of multiple 2-dimensional points."""

    @staticmethod
    def thomas_algorithm(alpha_values: list[float], beta_values: list[float], gamma_values: list[float], delta_values: list[float]) -> list[float]:
        """Solves a system of equations using the Thomas algorithm."""
        a_values = [0] * len(alpha_values)
        b_values = [0] * len(alpha_values)

        x_values = [0] * len(alpha_values)

        a_values[0] = -gamma_values[0] / beta_values[0]
        b_values[0] = delta_values[0] / beta_values[0]

        for i in range(1, len(a_values)):
            denominator = alpha_values[i] * a_values[i - 1] + beta_values[i]

            a_values[i] = -gamma_values[i] / denominator
            b_values[i] = (delta_values[i] - alpha_values[i] * b_values[i - 1]) / denominator

        x_values[-1] = b_values[-1]
        for i in range(len(x_values) - 2, -1, -1):
            x_values[i] = x_values[i + 1] * a_values[i] + b_values[i]

        return x_values

    def __init__(self, x_values: list[float], y_values: list[float]):
        self.x = x_values

        dx = [x_values[i + 1] - x_values[i] for i in range(0, len(x_values) - 1)]
        dy = [y_values[i + 1] - y_values[i] for i in range(0, len(y_values) - 1)]

        c_alpha = [0] * len(dx)
        c_beta =  [0] * len(dx)
        c_gamma = [0] * len(dx)
        c_delta = [0] * len(dx)

        c_alpha[0] = 0
        c_beta[0] = 1
        c_gamma[0] = 0
        c_delta[0] = 0

        for i in range(1, len(c_alpha)):
            c_alpha[i] = dx[i - 1]
            c_beta[i]  = 2 * (dx[i - 1] + dx[i])
            c_gamma[i] = dx[i]
            c_delta[i] = 3 * (dy[i] / dx[i] - dy[i - 1] / dx[i - 1])

        c_gamma[-1] = 0

        self.c = CubicSpline.thomas_algorithm(c_alpha, c_beta, c_gamma, c_delta)

        self.a = [y for y in y_values[:-1]]
        self.d = [(self.c[i + 1] - self.c[i]) / (3 * dx[i]) for i in range(len(self.a) - 1)]
        self.d.append(-self.c[-1] / (3 * dx[-1]))
        self.b = [dy[i] / dx[i] - dx[i] / 3 * (self.c[i + 1] + 2 * self.c[i]) for i in range(len(self.a) - 1)]
        self.b.append(dy[-1] / dx[-1] - 2 / 3 * dx[-1] * self.c[-1])

    def get_x(self, i: float) -> float:
        if i < 0:
            return self.x[0]
        if i >= len(self.x) - 1:
            return self.x[-1]

        i_whole = int(numpy.floor(i))
        i_fract = i % 1.0

        x0 = self.x[i_whole]
        x = x0 + (self.x[i_whole + 1] - x0) * i_fract

        return x

    def get_y(self, x: float) -> float:
        i = len(self.x) - 2
        while i > 0:
            if self.x[i] < x:
                break
            i -= 1

        x0 = self.x[i]
        dx = x - x0

        y = self.a[i] + self.b[i] * dx + self.c[i] * dx*dx + self.d[i] * dx*dx*dx

        return y

class GeographicPosition:
    """Represents a position on Earth using a geographic coordinate system."""

    EARTH_RADIUS_KILOMETERS: Final[float] = 6371

    def __init__(self, latitude: float, longitude: float, elevation: float):
        """Creates a new GeographicPosition object. Latitude and longitude are represented as radians, while elevation is represented as meters from sea level."""
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation

    @staticmethod
    def haversine_angle(first: GeographicPosition, second: GeographicPosition) -> float:
        """Returns the haversine distance between two points in radians."""
        haversine = (1 - numpy.cos(second.latitude - first.latitude) + numpy.cos(second.latitude) * numpy.cos(first.latitude) * (1 - numpy.cos(second.longitude - first.longitude))) / 2
        return 2 * numpy.arcsin(numpy.sqrt(haversine))

    @staticmethod
    def haversine_distance_kilometers(first: GeographicPosition, second: GeographicPosition):
        """Returns the haversine distance between two points in kilometers."""
        return GeographicPosition.haversine_angle(first, second) * GeographicPosition.EARTH_RADIUS_KILOMETERS



figure, axes = plot.subplots(nrows=1, ncols=2, tight_layout=True)


url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"
response = requests.get(url)
data = response.json()

with open("points", "w") as file:
  for point in data["results"]:
    file.write(f"Longitude: {point["longitude"]}, Latitude: {point['latitude']}, Elevation: {point['elevation']}\n")

points: list[GeographicPosition] = []
for point in data["results"]:
    points.append(GeographicPosition(numpy.radians(point["latitude"]), numpy.radians(point["longitude"]), point["elevation"]))


# Display relation between elevation and total distance from start
distances_meters: list[float] = [0]
for i in range(1, len(points)):
    distances_meters.insert(i, distances_meters[i - 1] + GeographicPosition.haversine_distance_kilometers(points[i], points[i - 1]))

# Print information about route
print(f"Total distance of route: {distances_meters[-1]} km")

total_ascent = 0
total_descent = 0
for i in range(1, len(points) - 1):
    total_ascent += numpy.fmax(0, points[i].elevation - points[i - 1].elevation)
    total_descent += numpy.fmax(0, points[i - 1].elevation - points[i].elevation)

print(f"Total ascent of route: {total_ascent} m")
print(f"Total descent of route: {total_descent} m")

mass_kilograms = 80
energy = mass_kilograms * 9.81 * total_ascent
print(f"Total energy used during ascent: {energy / 1000} kJ")
print(f"Total energy used during ascent: {energy / 4184} calories")

# Calculate a cubic spline from sample points (cumulative distance to elevation)
def plot_spline(spline: CubicSpline, color: str) -> list[Line2D]:
    subpoint_amount = 100
    spline_x_positions = [spline.get_x(i) for i in numpy.linspace(0, len(spline.x) - 1, subpoint_amount)]
    spline_y_positions = [spline.get_y(x) for x in spline_x_positions]
    return axes[0].plot(spline_x_positions, spline_y_positions, linestyle="-", linewidth=2, color=color)

def plot_splines_error(spline_one: CubicSpline, spline_two: CubicSpline, color: str) -> list[Line2D]:
    subpoint_amount = 100
    spline_x_error = [spline_one.get_x(i) for i in numpy.linspace(0, len(spline_one.x) - 1, subpoint_amount)]
    spline_y_error = [numpy.abs(spline_one.get_y(x) - spline_two.get_y(x)) for x in spline_x_error]
    return axes[1].plot(spline_x_error, spline_y_error, linestyle="-", linewidth=2, color=color)

x_positions = [distances_meter for distances_meter in distances_meters]
y_positions = [point.elevation / 1000 for point in points]
cubic_spline = CubicSpline(x_positions, y_positions)
plot_spline(cubic_spline, "black")[0].set_label("21 (всі) точка")

x_positions_10_points = [x_positions[i] for i in numpy.linspace(0, len(x_positions) - 1, 10, dtype=int)]
y_positions_10_points = [y_positions[i] for i in numpy.linspace(0, len(y_positions) - 1, 10, dtype=int)]
cubic_spline_10_points = CubicSpline(x_positions_10_points, y_positions_10_points)
plot_spline(cubic_spline_10_points, "red")[0].set_label("10 точок")

x_positions_15_points = [x_positions[i] for i in numpy.linspace(0, len(x_positions) - 1, 15, dtype=int)]
y_positions_15_points = [y_positions[i] for i in numpy.linspace(0, len(y_positions) - 1, 15, dtype=int)]
cubic_spline_15_points = CubicSpline(x_positions_15_points, y_positions_15_points)
plot_spline(cubic_spline_15_points, "green")[0].set_label("15 точок")

x_positions_20_points = [x_positions[i] for i in numpy.linspace(0, len(x_positions) - 1, 20, dtype=int)]
y_positions_20_points = [y_positions[i] for i in numpy.linspace(0, len(y_positions) - 1, 20, dtype=int)]
cubic_spline_20_points = CubicSpline(x_positions_20_points, y_positions_20_points)
plot_spline(cubic_spline_20_points, "blue")[0].set_label("20 точок")

axes[0].set_xlabel("Висота (км)")
axes[0].set_ylabel("Кумулятивна відстань (км)")
axes[0].set_title("Візуалізація кубічних сплайнів")
axes[0].legend()

plot_splines_error(cubic_spline, cubic_spline_10_points, "red")[0].set_label("10 точок")
plot_splines_error(cubic_spline, cubic_spline_15_points, "green")[0].set_label("15 точок")
plot_splines_error(cubic_spline, cubic_spline_20_points, "blue")[0].set_label("20 точок")

axes[1].set_xlabel("Висота (км)")
axes[1].set_ylabel("Кумулятивна відстань (км)")
axes[1].set_title("Візуалізація похибки сплайнів із меншими кількостями точок")
axes[1].legend()

plot.show()