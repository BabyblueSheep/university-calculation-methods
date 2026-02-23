import numpy, matplotlib.pyplot as plot
import requests

from typing import Final



class CubicSpline:
    """Represents a cubic spline composed of multiple 2-dimensional points."""

    @staticmethod
    def thomas_algorithm(alpha_values: list[float], beta_values: list[float], gamma_values: list[float], delta_values: list[float]) -> list[float]:
        """Solves a system of equations using the Thomas algorithm."""
        a_values = [-1] * len(alpha_values)
        b_values = [-1] * len(alpha_values)

        x_values = [-1] * len(alpha_values)

        a_values[0] = -gamma_values[0] / beta_values[0]
        b_values[0] = delta_values[0] / beta_values[0]

        for i in range(1, len(a_values)):
            a_values[i] = -gamma_values[i] / (alpha_values[i] * a_values[i - 1] + beta_values[i])
            b_values[i] = (delta_values[i] - alpha_values[i] * b_values[i - 1]) / (alpha_values[i] * a_values[i - 1] + beta_values[i])

        x_values[-1] = b_values[-1]
        for i in range(len(x_values) - 2, -1, -1):
            x_values[i] = x_values[i + 1] * a_values[i] + b_values[i]

        return x_values

    def __init__(self, x_values: list[float], y_values: list[float]):
        self.x = x_values
        self.y = y_values

        dx = [x_values[i] - x_values[i - 1] for i in range(1, len(x_values))]
        dx.append(dx[-1])
        dy = [y_values[i] - y_values[i - 1] for i in range(1, len(y_values))]
        dy.append(dy[-1])

        self.a = [y for y in self.y[:]]

        c_alpha = [-1] * (len(self.a))
        c_beta =  [-1] * (len(self.a))
        c_gamma = [-1] * (len(self.a))
        c_delta = [-1] * (len(self.a))

        c_alpha[0] = 0
        c_beta[0] = 1
        c_gamma[0] = 0
        c_delta[0] = 0

        for i in range(1, len(c_alpha) - 1):
            c_alpha[i] = dx[i - 1]
            c_beta[i]  = 2 * (dx[i - 1] + dx[i])
            c_gamma[i] = dx[i]
            c_delta[i] = 3 * (dy[i] / dx[i] - dy[i - 1] / dx[i - 1])

        c_gamma[-1] = 0

        self.c = CubicSpline.thomas_algorithm(c_alpha, c_beta, c_gamma, c_delta)

        self.d = [(self.c[i + 1] - self.c[i]) / (3 * dx[i]) for i in range(len(self.a) - 1)]

        self.b = [dy[i] / dx[i] - dx[i] / 3 * (self.c[i + 1] + 2 * self.c[i]) for i in range(len(self.a) - 1)]

    def get(self, i: float) -> tuple[float, float]:
        """Gets an interpolated coordinate along the spline."""
        if i < 0:
            i = 0
        if i > len(self.x) - 2:
            i = len(self.x) - 2

        i_whole = int(numpy.floor(i))
        i_fract = i % 1.0

        x0 = self.x[i_whole]
        x = x0 + (self.x[i_whole + 1] - x0) * i_fract
        dx = x - x0

        y = self.a[i_whole] + self.b[i_whole] * dx + self.c[i_whole] * dx*dx + self.d[i_whole] * dx*dx*dx

        return x, y





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

    def project_mercator(self):
        scale = 1
        longitude_central_meridian = 0
        x = self.longitude - longitude_central_meridian
        y = numpy.log(numpy.tan(numpy.pi / 4 + self.latitude / 2))
        return scale * x, scale * y



figure, axes = plot.subplots(nrows=1, ncols=2, constrained_layout=True)


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

axes[0].plot(distances_meters, [point.elevation / 1000 for point in points])


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

# Calculate a cubic spline from sample points
projected_positions = [point.project_mercator() for point in points]
x_positions = [position[0] for position in projected_positions]
y_positions = [position[1] for position in projected_positions]

cubic_spline = CubicSpline(x_positions, y_positions)

spline_x_positions = [cubic_spline.get(i)[0] for i in numpy.arange(0, len(projected_positions) - 1, 0.25)]
spline_y_positions = [cubic_spline.get(i)[1] for i in numpy.arange(0, len(projected_positions) - 1, 0.25)]

spline_x_positions_granular = [cubic_spline.get(i)[0] for i in numpy.arange(0, len(projected_positions) - 1, 0.1)]
spline_y_positions_granular = [cubic_spline.get(i)[1] for i in numpy.arange(0, len(projected_positions) - 1, 0.1)]

spline_y_positions_granular_error = [y_positions[int(numpy.floor(i))] - cubic_spline.get(i)[1] for i in numpy.arange(0, len(projected_positions) - 1, 0.1)]

#axes[1].plot(spline_x_positions_granular, spline_y_positions_granular_error, linestyle="--", linewidth=2, color="red")
axes[1].plot(spline_x_positions_granular, spline_y_positions_granular, linestyle="--", linewidth=2, color="green")
axes[1].plot(spline_x_positions, spline_y_positions, linestyle="--", linewidth=2, color="blue")
axes[1].plot(x_positions, y_positions, linestyle="--", linewidth=2, color="black")

plot.show()