import numpy

rng = numpy.random.default_rng()

def form_matrix(size: int) -> numpy.ndarray:
    matrix = numpy.zeros((size, size))
    for i in range(size):
        for j in range(size):
            matrix[i, j] = rng.random() * 100
    return matrix

a_matrix = form_matrix(100)
x_vector = numpy.ones(100) * 2.5

b_vector = numpy.zeros(100)
for i in range(a_matrix.shape[0]):
    for j in range(a_matrix.shape[1]):
        b_vector[i] += a_matrix[i, j] * x_vector[j]

def lu_decomposition(matrix: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
    l_matrix = numpy.zeros(matrix.shape)
    u_matrix = numpy.zeros(matrix.shape)

    for i in range(u_matrix.shape[0]):
        u_matrix[i, i] = 1

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
