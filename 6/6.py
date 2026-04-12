import numpy

rng = numpy.random.default_rng()

def form_random_matrix(size: int) -> numpy.ndarray:
    matrix = numpy.zeros((size, size))
    for i in range(size):
        for j in range(size):
            matrix[i, j] = rng.random()
    return matrix

def form_random_vector(size: int) -> numpy.ndarray:
    vector = numpy.zeros(size)
    for i in range(size):
        vector[i] = rng.random()
    return vector


def multiply_matrix_vector(a_matrix: numpy.ndarray, b_vector: numpy.ndarray) -> numpy.ndarray:
    x_vector = numpy.zeros(b_vector.shape)
    for i in range(a_matrix.shape[0]):
        for j in range(a_matrix.shape[1]):
            x_vector[i] += a_matrix[i, j] * b_vector[j]

    return x_vector

def lu_decomposition(matrix: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
    l_matrix = numpy.zeros(matrix.shape)
    u_matrix = numpy.zeros(matrix.shape)

    size = matrix.shape[0]

    for i in range(size):
        u_matrix[i, i] = 1

    for k in range(size):
        for i in range(k, size):
            l_matrix[i, k] = matrix[i, k] - numpy.dot(l_matrix[i, :k], u_matrix[:k, k])

        for j in range(k + 1, size):
            u_matrix[k, j] = (matrix[k, j] - numpy.dot(l_matrix[k, :k], u_matrix[:k, j])) / l_matrix[k, k]

        #for i in range(size):
        #    if i >= k:
        #        l_matrix[i, k] = matrix[i, k] - numpy.sum([l_matrix[i, j] * u_matrix[j, k] for j in range(k)])

        #for i in range(size):
        #    if i > k:
        #        u_matrix[k, i] = 1 / l_matrix[k, k] * ( matrix[k, i] - numpy.sum([l_matrix[k, j] * u_matrix[j, i] for j in range(k)]) )

    return l_matrix, u_matrix

def solve_lu_b(l_matrix: numpy.ndarray, u_matrix: numpy.ndarray, b_vector: numpy.ndarray) -> numpy.ndarray:
    size = l_matrix.shape[0]

    z_vector = numpy.zeros(size)
    z_vector[0] = b_vector[0] / l_matrix[0, 0]

    for k in range(1, size):
        z_vector[k] = 1 / l_matrix[k, k] * (b_vector[k] - numpy.sum([l_matrix[k, j] * z_vector[j] for j in range(k)]))

    x_vector = numpy.zeros(size)
    x_vector[size - 1] = z_vector[size - 1]

    for k in range(size - 2, -1, -1):
        x_vector[k] = z_vector[k] - numpy.sum([u_matrix[k, j] * x_vector[j] for j in range(k + 1, size)])

    return x_vector

def iterative_refinement(
        a_matrix: numpy.ndarray,
        b_vector: numpy.ndarray,
        x_vector_approximation: numpy.ndarray,
        error: float
    ) -> numpy.ndarray:

    l_matrix, u_matrix = lu_decomposition(a_matrix)

    x_vector_approximation_refined = x_vector_approximation.copy()

    iteration_count = 0
    while True:
        residual_vector = b_vector - multiply_matrix_vector(a_matrix, x_vector_approximation_refined)

        error_vector_approximation = solve_lu_b(l_matrix, u_matrix, residual_vector)

        if (
            numpy.all(numpy.abs(error_vector_approximation) <= error) #and
            #numpy.all( numpy.abs(multiply_matrix_vector(a_matrix, x_vector_approximation_refined) - b_vector ) <= error)
        ) or iteration_count >= 1000:
            return x_vector_approximation_refined

        x_vector_approximation_refined += error_vector_approximation

        iteration_count += 1



a_matrix = form_random_matrix(100) * 100
x_vector = form_random_vector(100) * 50 + 5
b_vector = multiply_matrix_vector(a_matrix, x_vector)

l_matrix, u_matrix = lu_decomposition(a_matrix)

x_vector_approximation = solve_lu_b(l_matrix, u_matrix, b_vector)
epsilon = numpy.max(numpy.abs(multiply_matrix_vector(a_matrix, x_vector_approximation) - b_vector))
print(f"Помилка без СЛАР: {epsilon}")

x_vector_refined_approximation = iterative_refinement(a_matrix, b_vector, x_vector_approximation, 10**(-14))
epsilon = numpy.max(numpy.abs(multiply_matrix_vector(a_matrix, x_vector_refined_approximation) - b_vector))
print(f"Помилка із СЛАР: {epsilon}")