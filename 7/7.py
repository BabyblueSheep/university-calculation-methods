import numpy

rng = numpy.random.default_rng()

def form_random_matrix(size: int, scale: float) -> numpy.ndarray:
    matrix = numpy.zeros((size, size))
    for i in range(size):
        row_sum = 0
        for j in range(size):
            matrix[i, j] = rng.random()
            row_sum += matrix[i, j]
        row_sum -= matrix[i, i]
        matrix[i, i] = row_sum * scale
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

def vector_difference_norm(vector_first: numpy.ndarray, vector_second: numpy.ndarray) -> float:
    return numpy.max(numpy.abs(vector_first - vector_second))

def matrix_norm(matrix: numpy.ndarray) -> float:
    return numpy.max(numpy.sum(numpy.abs(matrix), axis=1))

def simple_iterative_method(a_matrix: numpy.ndarray, b_vector: numpy.ndarray, epsilon: float) -> numpy.ndarray:
    size = a_matrix.shape[0]

    tau = 0.01 / matrix_norm(a_matrix)

    x_vector = numpy.ones(b_vector.shape)

    for k in range(10000):

        old_x_vector = x_vector.copy()

        for i in range(size):
            matrix_sum = 0
            for j in range(size):
                matrix_sum += a_matrix[i, j] * old_x_vector[j]

            x_vector[i] = old_x_vector[i] - tau * matrix_sum + tau * b_vector[i]

        #if vector_difference_norm(old_x_vector, x_vector) <= epsilon:
        #    break

    return x_vector

def jacobi_method(a_matrix: numpy.ndarray, b_vector: numpy.ndarray, epsilon: float) -> numpy.ndarray:
    size = a_matrix.shape[0]

    x_vector = numpy.ones(b_vector.shape)

    for k in range(10000):

        old_x_vector = x_vector.copy()

        for i in range(size):
            matrix_sum = 0
            for j in range(size):
                matrix_sum += a_matrix[i, j] * old_x_vector[j]
            matrix_sum -= a_matrix[i, i] * old_x_vector[i]

            x_vector[i] = (b_vector[i] - matrix_sum) / a_matrix[i, i]

        #if vector_difference_norm(old_x_vector, x_vector) <= epsilon:
        #    break

    return x_vector

def seidel_method(a_matrix: numpy.ndarray, b_vector: numpy.ndarray, epsilon: float) -> numpy.ndarray:
    size = a_matrix.shape[0]

    x_vector = numpy.ones(b_vector.shape)

    for k in range(10000):

        old_x_vector = x_vector.copy()

        for i in range(size):
            matrix_sum = 0
            for j in range(size):
                if j < i:
                    matrix_sum += a_matrix[i, j] * x_vector[j]
                if j > i:
                    matrix_sum += a_matrix[i, j] * old_x_vector[j]

            x_vector[i] = (b_vector[i] - matrix_sum) / a_matrix[i, i]

        #if vector_difference_norm(old_x_vector, x_vector) <= epsilon:
        #    break

    return x_vector


a_matrix = form_random_matrix(100, 10) * 100
x_vector = form_random_vector(100) * 50 + 5
b_vector = multiply_matrix_vector(a_matrix, x_vector)

print(vector_difference_norm(b_vector, multiply_matrix_vector(a_matrix, x_vector)))

x_vector_approximation = simple_iterative_method(a_matrix, b_vector, 10**(-14))
print(vector_difference_norm(b_vector, multiply_matrix_vector(a_matrix, x_vector_approximation)))

x_vector_approximation = jacobi_method(a_matrix, b_vector, 10**(-14))
print(vector_difference_norm(b_vector, multiply_matrix_vector(a_matrix, x_vector_approximation)))

x_vector_approximation = seidel_method(a_matrix, b_vector, 10**(-14))
print(vector_difference_norm(b_vector, multiply_matrix_vector(a_matrix, x_vector_approximation)))