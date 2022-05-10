import numpy as np
import time


class FitFunctions:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """
        self.coef = None
        self.cramer = None

    def fit(self, f: callable, a: float, b: float, d: int, maxtime: float) -> callable:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape. Using mean least error


        Parameters
        ----------
        f : callable. 
            A function which returns an approximate (noisy) Y value given X. 
        a: float
            Start of the fitting range
        b: float
            End of the fitting range
        d: int 
            The expected degree of a polynomial matching f
        maxtime : float
            This function returns after at most maxtime seconds. 

        Returns
        -------
        a function:float->float that fits f between a and b
        """

        def zeros_matrix(rows, cols):
            A = []
            for i in range(rows):
                A.append([])
                for j in range(cols):
                    A[-1].append(0.0)

            return A

        def copy_matrix(M):
            rows = len(M)
            cols = len(M[0])

            MC = zeros_matrix(rows, cols)

            for i in range(rows):
                for j in range(rows):
                    MC[i][j] = M[i][j]

            return MC

        def invert_mat(matrix):
            n = len(matrix)
            AM = copy_matrix(matrix)
            I = np.identity(n)
            IM = copy_matrix(I)
            indices = list(range(n))
            for fd in range(n):
                fdScalaer = 1.0 / AM[fd][fd]
                for j in range(n):
                    AM[fd][j] *= fdScalaer
                    IM[fd][j] *= fdScalaer
                for i in indices[0:fd] + indices[fd + 1:]:
                    crScaler = AM[i][fd]
                    for j in range(n):
                        AM[i][j] = AM[i][j] - crScaler * AM[fd][j]
                        IM[i][j] = IM[i][j] - crScaler * IM[fd][j]
            return np.array(IM)

        clock = time.time()
        n = d * d

        def x_to_the_power_of_i(x, i):
            return np.power(x, i)

        #  Generate Points
        x_values = np.linspace(a, b, num=n, endpoint=True)
        y_values = [0 for _ in x_values]
        times_of_sum = 0

        while time.time() - clock < maxtime - 1.0:
            times_of_sum += 1
            for i in range(len(x_values)):
                y_values[i] += f(x_values[i])

        for k, y in enumerate(y_values):
            y_values[k] = y / times_of_sum

        a_matrix = []
        y_vector = []

        j = d
        temp_sum_of_xi = 0
        for row in range(d + 1):
            temp_row = []
            while j >= 0:
                for x in x_values:
                    temp_sum_of_xi += x_to_the_power_of_i(x, j + (d - row))
                j = j - 1
                temp_row.append(temp_sum_of_xi)
                temp_sum_of_xi = 0
            j = d
            a_matrix.append(temp_row)

        while j >= 0:
            y_vector_in_position_i = 0
            for i in range(len(x_values)):
                y_vector_in_position_i += y_values[i] * x_to_the_power_of_i(x_values[i], j)
            y_vector.append(y_vector_in_position_i)
            j = j - 1

        y_vector = np.array(y_vector)
        a_inv = invert_mat(a_matrix)
        coef = np.dot(a_inv, y_vector)
        self.coef = coef

        def y(x):
            result = 0
            for r, c in enumerate(coef):
                # print(c)
                result = result + c * np.power(x, len(coef) - r - 1)
            return result

        return y
