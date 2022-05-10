import numpy as np


class Interpolation:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        starting to interpolate arbitrary functions.
        """

        pass

    def interpolate(self, f: callable, a: float, b: float, n: int) -> callable:
        """
        This function make interpolation using bezier curves using given f n times.

        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        n : int
            maximal number of points to use.

        Returns
        -------
        The interpolating function.
        """

        def bezier_cubic(p1, p2, p3, p4):
            matrix = np.array(
                [[-1, +3, -3, +1],
                 [+3, -6, +3, 0],
                 [-3, +3, 0, 0],
                 [+1, 0, 0, 0]],
                dtype=np.float32
            )
            points = np.array([p1, p2, p3, p4], dtype=np.float32)

            def function_get_t(t):
                t = np.array([t ** 3, t ** 2, t, 1], dtype=np.float32)
                return t.dot(matrix).dot(points)

            return function_get_t

        def bezier2(P1, P2, P3):
            M = np.array(
                [[1, -2, 1],
                 [-2, 2, 0],
                 [1, 0, 0]],
                dtype=np.float32
            )
            P = np.array([P1, P2, P3], dtype=np.float32)

            def f(t):
                T = np.array([t ** 2, t, 1], dtype=np.float32)
                return T.dot(M).dot(P)

            return f

        if n == 1:
            return lambda x: f(x)
        range_ab = b - a
        if n == 2:  # return liner function between 2 point-
            def g(x):
                y1 = f(a)
                y2 = f(b)
                m = (y1 - y2) / (a - b)
                c = y1 - m * a
                return x * m + c

            return g

        if n == 3:
            x_values = np.linspace(a, b, n, endpoint=True)
            p = [(x, f(x)) for x in x_values]
            curve = bezier2(p[0], p[1], p[2])

            def g(x):
                normalized_x = (x - a) / (b - a)
                return curve(normalized_x)

            return g
        # Can use only n samples
        n = n // 2

        # find derivatives in given point and function
        def d(f_1, y):
            h = 1e-10
            return lambda x: (f_1(x + h) - y) / h

        #  spreading 2D linear space given x in [a, b]
        x_values = np.linspace(a, b, n, endpoint=True)
        #  give us the y value for every x
        y_values = np.array(list(map(f, x_values)))
        #  give us the derivatives for each point
        d_values = np.array([d(f, y_values[x])(x_values[x]) for x in range(n)])

        all_bezier = {}

        difference = abs(x_values[0] - x_values[1])
        third_difference = difference / 3

        for i in range(n - 1):
            """
            for loop that generate for each pair of points in x_values to points in th direction of the given f
            and build bezier cubic for this for points
            """
            p_0 = (x_values[i], y_values[i])
            # p_1 <- (x,y) such that : x <- will be x_0 + (x_0 - x_1)/3 && y <- f(x_0) + d(f(x_0))*(x_0 - x_1)/3
            p_1 = (x_values[i] + third_difference, y_values[i] + d_values[i] * third_difference)
            # p_3 <- (x_1,y_1)
            p_3 = (x_values[i + 1], y_values[i + 1])
            # p_2 <- (x,y) such that : x <- will be x_1 - (x_0 - x_1)/3 && y <- f(x_1) - d(f(x_1))*(x_0 - x_1)/3
            p_2 = (x_values[i + 1] - third_difference, y_values[i + 1] - d_values[i + 1] * third_difference)

            # Update all_bezier dict such that: { key=i : value = bezier cubic for (P_(i) && P_(i+1) }
            all_bezier[i] = bezier_cubic(p_0, p_1, p_2, p_3)


        def g(x):
            """
            The interpolate function define as (g: R -> R).
            To use the bizer curves we found in the for loop. Find which curve to use (bezier_i) to given x,
            afterwards normalized x to the interval [0,1] .
            """
            # Find which bezier to use
            index = int(((x - a) / range_ab) * (n - 1))

            # Normalized x using: New value = (value – min - index*range) / (range)
            normalized_x = (x - a - index * difference) / difference
            if index == n - 1:
                return all_bezier[n - 2](1)[1]
            return all_bezier[index](normalized_x)[1]

        return g
