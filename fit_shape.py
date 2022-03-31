from sklearn.cluster import KMeans
import numpy as np
from functionUtils import AbstractShape


class FitShape:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """
        self.sample_data = None
        self.polygon_area = 0

    def radial_sort_line(self, x, y):
        """Sort unordered verts of an unclosed line by angle from their center."""
        # Radial sort
        x0, y0 = x.mean(), y.mean()
        angle = np.arctan2(y - y0, x - x0)

        idx = angle.argsort()
        x, y = x[idx], y[idx]

        # Split at opening in line
        dx = np.diff(np.append(x, x[-1]))
        dy = np.diff(np.append(y, y[-1]))
        max_gap = np.abs(np.hypot(dx, dy)).argmax() + 1

        x = np.append(x[max_gap:], x[:max_gap])
        y = np.append(y[max_gap:], y[:max_gap])
        return x, y

    def area(self, contour: callable, maxerr=0.001) -> np.float32:
        """
        Compute the area of the shape with the given contour.

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """

        def area_polygon(p):
            return 0.5 * abs(sum(x0 * y1 - x1 * y0 for ((x0, y0), (x1, y1)) in segments(p)))

        def segments(p):
            return zip(p, p[1:] + [p[0]])

        shape = [p for p in contour(100)]

        res = area_polygon(shape)

        # print(f'the area is {res}')
        return np.float32(res)

    def generate_data(self, sample, n=10000) -> tuple:
        x_values = []
        y_values = []
        for i in range(n):
            x, y = sample()
            x_values.append(x)
            y_values.append(y)
        self.sample_data = x_values, y_values
        return x_values, y_values

    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        """
        Build a function that accurately fits the noisy data points sampled from
        some closed shape.

        Parameters
        ----------
        sample : callable.
            An iterable which returns a data point that is near the shape contour.
        maxtime : float
            This function returns after at most maxtime seconds.

        Returns
        -------
        An object extending AbstractShape.
        """

        class MyShape(AbstractShape):
            # change this class with anything you need to implement the shape
            def __init__(self, area, countor):
                # super(MyShape, self).__init__()
                self.t_area = area
                self.t_contour = countor
                pass

            def contour(self, n):
                return

            def area(self) -> np.float32:
                return np.float32(self.t_area)

        cluster = []
        x_value, y_value = np.array(self.generate_data(sample))
        for i in range(len(x_value)):
            p = [x_value[i], y_value[i]]
            cluster.append(p)
        cluster = np.array(cluster)
        k_means = KMeans(n_clusters=18, random_state=0).fit(cluster).cluster_centers_
        # print(k_means)
        xk = np.array([x[0] for x in k_means])
        yk = np.array([y[1] for y in k_means])

        # self.plot_shape(xk, yk)
        x_sort, y_sort = self.radial_sort_line(xk, yk)

        def make_contour(n=20):
            p = []
            for j in range(len(x_sort)):
                xj, yj = x_sort[j], y_sort[j]
                p.append((xj, yj))
            return p

        area = self.area(make_contour)

        # replace these lines with your solution
        result = MyShape(area, make_contour)

        return result
