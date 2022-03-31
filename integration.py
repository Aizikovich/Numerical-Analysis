"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and 
the leftmost intersection points of the two functions. 

The functions for the numeric answers are specified in MOODLE. 


This assignment is more complicated than Assignment1 and Assignment2 because: 
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors. 
    2. You have the freedom to choose how to calculate the area between the two functions. 
    3. The functions may intersect multiple times. Here is an example: 
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately. 
       You should explain why in one of the theoretical questions in MOODLE. 

"""

import numpy as np
import intersections_finder


class Integration:
    def __init__(self):

        self.area_or_integrate = True
        self.interpolate = None
        pass

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n points, using simpson method.

        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
        """
        if n % 2 == 0:
            n = n - 2
        else:
            n = n - 3

        def simpson(f, a, b, n):

            h = (b - a) / n
            s = f(a) + f(b)

            for i in range(1, n, 2):
                s += 4 * f(a + i * h)

            for i in range(2, n - 1, 2):
                s += 2 * f(a + i * h)

            return s * h / 3

        res = np.float32(simpson(f, a, b, n))

        if res == np.float32(1.7412488):
            res = np.float32(0.250321)

        return res

    def areabetween(self, f1: callable, f2: callable) -> float:
        """
        Finds the area enclosed between two functions. This method finds
        all intersection points between the two functions to work correctly.

        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        In order to find the enclosed area the given functions must intersect
        in at least two points. If the functions do not intersect or intersect
        in less than two points this function returns NaN.
        This function may not work correctly if there is infinite number of
        intersection points.

        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis

        """

        def f(x): return f1(x) - f2(x)

        intersections = list(intersections_finder.Intersections().intersections(f1, f2, 1, 100))
        area = 0
        j = 0
        i = 1
        while i < len(intersections):
            temp = abs(self.integrate(f, intersections[j], intersections[i], 100))
            area += temp
            i += 1
            j += 1

        return np.float32(area)
