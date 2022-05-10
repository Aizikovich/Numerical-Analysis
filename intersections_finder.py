import copy
import math
import time
import numpy as np
import random
from collections.abc import Iterable


class Intersections:
    def __init__(self):
        self.memo = []
        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        This functions find all intersections between f1 and f2, for both f1 & f2 are continuous functions.
        This function  not work correctly if there is infinite number of intersection points.

        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """

        def f(x):
            return f1(x) - f2(x)

        def bisection(left, right):
            current_root = (left + right) / 2
            while abs(f(current_root)) > maxerr:
                if f(left) * f(current_root) < 0:
                    right = current_root
                else:
                    left = current_root
                current_root = (left + right) / 2
            return current_root

        def secant(x0, x1, tol=maxerr, max_iterations=100):
            left = copy.copy(x0)
            right = copy.copy(x1)
            # store y values instead of recomputing them
            fx0 = f(x0)
            fx1 = f(x1)

            # iterate up to maximum number of times
            for _ in range(max_iterations):
                # see whether the answer has converged
                if abs(fx1) < tol and left <= x1 <= right:
                    return x1
                # do calculation
                try:
                    x2 = (x0 * fx1 - x1 * fx0) / (fx1 - fx0)
                    # shift variables (prepare for next loop)
                except RuntimeError:
                    return None
                x0, x1 = x1, x2
                try:
                    fx0, fx1 = fx1, f(x2)
                except ValueError:
                    break
            return None

        result = [a] if abs(f(a)) <= maxerr else []
        l, r = a, a + 0.01

        while r <= b:
            fl = f(l)
            fr = f(r)
            if f(l) * f(r) < 0:  # bisection
                root = bisection(l, r)
                result.append(root)
                l, r = r, r + 0.05
            else:  # try secant
                s_root = secant(l, r)
                if s_root is None:
                    r = r + 0.05
                else:  # if root is not None:
                    result.append(s_root)
                    l = s_root + 0.05
                    r = r + 0.05

        if abs(f(b)) <= maxerr:
            result.append(b)

        return result


