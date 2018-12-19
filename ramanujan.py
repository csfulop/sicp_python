"""
Naive implementation to find Ramanujan numbers (SICP exercise 3.71)
This version checks every int so this is the slowest one.
There are two other versions:
-one in ch35_streams.py based on streams
-one in ramanujan_generator.py which is a recursive python generator version
"""

def _memo(f):
    class content:
        result = {}

    def wrapper(n):
        if n not in content.result:
            content.result[n] = f(n)
        return content.result[n]

    return wrapper


@_memo
def _rounded_cube(n):
    return round(n ** (1. / 3))


@_memo
def _is_cube_number(n):
    n = abs(n)
    return _rounded_cube(n) ** 3 == n


def _is_ramanujan(n):
    sums = []
    for i in range(1, _rounded_cube(n)):
        j = _rounded_cube(n - i ** 3)
        if j <= i:
            return False, sums
        if _is_cube_number(n - i ** 3):
            sums.append((i, j))
            if len(sums) > 1:
                return True, sums
    return False, sums


def ramanujans():
    i = 1;
    while True:
        result, sums = _is_ramanujan(i)
        if result:
            yield i, sums
        i = i + 1
