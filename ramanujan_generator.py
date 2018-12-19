"""
Solution with recursive python generator for SICP exercise 3.71
This is the fastest one.
There are two other versions:
-one in ch35_streams.py based on streams
-one in ramanujan.py which is a naive implementation and checks every int
"""

import itertools


def integers_from(n):
    while True:
        yield n
        n = n + 1


def interleave(g1, g2):
    while True:
        yield next(g1)
        g1, g2 = g2, g1


def generator_map(f, g):
    while True:
        yield (f(next(g)))


def generator_filter(f, g):
    while True:
        i = next(g)
        if f(i):
            yield i


def pairs(g1, g2):
    a, b = next(g1), next(g2)
    yield (a, b)
    g21, g22 = itertools.tee(g2)
    i = interleave(
        generator_map(lambda x: (a, x), g21),
        pairs(g1, g22)
    )
    while True:
        yield next(i)


def merge_weighted(g1, g2, weight):
    a, b = next(g1), next(g2)
    while True:
        if weight(a) <= weight(b):
            yield a
            a = next(g1)
        else:
            yield b
            b = next(g2)


def weighted_pairs(g1, g2, weight):
    a, b = next(g1), next(g2)
    yield (weight(a, b), a, b)
    g21, g22 = itertools.tee(g2)
    i = merge_weighted(
        generator_map(lambda x: (weight(a, x), a, x), g21),
        weighted_pairs(g1, g22, weight),
        lambda x: x[0]
    )
    while True:
        yield next(i)


def merge_with_next_item(g):
    a = next(g)
    while True:
        a, b = next(g), a
        yield (a, b)


def ramanujan_numbers():
    g1 = weighted_pairs(integers_from(0), integers_from(0), lambda x, y: x ** 3 + y ** 3)
    g2 = merge_with_next_item(g1)
    return generator_filter(lambda x: x[0][0] == x[1][0], g2)
