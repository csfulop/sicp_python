"""
Random exercises for SICP Structure and Interpretation of Computer Programs /  chapter 3.5 - Streams
"""

import math
import operator
import time


def generator_filter(filter, generator):
    for i in generator:
        if filter(i):
            yield i


def fibs_generator():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


# ------------------------------------------------------------------------------


# SICP 3.5.1

EMPTY_STREAM = ()


def _memo(f):
    class content:
        result = {}

    def wrapper(*args):
        if not args in content.result:
            content.result[args] = f(*args)
        return content.result[args]

    return wrapper


def cons_stream(x, y):
    assert callable(y)
    return x, _memo(y)
    # return x, y


class EndOfStream(Exception):
    pass


def stream_car(stream):
    if stream == EMPTY_STREAM:
        raise EndOfStream
    return stream[0]


def stream_cdr(stream):
    if stream == EMPTY_STREAM:
        raise EndOfStream
    return stream[1]()


def stream_enumerate_interval(low, high):
    if low > high:
        return EMPTY_STREAM
    else:
        return cons_stream(low, lambda: stream_enumerate_interval(low + 1, high))


def stream_ref(stream, n):
    if n == 0:
        return stream_car(stream)
    else:
        return stream_ref(stream_cdr(stream), n - 1)


def stream_map(f, s):
    if s == EMPTY_STREAM:
        return EMPTY_STREAM
    else:
        # print('stream_map(%s)'%stream_car(s))
        return cons_stream(
            f(stream_car(s)),
            lambda: stream_map(f, stream_cdr(s))
        )


def stream_filter(f, s):
    if s == EMPTY_STREAM:
        return EMPTY_STREAM
    elif f(stream_car(s)):
        return cons_stream(
            stream_car(s),
            lambda: stream_filter(f, stream_cdr(s))
        )
    else:
        return stream_filter(f, stream_cdr(s))


# SICP 3.5.2

def integers_from(n):
    return cons_stream(
        n,
        lambda: integers_from(n + 1)
    )


def primes_stream():
    def not_divisible(x, y):
        # print('  %d %% %d' % (x, y))
        return x % y != 0

    def sieve(s):
        return cons_stream(
            stream_car(s),
            lambda: sieve(
                stream_filter(
                    lambda x: not_divisible(x, stream_car(s)),
                    stream_cdr(s)
                )
            )
        )

    return sieve(integers_from(2))


def fibs():
    def fibgen(a, b):
        return cons_stream(
            a,
            lambda: fibgen(b, a + b)
        )

    return fibgen(0, 1)


def ones():
    return cons_stream(
        1,
        lambda: ones()
    )


def zeros():
    return cons_stream(
        0,
        lambda: zeros()
    )


def add_streams(s1, s2):
    # print('%d + %d' % (stream_car(s1), stream_car(s2)))
    return cons_stream(
        stream_car(s1) + stream_car(s2),
        lambda: add_streams(stream_cdr(s1), stream_cdr(s2))
    )


@_memo
def integers():
    return cons_stream(
        1,
        lambda: add_streams(ones(), integers())
    )


@_memo
def fibs2():
    return cons_stream(
        0,
        lambda: cons_stream(
            1,
            lambda: add_streams(
                fibs2(),
                stream_cdr(fibs2())
            )
        )
    )


def scale_stream(s, factor):
    # print('%d * %d' % (stream_car(s), factor))
    return cons_stream(
        stream_car(s) * factor,
        lambda: scale_stream(stream_cdr(s), factor)
    )


@_memo
def double():
    return cons_stream(
        1,
        lambda: scale_stream(double(), 2)
    )


@_memo
def primes_stream_2():
    return cons_stream(
        2,
        lambda: stream_filter(is_prime, integers_from(3))
    )


def is_prime(n):
    # print('is_prime(%d)'%n)
    def iter(ps):
        if stream_car(ps) ** 2 > n:
            return True
        elif n % stream_car(ps) == 0:
            return False
        else:
            return iter(stream_cdr(ps))

    return iter(primes_stream_2())


# SICP 3.54
def mul_streams(s1, s2):
    # print('%d * %d' % (stream_car(s1), stream_car(s2)))
    return cons_stream(
        stream_car(s1) * stream_car(s2),
        lambda: mul_streams(stream_cdr(s1), stream_cdr(s2))
    )


# SICP 3.54
@_memo
def factorials():
    return cons_stream(
        1,
        lambda: mul_streams(factorials(), integers_from(2))
    )


# SICP 3.55
def partial_sums(s):
    @_memo
    def ps():
        return add_streams(s, cons_stream(0, lambda: ps()))

    return ps()


# SICP 3.56
@_memo
def only_prime_factors_2_3_5():
    return cons_stream(
        1,
        lambda: merge(
            scale_stream(only_prime_factors_2_3_5(), 2),
            merge(
                scale_stream(only_prime_factors_2_3_5(), 3),
                scale_stream(only_prime_factors_2_3_5(), 5)
            )
        )
    )


def merge(s1, s2):
    if s1 == EMPTY_STREAM:
        return s2
    if s2 == EMPTY_STREAM:
        return s1
    else:
        s1car = stream_car(s1)
        s2car = stream_car(s2)
        if s1car < s2car:
            return cons_stream(s1car, lambda: merge(stream_cdr(s1), s2))
        if s2car < s1car:
            return cons_stream(s2car, lambda: merge(s1, stream_cdr(s2)))
        else:
            return cons_stream(s1car, lambda: merge(stream_cdr(s1), stream_cdr(s2)))


# SICP 3.59
def integrate_series(s):
    return mul_streams(
        stream_map(lambda x: 1 / x, integers_from(1)),
        s
    )


@_memo
def exp_series():
    return cons_stream(
        1,
        lambda: integrate_series(exp_series())
    )


@_memo
def cosine_series():
    return cons_stream(
        1,
        lambda: integrate_series(scale_stream(sine_series(), -1))
    )


@_memo
def sine_series():
    return cons_stream(
        0,
        lambda: integrate_series(cosine_series())
    )


# SICP 3.60
def mul_series(s1, s2):
    return cons_stream(
        stream_car(s1) * stream_car(s2),
        lambda: add_streams(
            mul_series(stream_cdr(s1), s2),
            scale_stream(stream_cdr(s2), stream_car(s1))
        )
    )


# SICP 3.61
def invert_unit_series(s):
    pass


# SICP 3.5.3 - Formulating iterations as stream processes
def average(a, b):
    return (a + b) / 2


def sqrt_improve(guess, x):
    return average(guess, x / guess)


def sqrt_stream(x):
    @_memo
    def guesses():  # FIXME: this is a variable
        return cons_stream(
            1.0,
            lambda: stream_map(lambda guess: sqrt_improve(guess, x), guesses())
        )

    return guesses()


# PI/4 = 1 - 1/3 + 1/5 - 1/7 + ...

def _pi_summands(n):
    return cons_stream(
        1 / n,
        lambda: stream_map(operator.neg, _pi_summands(n + 2))
    )


def pi_stream():
    return scale_stream(partial_sums(_pi_summands(1)), 4)


def euler_transform(s):
    s0 = stream_ref(s, 0)
    s1 = stream_ref(s, 1)
    s2 = stream_ref(s, 2)
    return cons_stream(
        s2 - (s2 - s1) ** 2 / (s0 - 2 * s1 + s2),
        lambda: euler_transform(stream_cdr(s))
    )


def make_tableau(transform, s):
    return cons_stream(
        s,
        lambda: make_tableau(transform, transform(s))
    )


def accelerated_sequence(transform, s):
    return stream_map(stream_car, make_tableau(transform, s))


# SICP 3.64
def stream_limit(s, tolerance):
    s1 = stream_ref(s, 0)
    s2 = stream_ref(s, 1)
    # print('stream_limit(%f,%f)' % (s1, s2))
    if abs(s1 - s2) < tolerance:
        return s2
    else:
        return stream_limit(stream_cdr(s), tolerance)


def sqrt(x, tolerance):
    return stream_limit(sqrt_stream(x), tolerance)


# SICP 3.65
def _ln2_summands(n):
    return cons_stream(
        1 / n,
        lambda: stream_map(operator.neg, _ln2_summands(n + 1))
    )


def ln2_stream():
    return partial_sums(_ln2_summands(1))


# SICP 3.5.3 - Infinite streams of pairs

def interleave(s1, s2):
    if s1 == EMPTY_STREAM:
        return s2
    else:
        return cons_stream(
            stream_car(s1),
            lambda: interleave(s2, stream_cdr(s1))
        )


def pairs(s, t):
    return cons_stream(
        (stream_car(s), stream_car(t)),
        lambda: interleave(
            stream_map(lambda x: (stream_car(s), x), stream_cdr(t)),
            pairs(stream_cdr(s), stream_cdr(t))
        )
    )


# SICP 3.70
def merge_weighted(s1, s2, weight):
    if s1 == EMPTY_STREAM:
        return s2
    if s2 == EMPTY_STREAM:
        return s1
    else:
        s1car = stream_car(s1)
        s2car = stream_car(s2)
        if weight(s1car) <= weight(s2car):
            return cons_stream(s1car, lambda: merge_weighted(stream_cdr(s1), s2, weight))
        else:
            return cons_stream(s2car, lambda: merge_weighted(s1, stream_cdr(s2), weight))


def weighted_pairs(s1, s2, weight):
    s1car = stream_car(s1)
    s2car = stream_car(s2)
    return cons_stream(
        (weight(s1car, s2car), s1car, s2car),
        lambda: merge_weighted(
            stream_map(lambda x: (weight(s1car, x), s1car, x), stream_cdr(s2)),
            weighted_pairs(stream_cdr(s1), stream_cdr(s2), weight),
            lambda x: x[0]
        )
    )


def map_stream_with_next_item(s):
    return cons_stream(
        (stream_car(s), stream_car(stream_cdr(s))),
        lambda: map_stream_with_next_item(stream_cdr(s))
    )


# SICP 3.71
# FIXME: is memoization needed for this calculation? Without it maybe it would use less memory... NO
def ramanujan_numbers():
    s1 = weighted_pairs(integers_from(0), integers_from(0), lambda x, y: x ** 3 + y ** 3)
    s2 = map_stream_with_next_item(s1)
    return stream_filter(lambda x: x[0][0] == x[1][0], s2)


# SICP 3.5.5
RANDOM_INIT = int(time.time())

# MULTIPLIER = 6364136223846793005
# INCREMENT = 1442695040888963407
# MODULUS = 2 ** 64

MULTIPLIER = 6364136223846793005
INCREMENT = 1
MODULUS = 2 ** 64


# MULTIPLIER = 1103515245
# INCREMENT = 12345
# MODULUS = 2 ** 31

def rand_update(n):
    return ((n * MULTIPLIER + INCREMENT) % MODULUS) >> 32


@_memo
def random_numbers(seed=RANDOM_INIT):
    return cons_stream(
        seed,
        lambda: stream_map(rand_update, random_numbers(seed))
    )


# @_memo
# def random_numbers(seed=RANDOM_INIT):
#     return cons_stream(
#         random.getrandbits(64),
#         lambda: random_numbers()
#     )


def gcd(a, b):
    if b == 0:
        return a
    else:
        return gcd(b, a % b)


def map_successive_pairs(f, s):
    return cons_stream(
        f(stream_car(s), stream_car(stream_cdr(s))),
        lambda: map_successive_pairs(f, stream_cdr(stream_cdr(s)))
    )


def cesaro_stream(seed=RANDOM_INIT):
    def are_relative_primes(a, b):
        return gcd(a, b) == 1

    s = map_successive_pairs(
        are_relative_primes,
        random_numbers(seed)
    )
    return s


def monte_carlo(experiment_stream, passed=0, failed=0):
    def next(passed, failed):
        return cons_stream(
            passed / (passed + failed),
            lambda: monte_carlo(stream_cdr(experiment_stream), passed, failed)
        )

    if stream_car(experiment_stream):
        return next(passed + 1, failed)
    else:
        return next(passed, failed + 1)


def pi_monte_carlo():
    return stream_map(lambda p: math.sqrt(6 / p) if p != 0. else 0., monte_carlo(cesaro_stream()))
