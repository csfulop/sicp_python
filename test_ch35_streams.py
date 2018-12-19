import math
from unittest import TestCase

from ch35_streams import generator_filter, fibs_generator, cons_stream, stream_car, stream_cdr, EndOfStream, \
    EMPTY_STREAM, stream_enumerate_interval, stream_ref, stream_map, stream_filter, integers_from, primes_stream, \
    fibs, ones, add_streams, integers, fibs2, scale_stream, double, primes_stream_2, mul_streams, factorials, \
    partial_sums, only_prime_factors_2_3_5, integrate_series, exp_series, cosine_series, sine_series, zeros, \
    mul_series, sqrt_stream, _pi_summands, pi_stream, euler_transform, accelerated_sequence, sqrt, _ln2_summands, \
    ln2_stream, pairs, merge_weighted, weighted_pairs, ramanujan_numbers, random_numbers, gcd, pi_monte_carlo, \
    cesaro_stream, monte_carlo, map_successive_pairs


class TestGenerators(TestCase):
    def test_generator_filter(self):
        f = lambda x: x % 7 != 0
        self.assertEqual(list(generator_filter(f, range(0, 20))),
                         [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19])

    def test_fibs_generator(self):
        f = fibs_generator()
        self.assertEqual(next(f), 0)
        self.assertEqual(next(f), 1)
        self.assertEqual(next(f), 1)
        self.assertEqual(next(f), 2)
        self.assertEqual(next(f), 3)
        self.assertEqual(next(f), 5)
        self.assertEqual(next(f), 8)
        self.assertEqual(next(f), 13)
        self.assertEqual(next(f), 21)
        self.assertEqual(next(f), 34)


class TestStreams(TestCase):
    def test_stream_cons_car_cdr(self):
        s = cons_stream(1, lambda: 2)
        self.assertEqual(stream_car(s), 1)
        self.assertEqual(stream_cdr(s), 2)

    def test_end_of_stream(self):
        with self.assertRaises(EndOfStream):
            stream_car(EMPTY_STREAM)
        with self.assertRaises(EndOfStream):
            stream_cdr(EMPTY_STREAM)

    def test_cdr_is_memoized(self):
        class content:
            runs = 0

        def f(x):
            def g():
                content.runs += 1
                return x

            return g

        s = cons_stream(1, f(2))
        self.assertEqual(stream_car(s), 1)
        self.assertEqual(stream_cdr(s), 2)
        self.assertEqual(stream_cdr(s), 2)
        self.assertEqual(content.runs, 1)

    def test_stream_enumerate(self):
        s = stream_enumerate_interval(5, 10)
        for i in range(5, 11):
            self.assertEqual(stream_car(s), i)
            s = stream_cdr(s)
        self.assertEqual(s, EMPTY_STREAM)

    def test_stream_ref(self):
        s = stream_enumerate_interval(5, 20)
        for i in range(5, 21):
            self.assertEqual(stream_ref(s, i - 5), i)
        with self.assertRaises(EndOfStream):
            stream_ref(s, 16)

    def test_stream_map(self):
        f = lambda x: x + 1
        s1 = stream_enumerate_interval(0, 10)
        s2 = stream_map(f, s1)
        for i in range(0, 11):
            self.assertEqual(stream_car(s2), i + 1)
            s2 = stream_cdr(s2)
        self.assertEqual(s2, EMPTY_STREAM)

    def test_stream_filter(self):
        f = lambda x: x % 2 == 0
        s1 = stream_enumerate_interval(0, 11)
        s2 = stream_filter(f, s1)
        for i in range(0, 6):
            self.assertEqual(stream_car(s2), 2 * i)
            s2 = stream_cdr(s2)
        self.assertEqual(s2, EMPTY_STREAM)

    def test_integers_from(self):
        s = integers_from(1000)
        for i in range(1000, 1020):
            self.assertEqual(stream_car(s), i)
            s = stream_cdr(s)

    def test_primes(self):
        primes = primes_stream()
        self.assertEqual(stream_car(primes), 2)
        primes = stream_cdr(primes)
        self.assertEqual(stream_car(primes), 3)
        primes = stream_cdr(primes)
        self.assertEqual(stream_car(primes), 5)
        primes = stream_cdr(primes)
        self.assertEqual(stream_car(primes), 7)
        primes = stream_cdr(primes)
        self.assertEqual(stream_car(primes), 11)
        primes = stream_cdr(primes)
        self.assertEqual(stream_car(primes), 13)
        primes = stream_cdr(primes)
        self.assertEqual(stream_car(primes), 17)
        primes = stream_cdr(primes)
        self.assertEqual(stream_car(primes), 19)
        primes = stream_cdr(primes)
        self.assertEqual(stream_car(primes), 23)

    def test_1001_as_sum_of_two_primes(self):
        def primes_up_to(n):
            s = primes_stream()
            result = []
            while True:
                i = stream_car(s)
                result.append(i)
                s = stream_cdr(s)
                if i >= n:
                    break
            return result

        n = 1001
        primes = primes_up_to(n)
        print(primes)
        for p in primes:
            if (n - p) in primes:
                print("%d = %d + %d" % (n, p, n - p))

    def test_fibs(self):
        f = fibs()
        self.assertEqual(stream_car(f), 0)
        f = stream_cdr(f)
        self.assertEqual(stream_car(f), 1)
        f = stream_cdr(f)
        self.assertEqual(stream_car(f), 1)
        f = stream_cdr(f)
        self.assertEqual(stream_car(f), 2)
        f = stream_cdr(f)
        self.assertEqual(stream_car(f), 3)
        f = stream_cdr(f)
        self.assertEqual(stream_car(f), 5)
        f = stream_cdr(f)
        self.assertEqual(stream_car(f), 8)
        f = stream_cdr(f)
        self.assertEqual(stream_car(f), 13)
        f = stream_cdr(f)
        self.assertEqual(stream_car(f), 21)
        f = stream_cdr(f)
        self.assertEqual(stream_car(f), 34)

    def test_ones(self):
        s = ones()
        self.assertEqual(stream_ref(s, 0), 1)
        self.assertEqual(stream_ref(s, 10), 1)
        self.assertEqual(stream_ref(s, 100), 1)

    def test_add_streams(self):
        s1 = ones()
        s2 = integers_from(0)
        s3 = add_streams(s1, s2)
        for i in range(0, 10):
            self.assertEqual(stream_ref(s3, i), i + 1)

    def test_integers(self):
        i = integers()
        self.assertEqual(stream_ref(i, 0), 1)
        self.assertEqual(stream_ref(i, 1), 2)
        self.assertEqual(stream_ref(i, 10), 11)
        self.assertEqual(stream_ref(i, 100), 101)

    def test_fibs2(self):
        f = fibs2()
        self.assertEqual(stream_ref(f, 0), 0)
        self.assertEqual(stream_ref(f, 1), 1)
        self.assertEqual(stream_ref(f, 2), 1)
        self.assertEqual(stream_ref(f, 3), 2)
        self.assertEqual(stream_ref(f, 4), 3)
        self.assertEqual(stream_ref(f, 5), 5)
        self.assertEqual(stream_ref(f, 6), 8)
        self.assertEqual(stream_ref(f, 7), 13)
        self.assertEqual(stream_ref(f, 8), 21)
        self.assertEqual(stream_ref(f, 9), 34)

    def test_scale_stream(self):
        s = scale_stream(integers(), 3)
        for i in range(1, 10):
            self.assertEqual(stream_car(s), i * 3)
            s = stream_cdr(s)

    def test_double(self):
        s = double()
        self.assertEqual(stream_ref(s, 0), 1)
        self.assertEqual(stream_ref(s, 1), 2)
        self.assertEqual(stream_ref(s, 2), 4)
        self.assertEqual(stream_ref(s, 3), 8)
        self.assertEqual(stream_ref(s, 4), 16)
        self.assertEqual(stream_ref(s, 5), 32)
        self.assertEqual(stream_ref(s, 6), 64)

    def test_primes_2(self):
        primes = primes_stream_2()
        self.assertEqual(stream_car(primes), 2)
        primes = stream_cdr(primes)
        self.assertEqual(stream_car(primes), 3)
        primes = stream_cdr(primes)
        self.assertEqual(stream_car(primes), 5)
        primes = stream_cdr(primes)
        self.assertEqual(stream_car(primes), 7)
        primes = stream_cdr(primes)
        self.assertEqual(stream_car(primes), 11)
        primes = stream_cdr(primes)
        self.assertEqual(stream_car(primes), 13)
        primes = stream_cdr(primes)
        self.assertEqual(stream_car(primes), 17)
        primes = stream_cdr(primes)
        self.assertEqual(stream_car(primes), 19)
        primes = stream_cdr(primes)
        self.assertEqual(stream_car(primes), 23)

    def test_mul_streams(self):
        s1 = integers_from(1)
        s2 = integers_from(2)
        self.assertEqual(stream_ref(mul_streams(s1, s2), 0), 2)
        self.assertEqual(stream_ref(mul_streams(s1, s2), 1), 6)
        self.assertEqual(stream_ref(mul_streams(s1, s2), 2), 12)
        self.assertEqual(stream_ref(mul_streams(s1, s2), 3), 20)

    def test_factorials(self):
        f = factorials()
        self.assertEqual(stream_ref(f, 0), 1)
        self.assertEqual(stream_ref(f, 1), 2)
        self.assertEqual(stream_ref(f, 2), 6)
        self.assertEqual(stream_ref(f, 3), 24)
        self.assertEqual(stream_ref(f, 4), 120)

    def test_partial_sums(self):
        s = partial_sums(integers())
        self.assertEqual(stream_ref(s, 0), 1)
        self.assertEqual(stream_ref(s, 1), 3)
        self.assertEqual(stream_ref(s, 2), 6)
        self.assertEqual(stream_ref(s, 3), 10)
        self.assertEqual(stream_ref(s, 4), 15)
        self.assertEqual(stream_ref(s, 5), 21)

    def test_only_prime_factors_2_3_5(self):
        s = only_prime_factors_2_3_5()
        self.assertEqual(stream_car(s), 1)
        s = stream_cdr(s)
        self.assertEqual(stream_car(s), 2)
        s = stream_cdr(s)
        self.assertEqual(stream_car(s), 3)
        s = stream_cdr(s)
        self.assertEqual(stream_car(s), 4)
        s = stream_cdr(s)
        self.assertEqual(stream_car(s), 5)
        s = stream_cdr(s)
        self.assertEqual(stream_car(s), 6)
        s = stream_cdr(s)
        self.assertEqual(stream_car(s), 8)
        s = stream_cdr(s)
        self.assertEqual(stream_car(s), 9)
        s = stream_cdr(s)
        self.assertEqual(stream_car(s), 10)
        s = stream_cdr(s)
        self.assertEqual(stream_car(s), 12)
        s = stream_cdr(s)
        self.assertEqual(stream_car(s), 15)
        s = stream_cdr(s)
        self.assertEqual(stream_car(s), 16)
        s = stream_cdr(s)
        self.assertEqual(stream_car(s), 18)
        s = stream_cdr(s)
        self.assertEqual(stream_car(s), 20)
        s = stream_cdr(s)
        self.assertEqual(stream_car(s), 24)
        s = stream_cdr(s)
        self.assertEqual(stream_car(s), 25)
        s = stream_cdr(s)

    def test_integrate_seriese(self):
        s1 = ones()
        s2 = integrate_series(s1)
        for i in range(1, 20):
            self.assertEqual(stream_ref(s2, i - 1), 1.0 / i)

    def test_exp_series(self):
        for x in range(1, 20):
            s = mul_streams(
                exp_series(),
                stream_map(lambda n: x ** n, integers_from(0))
            )

            result = stream_ref(partial_sums(s), 60)
            expected = math.exp(x)
            # print('%d: %f =?= %f' % (x, result, expected))
            self.assertAlmostEqual(result, expected, delta=0.1)

    def test_cosine_series(self):
        c = cosine_series()

        self.assertAlmostEqual(stream_ref(c, 0), 1, places=10)
        self.assertAlmostEqual(stream_ref(c, 1), 0., places=10)
        self.assertAlmostEqual(stream_ref(c, 2), -1. / 2, places=10)
        self.assertAlmostEqual(stream_ref(c, 3), 0., places=10)
        self.assertAlmostEqual(stream_ref(c, 4), 1. / (2 * 3 * 4), places=10)
        self.assertAlmostEqual(stream_ref(c, 5), 0., places=10)
        self.assertAlmostEqual(stream_ref(c, 6), -1. / (2 * 3 * 4 * 5 * 6), places=10)

    def test_sine_series(self):
        c = sine_series()

        self.assertAlmostEqual(stream_ref(c, 0), 0, places=10)
        self.assertAlmostEqual(stream_ref(c, 1), 1., places=10)
        self.assertAlmostEqual(stream_ref(c, 2), 0., places=10)
        self.assertAlmostEqual(stream_ref(c, 3), -1. / (2 * 3), places=10)
        self.assertAlmostEqual(stream_ref(c, 4), 0., places=10)
        self.assertAlmostEqual(stream_ref(c, 5), 1. / (2 * 3 * 4 * 5), places=10)
        self.assertAlmostEqual(stream_ref(c, 6), 0., places=10)
        self.assertAlmostEqual(stream_ref(c, 7), -1. / (2 * 3 * 4 * 5 * 6 * 7), places=10)

    def test_mul_series(self):
        a = cons_stream(1, lambda: cons_stream(2, lambda: zeros()))
        b = cons_stream(3, lambda: cons_stream(4, lambda: zeros()))

        ab = mul_series(a, b)

        expected = cons_stream(3, lambda: cons_stream(10, lambda: cons_stream(8, lambda: zeros())))

        for i in range(5):
            self.assertEqual(
                stream_ref(ab, i),
                stream_ref(expected, i)
            )

    def test_mul_series_sin2_cos2_should_be_1(self):
        s = sine_series()
        c = cosine_series()

        s2 = mul_series(s, s)
        c2 = mul_series(c, c)

        s2c2 = add_streams(s2, c2)

        self.assertEqual(stream_ref(s2c2, 0), 1)
        for i in range(1, 20):
            # print(stream_ref(s2c2, i))
            # some places have non-zero values (starting with 1e-17 and decreasing), probably some floating point errors
            # because of this using almostEqual
            self.assertAlmostEqual(stream_ref(s2c2, i), 0, delta=1e-10)

    def test_sqrt_stream(self):
        sqrt_2 = sqrt_stream(2)
        # for i in range(0,10):
        #     print(stream_ref(sqrt_2,i))
        self.assertAlmostEqual(stream_ref(sqrt_2, 10), math.sqrt(2), delta=1e-10)

    def test_pi_summands(self):
        s = _pi_summands(1)
        for i in range(0, 20):
            self.assertAlmostEqual(stream_ref(s, i), (-1) ** i * 1 / (2 * i + 1), delta=1e-10)

    def test_pi_stream(self):
        s = pi_stream()
        # for i in range(0,100):
        #     print(stream_ref(s,i))
        # print(math.pi)
        self.assertAlmostEqual(stream_ref(s, 20), math.pi, delta=0.1)

    def test_euler_pi_stream(self):
        s = euler_transform(pi_stream())
        # for i in range(0,100):
        #     print(stream_ref(s,i))
        # print(math.pi)
        self.assertAlmostEqual(stream_ref(s, 20), math.pi, delta=1e-4)

    def test_accelerated_pi_stream(self):
        s = accelerated_sequence(euler_transform, pi_stream())
        # for i in range(0, 10):
        #     print(stream_ref(s, i))
        # print(math.pi)
        self.assertAlmostEqual(stream_ref(s, 9), math.pi, delta=1e-14)

    def test_stream_limit(self):
        self.assertAlmostEqual(sqrt(2, tolerance=1e-2), math.sqrt(2), delta=1e-2)
        self.assertAlmostEqual(sqrt(2, tolerance=1e-5), math.sqrt(2), delta=1e-5)
        self.assertAlmostEqual(sqrt(2, tolerance=1e-10), math.sqrt(2), delta=1e-10)

    def test_ln2_summands(self):
        s = _ln2_summands(1)
        for i in range(0, 20):
            self.assertAlmostEqual(stream_ref(s, i), (-1) ** i * 1 / (i + 1), delta=1e-10)

    def test_ln2_stream(self):
        s = ln2_stream()
        # for i in range(0, 20):
        #     print(stream_ref(s, i))
        # print(math.log(2,math.e))
        self.assertAlmostEqual(stream_ref(s, 20), math.log(2, math.e), delta=0.1)

    def test_euler_ln2_stream(self):
        s = euler_transform(ln2_stream())
        # for i in range(0, 20):
        #     print(stream_ref(s, i))
        # print(math.log(2, math.e))
        self.assertAlmostEqual(stream_ref(s, 20), math.log(2, math.e), delta=1e-4)

    def test_accelerated_ln2_stream(self):
        s = accelerated_sequence(euler_transform, ln2_stream())
        # for i in range(0, 10):
        #     print(stream_ref(s, i))
        # print(math.log(2, math.e))
        self.assertAlmostEqual(stream_ref(s, 9), math.log(2, math.e), delta=1e-14)

    # FIXME: assert
    def test_pairs(self):
        s = pairs(integers_from(1), integers_from(1))
        for i in range(100):
            print(stream_ref(s, i))

    # FIXME: generate pairs ordered by increasing y then ordered by increasing x values
    def test_two_series(self):
        def f(i, j):
            return cons_stream(
                (i, j),
                lambda: f(0, j + 1) if i == j else f(i + 1, j)
            )

        s = f(0, 0)
        for i in range(500):
            print(stream_ref(s, i))

    def test_merge_weighted_with_integers(self):
        s = merge_weighted(integers_from(0), integers_from(0), lambda x: x)
        for i in range(20):
            self.assertEqual(stream_ref(s, 2 * i), i)
            self.assertEqual(stream_ref(s, 2 * i + 1), i)

    def test_merge_weighted_with_evens_and_odds(self):
        evens = stream_filter(lambda x: x % 2 == 0, integers_from(0))
        odds = stream_filter(lambda x: x % 2 == 1, integers_from(0))
        s = merge_weighted(evens, odds, lambda x: x)
        for i in range(20):
            self.assertEqual(stream_ref(s, i), i)

    def test_weighted_pairs(self):
        s = weighted_pairs(integers_from(0), integers_from(0), lambda x, y: x + y)
        for i in range(50):
            # print(stream_ref(s, i))
            self.assertLessEqual(stream_ref(s, i)[0], stream_ref(s, i + 1)[0])
            self.assertLessEqual(stream_ref(s, i)[1], stream_ref(s, i)[2])

    def test_weighted_pairs_with_ramanujan(self):
        s = weighted_pairs(integers_from(0), integers_from(0), lambda x, y: x ** 3 + y ** 3)
        for i in range(100):
            # print(stream_ref(s, i))
            self.assertLessEqual(stream_ref(s, i)[0], stream_ref(s, i + 1)[0])
            self.assertLessEqual(stream_ref(s, i)[1], stream_ref(s, i)[2])

    def test_ramanujan_numbers(self):
        """
            Execution results:
            Up to 10: execution time: 0.5sec
            Up to 100: execution time: 33sec; Memory usage=1.5GB

            Without memoization in cons_stream:
            Up to 100: execution time: 4min 31sec; Memory usage=30MB
        """
        s = ramanujan_numbers()
        for i in range(10):
            si = stream_ref(s, i)
            print((i, si))
            self.assertEqual(si[0][0], si[0][1] ** 3 + si[0][2] ** 3)
            self.assertEqual(si[1][0], si[1][1] ** 3 + si[1][2] ** 3)
            self.assertEqual(si[0][0], si[1][0])

    def test_random_numbers(self):
        s1 = random_numbers(0)
        s2 = random_numbers(0)
        for i in range(10):
            # print((i, stream_ref(s1, i), stream_ref(s2,i)))
            self.assertEqual(stream_car(s1), stream_car(s2))
            s1 = stream_cdr(s1)
            s2 = stream_cdr(s2)

    def test_gcd(self):
        self.assertEqual(gcd(10, 2), 2)
        self.assertEqual(gcd(2, 10), 2)
        self.assertEqual(gcd(10, 9), 1)
        self.assertEqual(gcd(64, 162), 2)

    def test_cesaro_stream(self):
        c = cesaro_stream(seed=0)
        r = random_numbers(0)
        for i in range(50):
            # print(stream_ref(s, i))
            self.assertEqual(stream_ref(c, i), gcd(stream_ref(r, 2 * i), stream_ref(r, 2 * i + 1)) == 1)

    def test_map_successive_pairs(self):
        s = map_successive_pairs(lambda x, y: (x, y), stream_map(lambda x: x % 2 == 1, integers_from(0)))
        for i in range(50):
            self.assertEqual(stream_car(s), (False, True))

    def test_monte_carlo_all_true(self):
        t = stream_map(lambda x: x == 0, zeros())
        m = monte_carlo(t)
        for i in range(100):
            self.assertEqual(stream_car(m), 1.0)
            m = stream_cdr(m)

    def test_monte_carlo_all_false(self):
        t = stream_map(lambda x: x == 1, zeros())
        m = monte_carlo(t)
        for i in range(100):
            self.assertEqual(stream_ref(m, i), 0.0)

    def test_monte_carlo_50_percent(self):
        t = stream_map(lambda x: x % 2 == 1, integers_from(0))
        m = monte_carlo(t)
        for i in range(50):
            self.assertEqual(stream_car(m), (i // 2 + i % 2) / (i + 1))
            m = stream_cdr(m)

    def test_pi_monte_carlo(self):
        p = pi_monte_carlo()
        for i in range(200):
            print((i,stream_car(p)))
            p = stream_cdr(p)
