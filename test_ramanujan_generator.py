from unittest import TestCase

from ramanujan_generator import integers_from, interleave, generator_map, pairs, merge_weighted, generator_filter, \
    weighted_pairs, merge_with_next_item, ramanujan_numbers


class TestRamanujanGenerator(TestCase):

    def test_integers_from(self):
        g = integers_from(1)
        for i in range(1, 20):
            self.assertEqual(next(g), i)

    def test_interleave(self):
        g = interleave(integers_from(1), integers_from(101))
        for i in range(1, 20):
            self.assertEqual(next(g), i)
            self.assertEqual(next(g), 100 + i)

    def test_generator_map(self):
        g = generator_map(lambda x: 2 * x, integers_from(1))
        for i in range(1, 20):
            self.assertEqual(next(g), 2 * i)

    def test_generator_filter(self):
        g = generator_filter(lambda x: x % 2 == 0, integers_from(1))
        for i in range(1, 10):
            self.assertEqual(next(g), 2 * i)

    # FIXME: assert
    def test_pairs(self):
        g = pairs(integers_from(1), integers_from(1))
        for i in range(20):
            print(next(g))

    def test_merge_weighted(self):
        g = merge_weighted(generator_filter(lambda x: x % 2 == 0, integers_from(1)),
                           generator_filter(lambda x: x % 2 == 1, integers_from(1)),
                           lambda x: x)
        for i in range(1, 20):
            self.assertEqual(next(g), i)

    def test_weighted_pairs(self):
        g = weighted_pairs(integers_from(1), integers_from(1), lambda x, y: x + y)
        a = next(g)
        for i in range(1, 50):
            a, b = next(g), a
            self.assertGreaterEqual(a[0], b[0])
            self.assertLessEqual(a[1], a[2])

    def test_weighted_pairs_with_ramanujan(self):
        g = weighted_pairs(integers_from(1), integers_from(1), lambda x, y: x ** 3 + y ** 3)
        a = next(g)
        for i in range(1, 100):
            a, b = next(g), a
            print(b)
            self.assertGreaterEqual(a[0], b[0])
            self.assertLessEqual(a[1], a[2])

    def test_merge_with_next_item(self):
        g = merge_with_next_item(integers_from(1))
        for i in range(20):
            a = next(g)
            self.assertEqual(a[0], a[1] + 1)

    def test_ramanujan_numbers(self):
        """
            Execution results:
            Up to 10: execution time: 23ms
            Up to 100: execution time: 752ms; Memory usage=1.5GB
            Up to 200: execution time: 3sec; Memory usage=10MB
       """
        g = ramanujan_numbers()
        for i in range(200):
            a = next(g)
            print(a)
            self.assertEqual(a[0][0], a[1][0])
            self.assertEqual(a[0][0], a[0][1] ** 3 + a[0][2] ** 3)
            self.assertEqual(a[1][0], a[1][1] ** 3 + a[1][2] ** 3)
