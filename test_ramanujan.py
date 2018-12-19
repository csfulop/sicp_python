from unittest import TestCase

from mockito import when, any_, unstub

import ramanujan
from ramanujan import ramanujans, _is_cube_number, _is_ramanujan


class TestRamanujan(TestCase):
    def tearDown(self):
        unstub()

    def test_ramanujans_with_mock(self):
        when(ramanujan)._is_ramanujan(any_()).thenReturn((False,[]))
        when(ramanujan)._is_ramanujan(3).thenReturn((True,[]))
        when(ramanujan)._is_ramanujan(5).thenReturn((True,[]))
        r = ramanujans()
        self.assertEqual(next(r)[0],3)
        self.assertEqual(next(r)[0],5)

    def test_is_cube_number(self):
        def _check_number(n):
            self.assertTrue(_is_cube_number(n), n)
            self.assertFalse(_is_cube_number(n+1), n+1)
            self.assertFalse(_is_cube_number(n-1), n-1)
        _check_number(8)
        _check_number(27)
        _check_number(64)
        _check_number(125)
        _check_number(12**3)
        _check_number(123456**3)

    def test_is_ramanujan(self):
        self.assertFalse(_is_ramanujan(1)[0])
        self.assertFalse(_is_ramanujan(27)[0])
        self.assertFalse(_is_ramanujan(1000)[0])
        self.assertTrue(_is_ramanujan(1729)[0])

    def test_ramanujans(self):
        r = ramanujans()
        self.assertEqual(next(r)[0],1729)
        self.assertEqual(next(r)[0],4104)

    def test_ramanujans_time(self):
        """
            Execution results:
            Withour memoization: Memory usage: constant 10MB
                Up to 10: execution time: 5sec
                Up to 20: execution time: 32sec
            With memoization:
                Up to 20: execution time: 21sec; Memory usage: 55MB
                Up to 50: execution time: ??sec; Memory usage: 200MB
                Up to 100: execution time: 13min; Memory usage: 780MB
        """
        r = ramanujans()
        for i in range(10):
            print((i,next(r)))

