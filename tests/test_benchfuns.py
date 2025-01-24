"""Tests for the benchmark functions.

Copyright (c) 2025 Martin F. Schubert
"""

import unittest

import jax.numpy as jnp
from parameterized import parameterized

from benchfuns import benchfuns


class BenchmarkFunctionTest(unittest.TestCase):
    @parameterized.expand([1, 2, 10])
    def test_sphere(self, d):
        x = jnp.linspace(-1, 1, d)
        self.assertSequenceEqual(x.shape, (d,))
        y = benchfuns.sphere(x)
        self.assertSequenceEqual(y.shape, ())

    @parameterized.expand([1, 2, 10])
    def test_lp_norm(self, d):
        x = jnp.linspace(-1, 1, d)
        self.assertSequenceEqual(x.shape, (d,))
        y = benchfuns.lp_norm(x)
        self.assertSequenceEqual(y.shape, ())

    @parameterized.expand([1, 2, 10])
    def test_rosenbrock(self, d):
        x = jnp.linspace(-1, 1, d)
        self.assertSequenceEqual(x.shape, (d,))
        y = benchfuns.rosenbrock(x)
        self.assertSequenceEqual(y.shape, ())

    def test_rosenbrock_2d(self):
        d = 2  # 2D only
        x = jnp.linspace(-1, 1, d)
        self.assertSequenceEqual(x.shape, (d,))
        y = benchfuns.rosenbrock_2d(x)
        self.assertSequenceEqual(y.shape, ())

    def test_tensor_product(self):
        d = 1  # 1D only
        x = jnp.linspace(-1, 1, d)
        self.assertSequenceEqual(x.shape, (d,))
        y = benchfuns.tensor_product(x)
        self.assertSequenceEqual(y.shape, ())

    def test_cantilever_beam(self):
        d = 2  # 2D only
        x = jnp.linspace(-1, 1, d)
        self.assertSequenceEqual(x.shape, (d,))
        y = benchfuns.cantilever_beam(x)
        self.assertSequenceEqual(y.shape, ())

    def test_water_flow(self):
        d = 8  # 8D only
        x = jnp.linspace(-1, 1, d)
        self.assertSequenceEqual(x.shape, (d,))
        y = benchfuns.water_flow(x)
        self.assertSequenceEqual(y.shape, ())

    def test_welded_beam(self):
        d = 3  # 3D only
        x = jnp.linspace(-1, 1, d)
        self.assertSequenceEqual(x.shape, (d,))
        y = benchfuns.welded_beam(x)
        self.assertSequenceEqual(y.shape, ())

    def test_branin(self):
        d = 2  # 2D only
        x = jnp.linspace(-1, 1, d)
        self.assertSequenceEqual(x.shape, (d,))
        y = benchfuns.branin(x)
        self.assertSequenceEqual(y.shape, ())

    @parameterized.expand([1, 2, 10])
    def test_ackley(self, d):
        x = jnp.linspace(-1, 1, d)
        self.assertSequenceEqual(x.shape, (d,))
        y = benchfuns.ackley(x)
        self.assertSequenceEqual(y.shape, ())

    def test_gramacy_lee(self):
        d = 1  # 1D only
        x = jnp.linspace(-1, 1, d)
        self.assertSequenceEqual(x.shape, (d,))
        y = benchfuns.gramacy_lee(x)
        self.assertSequenceEqual(y.shape, ())

    def test_salustowicz(self):
        d = 1  # 1D only
        x = jnp.linspace(-1, 1, d)
        self.assertSequenceEqual(x.shape, (d,))
        y = benchfuns.salustowicz(x)
        self.assertSequenceEqual(y.shape, ())
