"""Benchmark functions for surrogate modeling.

The functions here are the noise-free functions from https://docs.sciml.ai/Surrogates/.
Many of these functions are also included as benchmark functions in the Surrogaate
Modeling toolbox: https://smt.readthedocs.io/en/latest/

Copyright (c) 2025 Martin F. Schubert
"""

import jax.numpy as jnp


def sphere(x: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the sphere benchmark function at `x`."""
    assert x.ndim == 1
    return jnp.sum(x**2)


def lp_norm(x: jnp.ndarray, p: float = 1.3) -> jnp.ndarray:
    """Evaluate the lp-norm benchmark function at `x`."""
    assert x.ndim == 1
    return jnp.sum(jnp.abs(x) ** p) ** (1 / p)


def rosenbrock(x: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the Rosenbrock benchmark function at `x`."""
    assert x.ndim == 1
    return jnp.sum((x[1:] - x[:-1]) ** 2 + (x[:-1] - 1) ** 2)


def tensor_product(x: jnp.ndarray, a: float = 0.5) -> jnp.ndarray:
    """Evaluate the tensor product benchmark function at `x`."""
    assert x.ndim == 1
    return jnp.prod(jnp.cos(a * jnp.pi * x))


def cantilever_beam(x: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the cantilever beam benchmark function at `x`."""
    assert x.shape == (2,)
    w, t = x
    del x
    L = 100.0
    E = 2.770674127819261e7
    X = 530.8038576066307
    Y = 997.8714938733949
    result = 4 * L**3 / (E * w * t) * jnp.sqrt((Y / t**2) ** 2 + (X / w**2) ** 2)
    return jnp.asarray(result)


def water_flow(x: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the water flow benchmark function at `x`."""
    assert x.shape == (8,)
    r_w, r, T_u, H_u, T_l, H_l, L, K_w = x
    log_r_rw = jnp.log(r / r_w)
    result = (2 * jnp.pi * T_u * (H_u - H_l)) / (
        log_r_rw * (1 + 2 * L * T_u / (log_r_rw * r_w**2 * K_w) + T_u / T_l)
    )
    return jnp.asarray(result)


def welded_beam(x: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the welded beam benchmark function at `x`."""
    assert x.shape == (3,)
    h, _l, t = x
    a = 6000 / (jnp.sqrt(2) * h * _l)
    b = (6000 * (14 + 0.5 * _l) * jnp.sqrt(0.25 * (_l**2 + (h + t) ** 2))) / (
        2 * (0.707 * h * _l * (_l**2 / 12 + 0.25 * (h + t) ** 2))
    )
    return (jnp.sqrt(a**2 + b**2 + _l * a * b)) / (
        jnp.sqrt(0.25 * (_l**2 + (h + t) ** 2))
    )


def branin(x: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the Branin benchmark function at `x`."""
    assert x.shape == (2,)
    x1, x2 = x
    b = 5.1 / (4 * jnp.pi**2)
    c = 5 / jnp.pi
    r = 6
    a = 1
    s = 10
    t = 1 / (8 * jnp.pi)
    term1 = a * (x2 - b * x1**2 + c * x1 - r) ** 2
    term2 = s * (1 - t) * jnp.cos(x1)
    return jnp.asarray(term1 + term2 + s)


def ackley(x: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the Ackley benchmark function at `x`."""
    assert x.ndim == 1
    a = 20.0
    b = 0.2
    c = 2 * jnp.pi

    d = x.size
    term1 = -a * jnp.exp(-b * jnp.sqrt(1 / d * jnp.sum(x**2)))
    term2 = -jnp.exp(1 / d * jnp.sum(jnp.cos(c * x)))
    return term1 + term2 + a + jnp.exp(1)


def gramacy_lee(x: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the Gramacy and Lee benchmark function at `x`."""
    assert x.shape == (1,)
    return (jnp.sin(10 * jnp.pi * x) / (2 * x) + (x - 1) ** 4).squeeze()


def salustowicz(x: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the Salustowicz benchmark function at `x`."""
    assert x.shape == (1,)
    term1 = jnp.exp(-x) * x**3 * jnp.cos(x) * jnp.sin(x)
    term2 = jnp.cos(x) * jnp.sin(x) * jnp.sin(x) - 1
    return (term1 * term2).squeeze()
