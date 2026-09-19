"""Deterministic equal-weight sampling using the C++ MT19937-64 stream.

Neither Python ``random`` nor NumPy's MT19937 implements MT19937-64.
This implementation follows its standardized recurrence and explicitly maps
53 random bits to an open-unit-interval Box-Muller input, as the C++ tool does.
"""

import math
from pathlib import Path

import numpy as np

from .config import estimated_memory_bytes, numeric_values
from .orbit import covariance_factor, from_cartesian, rtn_basis, tau, to_cartesian, validate_elements


class MT19937_64:
    """Standard 64-bit Mersenne Twister, compatible with std::mt19937_64."""

    _MASK = (1 << 64)-1

    def __init__(self, seed):
        if not isinstance(seed, (int, np.integer)) or not 0 <= seed <= self._MASK:
            raise ValueError("seed must be a nonnegative 64-bit integer")
        self._state = [int(seed)]
        for i in range(1, 312):
            previous = self._state[-1]
            self._state.append((6364136223846793005*(previous ^ (previous >> 62))+i) & self._MASK)
        self._index = 312

    def __call__(self):
        if self._index >= 312:
            state = self._state
            for i in range(312):
                x = (state[i] & 0xFFFFFFFF80000000) | (state[(i+1) % 312] & 0x7FFFFFFF)
                state[i] = state[(i+156) % 312] ^ (x >> 1) ^ (0xB5026F5AA96619E9 if x & 1 else 0)
            self._index = 0
        x = self._state[self._index]
        self._index += 1
        x ^= (x >> 29) & 0x5555555555555555
        x ^= (x << 17) & 0x71D67FFFEDA60000
        x ^= (x << 37) & 0xFFF7EEE000000000
        x ^= x >> 43
        return x


class Normal:
    """Box-Muller normal stream with prefix stability across ensemble sizes."""

    def __init__(self, seed):
        self.engine = MT19937_64(seed)
        self._spare = None

    def get(self):
        if self._spare is not None:
            out, self._spare = self._spare, None
            return out
        u = (float(self.engine() >> 11)+.5)/9007199254740992.0
        v = (float(self.engine() >> 11)+.5)/9007199254740992.0
        radius, angle = math.sqrt(-2*math.log(u)), tau*v
        self._spare = radius*math.sin(angle)
        return radius*math.cos(angle)


def apply_offset(config, offset):
    """Apply Cartesian/RTN vector errors or additive equinoctial errors.

    RTN velocities are inertial velocity-error components resolved in the
    nominal epoch basis, not derivatives of coordinates in a rotating frame.
    """
    if config.uncertainty_coordinates == "equinoctial":
        return np.asarray(config.nominal)+offset
    cart = to_cartesian(config.nominal, config.backend.mu)
    if config.uncertainty_coordinates == "cartesian":
        cart += offset
    elif config.uncertainty_coordinates == "rtn":
        basis = rtn_basis(cart)
        for i in range(3):
            for j in range(3):
                cart[i] += basis[j,i]*offset[j]
                cart[i+3] += basis[j,i]*offset[j+3]
    else:
        raise ValueError("Invalid uncertainty_coordinates")
    e = from_cartesian(cart, config.backend.mu)
    e[5] = config.nominal[5]+math.remainder(e[5]-config.nominal[5], tau)
    return e


def sample_initial(c):
    """Return an N-by-6 float64 array; reject invalid samples without redrawing."""
    floor = c.backend.earth_radius_m+c.minimum_altitude_m
    if c.empirical_samples_csv:
        out = []
        with Path(c.empirical_samples_csv).open(encoding="utf-8-sig") as stream:
            for row, raw in enumerate(stream, 1):
                line = raw.partition('#')[0].strip()
                if not line:
                    continue
                try:
                    value = numeric_values(line, 6)
                    if c.uncertainty_coordinates == "cartesian":
                        value = from_cartesian(value, c.backend.mu)
                        value[5] = c.nominal[5]+math.remainder(value[5]-c.nominal[5], tau)
                    elif c.uncertainty_coordinates == "rtn":
                        value = apply_offset(c, value)
                    if estimated_memory_bytes(c, len(out)+1) > c.max_memory_mb*1024*1024:
                        raise ValueError("Empirical ensemble exceeds max_memory_mb")
                    validate_elements(value, c.backend.mu, floor)
                    out.append(value)
                except (ValueError, RuntimeError) as error:
                    raise ValueError(f"Invalid empirical row {row}: {error}") from error
        if len(out) < 2:
            raise ValueError("Empirical ensemble requires >=2 equal-weight samples")
        return np.array(out)
    if estimated_memory_bytes(c, c.samples) > c.max_memory_mb*1024*1024:
        raise ValueError("Estimated run storage exceeds max_memory_mb before sampling")
    out = np.empty((c.samples, 6))
    rng, lower = Normal(c.seed), covariance_factor(c.covariance)
    for sample in range(c.samples):
        z = [rng.get() for _ in range(6)]
        offset = np.zeros(6)
        for i in range(6):
            for j in range(i+1):
                offset[i] += lower[i,j]*z[j]
        try:
            value = apply_offset(c, offset)
            validate_elements(value, c.backend.mu, floor)
            out[sample] = value
        except (ValueError, RuntimeError) as error:
            raise ValueError(f"Invalid sampled orbit {sample}: {error}. Supply a physically valid posterior; samples are not clipped or redrawn.") from error
    return out
