"""SI equinoctial orbit geometry, independent of the native DSST backend.

Elements are ``[a, e*cos(omega+Omega), e*sin(omega+Omega),
tan(i/2)*cos(Omega), tan(i/2)*sin(Omega), unwrapped mean longitude]``.
RTN basis vectors are rows, so ``basis.T @ components`` rotates to inertial.
"""

import math

import numpy as np

pi = PI = math.pi
tau = TAU = math.tau
day = DAY = 86400.0


def _six(value):
    out = np.asarray(value, dtype=np.float64)
    if out.shape != (6,):
        raise ValueError("Expected exactly six values")
    return out


def wrap_angle(angle):
    return angle % tau


def dot(a, b):
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def cross(a, b):
    return np.array([a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]])


def norm(a):
    return math.sqrt(dot(a, a))


def rtn_basis(state):
    state = _six(state)
    r, v = state[:3].copy(), state[3:]
    h = cross(r, v)
    radius, hn = norm(r), norm(h)
    if not (radius > 0.0 and hn > 0.0 and np.isfinite(state).all()):
        raise ValueError("Degenerate/nonfinite Cartesian orbit")
    r /= radius
    h /= hn
    return np.array([r, cross(h, r), h])


def from_keplerian(a, e, inclination, raan, argument_perigee, mean_anomaly):
    if not (0.0 <= inclination < pi - 1e-8 and 0.0 <= e < 1.0):
        raise ValueError("Require 0 <= eccentricity < 1 and inclination < 180 degrees")
    p, h = raan + argument_perigee, math.tan(inclination / 2.0)
    return np.array([a, e*math.cos(p), e*math.sin(p), h*math.cos(raan), h*math.sin(raan), p+mean_anomaly])


def validate_elements(elements, mu, minimum_perigee_m=0.0):
    e = _six(elements)
    if not np.isfinite(e).all():
        raise ValueError("Nonfinite orbital element")
    ecc = math.hypot(e[1], e[2])
    if not (mu > 0.0 and math.isfinite(mu) and e[0] > 0.0 and ecc < 1.0):
        raise ValueError("Require finite mu > 0 and bound elliptic orbit")
    if math.hypot(e[3], e[4]) > 1e6:
        raise ValueError("Near-retrograde equinoctial singularity is unsupported")
    if e[0] * (1.0 - ecc) < minimum_perigee_m:
        raise ValueError("Orbit perigee below configured no-drag validity floor")


def _plane_basis(q, p):
    den = 1.0 + p*p + q*q
    return (np.array([(1-p*p+q*q)/den, 2*p*q/den, -2*p/den]),
            np.array([2*p*q/den, (1+p*p-q*q)/den, 2*q/den]))


def to_cartesian(elements, mu):
    e = _six(elements)
    validate_elements(e, mu)
    a, ex, ey, q, p = map(float, e[:5])
    lm = math.remainder(e[5], tau)
    ecc = math.hypot(ex, ey)
    lo, hi, f = lm-ecc, lm+ecc, lm
    for _ in range(80):
        value = f-ex*math.sin(f)+ey*math.cos(f)-lm
        if abs(value) < 2e-15:
            break
        if value > 0:
            hi = f
        else:
            lo = f
        next_f = f-value/(1-ex*math.cos(f)-ey*math.sin(f))
        f = next_f if lo < next_f < hi else 0.5*(lo+hi)
    else:
        raise RuntimeError("Eccentric longitude solve did not converge")
    b, c, s = 1/(1+math.sqrt(1-ecc*ecc)), math.cos(f), math.sin(f)
    x = a*((1-b*ey*ey)*c+b*ex*ey*s-ex)
    y = a*((1-b*ex*ex)*s+b*ex*ey*c-ey)
    fd = math.sqrt(mu/a)/a/(1-ex*c-ey*s)
    xd = a*(-(1-b*ey*ey)*s+b*ex*ey*c)*fd
    yd = a*((1-b*ex*ex)*c-b*ex*ey*s)*fd
    u, v = _plane_basis(q, p)
    return np.concatenate((x*u+y*v, xd*u+yd*v))


def from_cartesian(state, mu):
    state = _six(state)
    if not np.isfinite(state).all():
        raise ValueError("Nonfinite Cartesian state")
    if not (mu > 0 and math.isfinite(mu)):
        raise ValueError("Invalid gravitational parameter")
    r, v = state[:3], state[3:]
    h = cross(r, v)
    rn, hn, v2 = norm(r), norm(h), dot(v, v)
    if not (rn > 0 and hn > 0):
        raise ValueError("Degenerate Cartesian orbit")
    inverse_a = 2.0/rn-v2/mu
    if inverse_a <= 0:
        raise ValueError("Require bound elliptic orbit")
    a = 1.0/inverse_a
    w = h/hn
    if 1+w[2] < 1e-12:
        raise ValueError("Retrograde equinoctial singularity")
    q, p = -w[1]/(1+w[2]), w[0]/(1+w[2])
    u, vv = _plane_basis(q, p)
    ev = cross(v, h)/mu-r/rn
    ex, ey = dot(ev, u), dot(ev, vv)
    ecc = math.hypot(ex, ey)
    out = np.array([a, ex, ey, q, p, 0.0])
    validate_elements(out, mu)
    beta = 1/(1+math.sqrt(1-ecc*ecc))
    xx, yy = dot(r, u)/a+ex, dot(r, vv)/a+ey
    m11, m22, m12 = 1-beta*ey*ey, 1-beta*ex*ex, beta*ex*ey
    det = m11*m22-m12*m12
    cf, sf = (m22*xx-m12*yy)/det, (m11*yy-m12*xx)/det
    out[5] = wrap_angle(math.atan2(sf, cf)-ex*sf+ey*cf)
    return out


def covariance_factor(covariance):
    """Scale-aware lower PSD factor; deterministic axes remain exactly zero."""
    c = np.asarray(covariance, dtype=np.float64)
    if c.shape != (6, 6):
        raise ValueError("Covariance must have six rows and six columns")
    if not np.isfinite(c).all() or np.any(np.diag(c) < 0):
        raise ValueError("Negative/nonfinite covariance diagonal or entry")
    scale = np.sqrt(np.diag(c))
    correlation, lower, out = (np.zeros((6, 6)) for _ in range(3))
    for i in range(6):
        for j in range(6):
            tolerance = 1e-12*max(abs(c[i,j]), abs(c[j,i]), scale[i]*scale[j], 1e-300)
            if abs(c[i,j]-c[j,i]) > tolerance:
                raise ValueError("Covariance must be symmetric")
            if scale[i] == 0 or scale[j] == 0:
                if c[i,j] != 0:
                    raise ValueError("Zero-variance covariance row must be zero")
            else:
                correlation[i,j] = c[i,j]/scale[i]/scale[j]
    for i in range(6):
        for j in range(i+1):
            x = correlation[i,j]
            for k in range(j):
                x -= lower[i,k]*lower[j,k]
            if i == j:
                if x < -1e-12:
                    raise ValueError("Covariance is not positive semidefinite")
                lower[i,j] = math.sqrt(max(0.0, x))
            elif lower[j,j] > 1e-14:
                lower[i,j] = x/lower[j,j]
            elif abs(x) > 1e-12:
                raise ValueError("Covariance is not positive semidefinite")
            out[i,j] = scale[i]*lower[i,j]
    return out
