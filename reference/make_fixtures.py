import json
import os

import numpy as np

from spectral2d import Spectral2D, random_field

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures.json")


def flat(a):
    z = np.asarray(a, dtype=np.complex128).ravel()
    out = np.empty(z.size * 2, dtype=np.float64)
    out[0::2] = z.real
    out[1::2] = z.imag
    return [float(x) for x in out]


def main():
    rng = np.random.default_rng(11)
    fx = {}

    n = 32
    a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    fx["fft"] = {
        "n": n,
        "input": flat(a),
        "forward": flat(np.fft.fft2(a)),
        "inverse": flat(np.fft.ifft2(a)),
    }

    s = Spectral2D(n=n, beta=7.0, mu=0.02, tau=1.5)
    s.set_vorticity(random_field(n, k0=6, seed=3))

    uh = 1j * s.ky * s.w * s.k2_inv
    vh = -1j * s.kx * s.w * s.k2_inv
    fx["velocity"] = {
        "n": n,
        "what": flat(s.w),
        "packA": flat(s.w + 1j * uh),
        "packB": flat(vh),
    }

    omega, u, v = s.unpack_fields(s.w)
    z = np.fft.fft2(u * omega + 1j * (v * omega))
    fx["assemble"] = {
        "n": n,
        "z": flat(z),
        "rhs": flat(-s.nonlinear(s.w)),
    }

    dt = 0.011
    before = s.w.copy()
    s.step(dt=dt)
    fx["step"] = {
        "n": n,
        "beta": s.beta,
        "mu": s.mu,
        "nuH": s.nu_h,
        "hyperP": s.hyper_p,
        "dt": dt,
        "before": flat(before),
        "after": flat(s.w),
    }

    s2 = Spectral2D(n=n, beta=7.0, mu=0.02, tau=1.5)
    s2.w = before.copy()
    for _ in range(20):
        s2.step(dt=dt)
    fx["run20"] = {
        "n": n,
        "dt": dt,
        "after": flat(s2.w),
        "energy": float(s2.energy()),
        "enstrophy": float(s2.enstrophy()),
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(fx, fh)
    print("wrote %s (%.1f KB)" % (os.path.normpath(OUT), os.path.getsize(OUT) / 1024.0))
    for key in fx:
        print("  fixture:", key)


if __name__ == "__main__":
    raise SystemExit(main())
