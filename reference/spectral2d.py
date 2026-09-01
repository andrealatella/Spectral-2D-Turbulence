import numpy as np

TWO_PI = 2.0 * np.pi


class Spectral2D:
    def __init__(self, n=256, beta=0.0, mu=0.0, tau=None, hyper_p=4, cfl=0.4, dt_max=0.05):
        self.n = n
        self.beta = beta
        self.mu = mu
        self.hyper_p = hyper_p
        self.cfl = cfl
        self.dt_max = dt_max
        self.dx = TWO_PI / n

        k = np.fft.fftfreq(n, 1.0 / n)
        self.kx = k[None, :].astype(np.float64) * np.ones((n, 1))
        self.ky = k[:, None].astype(np.float64) * np.ones((1, n))
        self.k2 = self.kx**2 + self.ky**2
        self.k2_inv = np.where(self.k2 > 0, 1.0 / np.where(self.k2 > 0, self.k2, 1.0), 0.0)

        kcut = n / 3.0
        self.dealias = (np.abs(self.kx) < kcut) & (np.abs(self.ky) < kcut)
        self.nu_h = 0.0 if tau is None else 1.0 / (tau * kcut ** (2 * hyper_p))

        self.lin = -self.mu - self.nu_h * self.k2**hyper_p + 1j * self.beta * self.kx * self.k2_inv
        self.lin[0, 0] = -self.mu

        self.w = np.zeros((n, n), dtype=np.complex128)
        self.t = 0.0
        self.conj_idx = (-np.arange(n)) % n

    def set_vorticity(self, field):
        self.w = np.fft.fft2(field.astype(np.float64)).astype(np.complex128) * self.dealias
        self.w[0, 0] = 0.0

    def vorticity(self):
        return np.real(np.fft.ifft2(self.w))

    def velocity(self):
        uh = 1j * self.ky * self.w * self.k2_inv
        vh = -1j * self.kx * self.w * self.k2_inv
        return np.real(np.fft.ifft2(uh)), np.real(np.fft.ifft2(vh))

    def unpack_fields(self, w):
        uh = 1j * self.ky * w * self.k2_inv
        vh = -1j * self.kx * w * self.k2_inv
        a = np.fft.ifft2(w + 1j * uh)
        b = np.fft.ifft2(vh)
        return np.real(a), np.imag(a), np.real(b)

    def nonlinear(self, w):
        omega, u, v = self.unpack_fields(w)
        z = np.fft.fft2(u * omega + 1j * (v * omega))
        zm = z[np.ix_(self.conj_idx, self.conj_idx)]
        q1 = 0.5 * (z + np.conj(zm))
        q2 = -0.5j * (z - np.conj(zm))
        return (1j * self.kx * q1 + 1j * self.ky * q2) * self.dealias

    def rhs(self, w, forcing):
        f = -self.nonlinear(w)
        if forcing is not None:
            f = f + forcing
        return f

    def timestep(self):
        u, v = self.velocity()
        speed = np.max(np.abs(u) + np.abs(v))
        if speed <= 1e-12:
            return self.dt_max
        return min(self.dt_max, self.cfl * self.dx / speed)

    def expl(self, a, dt):
        return np.exp(self.lin * (a * dt))

    def step(self, dt=None, forcing=None):
        if dt is None:
            dt = self.timestep()
        w0 = self.w
        f0 = self.rhs(w0, forcing)
        w1 = self.expl(1.0, dt) * (w0 + dt * f0)
        f1 = self.rhs(w1, forcing)
        w2 = 0.75 * self.expl(0.5, dt) * w0 + 0.25 * self.expl(-0.5, dt) * (w1 + dt * f1)
        f2 = self.rhs(w2, forcing)
        self.w = (1.0 / 3.0) * self.expl(1.0, dt) * w0 + (2.0 / 3.0) * self.expl(0.5, dt) * (
            w2 + dt * f2
        )
        self.w[0, 0] = 0.0
        self.t += dt
        return dt

    def energy(self):
        return 0.5 * np.sum(np.abs(self.w) ** 2 * self.k2_inv) / self.n**4

    def enstrophy(self):
        return 0.5 * np.sum(np.abs(self.w) ** 2) / self.n**4

    def spectrum(self):
        kr = np.sqrt(self.k2)
        bins = np.arange(0, self.n // 2 + 1)
        idx = np.clip(np.round(kr).astype(int), 0, self.n // 2)
        dens = 0.5 * np.abs(self.w) ** 2 * self.k2_inv / self.n**4
        e = np.bincount(idx.ravel(), weights=dens.ravel(), minlength=len(bins))
        return bins, e[: len(bins)]

    def zonal_mean_u(self):
        u, _ = self.velocity()
        return np.mean(u, axis=1)


def random_field(n, k0=14.0, seed=0, amplitude=1.0):
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(n, 1.0 / n)
    kx = k[None, :] * np.ones((n, 1))
    ky = k[:, None] * np.ones((1, n))
    k2 = kx**2 + ky**2
    kr = np.sqrt(k2)
    envelope = np.where(k2 > 0, kr**2 / (1.0 + (kr / k0) ** 8), 0.0)
    phase = rng.uniform(0, TWO_PI, (n, n))
    spec = np.sqrt(envelope) * np.exp(1j * phase)
    field = np.real(np.fft.ifft2(spec))
    field -= field.mean()
    return amplitude * field / np.std(field)


def _check(label, value, lo, hi):
    good = lo <= value <= hi
    print("  %-34s %-13.5e %s" % (label, value, "ok" if good else "FAIL [%g, %g]" % (lo, hi)))
    return good


def main():
    ok = True

    print("packed nonlinear term vs direct evaluation")
    s = Spectral2D(n=64, beta=0.0)
    s.set_vorticity(random_field(64, k0=8, seed=1))
    omega, u, v = s.unpack_fields(s.w)
    direct = (1j * s.kx * np.fft.fft2(u * omega) + 1j * s.ky * np.fft.fft2(v * omega)) * s.dealias
    err = np.max(np.abs(s.nonlinear(s.w) - direct)) / np.max(np.abs(direct))
    ok &= _check("relative difference", err, 0.0, 1e-12)

    print("\nvorticity reconstruction and incompressibility")
    u, v = s.velocity()
    div = np.max(np.abs(np.real(np.fft.ifft2(1j * s.kx * np.fft.fft2(u) + 1j * s.ky * np.fft.fft2(v)))))
    curl = np.real(np.fft.ifft2(1j * s.kx * np.fft.fft2(v) - 1j * s.ky * np.fft.fft2(u)))
    ok &= _check("max |div u|", div, 0.0, 1e-10)
    ok &= _check("max |curl u - omega|", np.max(np.abs(curl - s.vorticity())), 0.0, 1e-10)

    print("\ninviscid invariants, 200 steps at dt=2e-3 (beta=0)")
    s = Spectral2D(n=128, beta=0.0, mu=0.0, tau=None)
    s.set_vorticity(random_field(128, k0=10, seed=2))
    e0, z0 = s.energy(), s.enstrophy()
    for _ in range(200):
        s.step(dt=2e-3)
    ok &= _check("relative energy drift", abs(s.energy() / e0 - 1.0), 0.0, 1e-7)
    ok &= _check("relative enstrophy drift", abs(s.enstrophy() / z0 - 1.0), 0.0, 1e-7)

    print("\ninviscid invariants with beta=12 (Rossby waves in the integrating factor)")
    s = Spectral2D(n=128, beta=12.0, mu=0.0, tau=None)
    s.set_vorticity(random_field(128, k0=10, seed=2))
    e0, z0 = s.energy(), s.enstrophy()
    for _ in range(200):
        s.step(dt=2e-3)
    ok &= _check("relative energy drift", abs(s.energy() / e0 - 1.0), 0.0, 1e-7)
    ok &= _check("relative enstrophy drift", abs(s.enstrophy() / z0 - 1.0), 0.0, 1e-7)

    print("\nlinear Rossby wave dispersion, single mode (kx=3, ky=2), beta=8")
    n = 64
    s = Spectral2D(n=n, beta=8.0, mu=0.0, tau=None)
    kxm, kym = 3, 2
    field = np.zeros((n, n), dtype=np.complex128)
    field[kym, kxm] = 1.0
    s.w = field * n * n * 1e-7
    w_init = s.w[kym, kxm]
    tend = 0.5
    steps = 250
    for _ in range(steps):
        s.step(dt=tend / steps)
    expected_phase = s.beta * kxm / (kxm**2 + kym**2) * tend
    got_phase = np.angle(s.w[kym, kxm] / w_init)
    dphase = np.angle(np.exp(1j * (got_phase - expected_phase)))
    ok &= _check("phase error vs analytic", abs(dphase), 0.0, 1e-9)
    ok &= _check("amplitude drift", abs(abs(s.w[kym, kxm] / w_init) - 1.0), 0.0, 1e-9)

    print("\nTaylor-Green vortex, N=64, k=2, Navier-Stokes viscosity")
    n, kk = 64, 2
    s = Spectral2D(n=n, beta=0.0, mu=0.0, tau=40.0, hyper_p=1)
    xs = np.arange(n) * TWO_PI / n
    gx, gy = np.meshgrid(xs, xs, indexing="xy")
    s.set_vorticity(4.0 * np.cos(kk * gx) * np.cos(kk * gy))
    w0 = s.w.copy()
    nl = s.nonlinear(w0)
    ok &= _check(
        "nonlinear term vanishes identically", np.max(np.abs(nl)) / np.max(np.abs(w0)), 0.0, 1e-14
    )
    t_end, steps = 3.0, 300
    for _ in range(steps):
        s.step(dt=t_end / steps)
    decay = abs(s.w[kk, kk] / w0[kk, kk])
    expected = np.exp(-s.nu_h * (2.0 * kk * kk) ** s.hyper_p * t_end)
    ok &= _check("amplitude vs exp(-nu |k|^2 t)", abs(decay / expected - 1.0), 0.0, 1e-9)

    print("\ndecaying turbulence at N=256, enstrophy-range slope")
    s = Spectral2D(n=256, beta=0.0, mu=0.0, tau=2.0)
    s.set_vorticity(random_field(256, k0=24, seed=5, amplitude=6.0))
    z0 = s.enstrophy()
    while s.t < 12.0:
        s.step()
    bins, e = s.spectrum()
    band = (bins >= 8) & (bins <= 45) & (e > 0)
    slope = np.polyfit(np.log(bins[band]), np.log(e[band]), 1)[0]
    print("  %-34s %-13.3f %s" % ("fitted E(k) slope", slope, "ok" if -4.2 < slope < -2.2 else "FAIL"))
    ok &= -4.2 < slope < -2.2
    ok &= _check("enstrophy decayed", 1.0 - s.enstrophy() / z0, 0.05, 1.0)
    ok &= _check("final time", s.t, 12.0, 13.0)

    print("\nbeta-plane jets at N=128, zonal anisotropy after forcing-free evolution")
    s = Spectral2D(n=128, beta=25.0, mu=0.0, tau=1.0)
    s.set_vorticity(random_field(128, k0=16, seed=7, amplitude=8.0))
    while s.t < 20.0:
        s.step()
    u, v = s.velocity()
    ratio = np.mean(u**2) / max(np.mean(v**2), 1e-30)
    print("  %-34s %-13.3f %s" % ("<u^2>/<v^2>", ratio, "ok" if ratio > 1.5 else "FAIL"))
    ok &= ratio > 1.5

    print("\nREFERENCE SOLVER:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
