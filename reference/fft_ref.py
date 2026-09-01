import numpy as np

TWO_PI = 2.0 * np.pi


def radix_schedule(n):
    radices = []
    m = n
    while m % 4 == 0:
        radices.append(4)
        m //= 4
    if m == 2:
        radices.append(2)
    elif m != 1:
        raise ValueError("N=%d is not of the form 4^a or 2 * 4^a" % n)
    return radices


def digit_reverse(i, radices):
    out = 0
    for r in reversed(radices):
        out = out * r + (i % r)
        i //= r
    return out


def fft_line(x, inverse=False):
    n = len(x)
    radices = radix_schedule(n)
    sgn = 1.0 if inverse else -1.0

    buf = np.zeros(n, dtype=np.complex128)
    for i in range(n):
        buf[digit_reverse(i, radices)] = x[i]

    m = 1
    for r in radices:
        if r == 4:
            for bt in range(n // 4):
                j = bt % m
                k = (bt // m) * 4 * m
                ang = sgn * TWO_PI * j / (4 * m)
                a0 = buf[k + j]
                a1 = buf[k + j + m] * np.exp(1j * ang)
                a2 = buf[k + j + 2 * m] * np.exp(2j * ang)
                a3 = buf[k + j + 3 * m] * np.exp(3j * ang)
                b0 = a0 + a2
                b1 = a0 - a2
                b2 = a1 + a3
                b3 = 1j * sgn * (a1 - a3)
                buf[k + j] = b0 + b2
                buf[k + j + m] = b1 + b3
                buf[k + j + 2 * m] = b0 - b2
                buf[k + j + 3 * m] = b1 - b3
            m *= 4
        else:
            for bt in range(n // 2):
                j = bt % m
                k = (bt // m) * 2 * m
                ang = sgn * TWO_PI * j / (2 * m)
                a0 = buf[k + j]
                a1 = buf[k + j + m] * np.exp(1j * ang)
                buf[k + j] = a0 + a1
                buf[k + j + m] = a0 - a1
            m *= 2

    if m != n:
        raise AssertionError("radix schedule reached m=%d for N=%d" % (m, n))
    return buf / n if inverse else buf


def fft2_model(a, inverse=False):
    n = a.shape[0]
    out = np.array(a, dtype=np.complex128)
    for row in range(n):
        out[row, :] = fft_line(out[row, :], inverse)
    for col in range(n):
        out[:, col] = fft_line(out[:, col], inverse)
    return out


def check_permutation(n):
    radices = radix_schedule(n)
    seen = sorted(digit_reverse(i, radices) for i in range(n))
    return seen == list(range(n))


def main():
    rng = np.random.default_rng(20260815)
    ok = True

    for n in (32, 64, 128, 256, 512, 1024):
        bijective = check_permutation(n)
        x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        ref = np.fft.fft(x)
        ef = np.max(np.abs(fft_line(x, False) - ref)) / np.max(np.abs(ref))
        ei = np.max(np.abs(fft_line(x, True) - np.fft.ifft(x))) * n / np.max(np.abs(ref))
        good = bijective and ef < 1e-13 and ei < 1e-13
        ok &= good
        print(
            "N=%-5d radices=%-18s bijective=%-5s fwd=%.3e inv=%.3e %s"
            % (n, radix_schedule(n), bijective, ef, ei, "ok" if good else "FAIL")
        )

    for n in (32, 64, 128):
        a = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        ref = np.fft.fft2(a)
        e2 = np.max(np.abs(fft2_model(a, False) - ref)) / np.max(np.abs(ref))
        rt = np.max(np.abs(fft2_model(fft2_model(a, False), True) - a))
        ok &= e2 < 1e-13 and rt < 1e-13
        print("2D N=%-4d fwd=%.3e roundtrip=%.3e" % (n, e2, rt))

    real_field = rng.standard_normal((64, 64))
    back = fft2_model(fft2_model(real_field.astype(np.complex128), False), True)
    resid = np.max(np.abs(back.imag))
    ok &= resid < 1e-13
    print("hermitian residual imag=%.3e" % resid)

    workgroup_bytes = {n: n * 8 for n in (256, 512, 1024)}
    print("\nworkgroup storage, one buffer:")
    for n, b in workgroup_bytes.items():
        print("  N=%-5d %5d bytes  (Stockham ping-pong would need %d)" % (n, b, 2 * b))

    print("\nDIT MODEL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
