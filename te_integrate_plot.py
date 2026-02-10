"""
k_parallel -> real-space (rho) integrator for multilayer Green's functions.

Design goal (your workflow):
    - Keep te_greens.py as the "engine" (you will keep editing it).
    - This file ONLY:
        (1) imports te_greens.gyy_TE (and later TM engines)
        (2) performs angular integration -> Bessel J0
        (3) performs k_parallel radial integral (Gauss-Legendre quadrature)
        (4) plots results
using only numpy and matplotlib (no scipy).

Math (2D in-plane Fourier/Bessel transform):
    G_yy(rho) = (1/(2π)) ∫_0^{∞} k_parallel * J0(k_parallel*rho) * G_yy(k_parallel) dk_parallel

Vectorised for Apple Silicon M3:
    All per-kp Python loops have been replaced with single
    batch calls that pass the entire kp array at once.
    NumPy dispatches the heavy lifting to BLAS/Accelerate on
    macOS (NEON + AMX).  When ``mlx`` is installed the backend
    can optionally run on the Apple GPU.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# You will keep modifying te_greens.py; we import from it on purpose.
from te_greens import gyy_TE
from te_greens import dgyy_dzobs_TE
from tm_greens import gxx_TM
from tm_greens import dgxx_dzobs_TM
from tm_greens import gzx_TM


def J0_series(x: np.ndarray) -> np.ndarray:
    """
    Minimal, dependency-free J0(x) approximation (series, stable for small/moderate x).

    J0(x) = Σ_{m=0}^∞ (-1)^m (x^2/4)^m / (m!)^2

    This is fine for plotting / debugging. If you later want faster & more accurate:
        - use scipy.special.j0
    """
    x = np.asarray(x, dtype=np.complex128)
    x2_over_4 = (x * x) / 4.0

    # Adaptive-ish truncation: more terms for larger |x|
    # (still cheap, but don't go crazy)
    max_abs = float(np.max(np.abs(x)))
    if max_abs < 5:
        M = 40
    elif max_abs < 20:
        M = 80
    else:
        M = 140

    out = np.zeros_like(x, dtype=np.complex128)
    term = np.ones_like(x, dtype=np.complex128)
    out += term
    for m in range(1, M):
        term *= (-x2_over_4) / (m * m)
        out += term
    return out

def J2_series(x: np.ndarray) -> np.ndarray:
    """
    Minimal, dependency-free J2(x) approximation (series).

    J2(x) = Σ_{m=0}^∞ (-1)^m (x/2)^(2m+2) / (m!(m+2)!)
    """
    x = np.asarray(x, dtype=np.complex128)

    max_abs = float(np.max(np.abs(x)))
    if max_abs < 5:
        M = 50
    elif max_abs < 20:
        M = 120
    else:
        M = 220

    out = np.zeros_like(x, dtype=np.complex128)

    # term for m=0: (x/2)^2 / (0! * 2!) = x^2 / 8
    term = (x * x) / 8.0
    out += term

    # recurrence for term_{m} -> term_{m+1}
    # term_{m+1} = term_m * [-(x^2/4)] / [(m+1)(m+3)]
    x2_over_4 = (x * x) / 4.0
    for m in range(0, M - 1):
        term *= (-x2_over_4) / ((m + 1) * (m + 3))
        out += term

    return out

def gauss_legendre(n: int, a: float = -1.0, b: float = 1.0):
    """
    Gauss-Legendre quadrature nodes and weights on [a, b].

    Uses Newton-Raphson iteration on Legendre polynomials (no scipy).
    Returns (nodes, weights) arrays of length *n*.
    """
    nodes = np.empty(n, dtype=float)
    weights = np.empty(n, dtype=float)
    m = (n + 1) // 2

    for i in range(m):
        # Initial guess for the i-th root (Abramowitz & Stegun approx)
        z = np.cos(np.pi * (i + 0.75) / (n + 0.5))

        for _ in range(100):
            p1, p2 = 1.0, 0.0
            for k in range(1, n + 1):
                p3 = p2
                p2 = p1
                p1 = ((2.0 * k - 1.0) * z * p2 - (k - 1.0) * p3) / k

            # derivative of Legendre polynomial
            pp = n * (z * p1 - p2) / (z * z - 1.0)
            z1 = z
            z = z1 - p1 / pp
            if abs(z - z1) < 1e-15:
                break

        nodes[i] = -z
        nodes[n - 1 - i] = z
        weights[i] = 2.0 / ((1.0 - z * z) * pp * pp)
        weights[n - 1 - i] = weights[i]

    # transform from [-1, 1] to [a, b]
    scale = (b - a) / 2.0
    shift = (a + b) / 2.0
    nodes = shift + scale * nodes
    weights = weights * scale

    return nodes, weights

def gyy_TE_rho(
    n_list,
    d_list,
    layer_src: int,
    z_src: float,
    layer_obs: int,
    z_obs: float,
    k0: float,
    rho: float,
    k_parallel_max: float,
    num_k: int,
) -> complex:

    # --- Gauss-Legendre nodes and weights on (0, k_parallel_max) ---
    kps, wts = gauss_legendre(num_k, a=0.0, b=k_parallel_max)

    # --- Vectorised batch calls (all kp values at once) ---
    Gyykp = gyy_TE(
        n_list, d_list,
        layer_src, z_src,
        layer_obs, z_obs,
        k0, kps
    )

    DGyykp = dgyy_dzobs_TE(
        n_list, d_list,
        layer_src, z_src,
        layer_obs, z_obs,
        k0, kps
    )

    Gxxkp = gxx_TM(
        n_list, d_list,
        layer_src, z_src,
        layer_obs, z_obs,
        k0, kps
    )

    DGxxkp = dgxx_dzobs_TM(
        n_list, d_list,
        layer_src, z_src,
        layer_obs, z_obs,
        k0, kps
    )

    Gzxkp = gzx_TM(
        n_list, d_list,
        layer_src, z_src,
        layer_obs, z_obs,
        k0, kps
    )

    # --- Bessel factor ---
    J0 = J0_series(kps * rho)
    J2 = J2_series(kps * rho)
    pref = 1.0 / np.pi

    # --- integrands ---
    integrand_1 = kps * J0 * Gyykp
    integrand_2 = kps * J0 * np.conj(DGyykp)
    integrand_3 = kps * J2 * Gyykp
    integrand_4 = kps * J2 * np.conj(DGyykp)
    integrand_5 = kps * J0 * Gxxkp
    integrand_6 = kps * J0 * np.conj(DGxxkp)
    integrand_7 = kps * J2 * Gxxkp
    integrand_8 = kps * J2 * np.conj(DGxxkp)
    integrand_9 = kps**2 * J0 * np.conj(Gzxkp)
    integrand_10 = kps**2 * J2 * np.conj(Gzxkp)

    # --- Gauss-Legendre weighted sums ---
    I1 = pref * np.dot(wts, integrand_1)
    I2 = pref * np.dot(wts, integrand_2)
    I3 = pref * np.dot(wts, integrand_3)
    I4 = pref * np.dot(wts, integrand_4)
    I5 = pref * np.dot(wts, integrand_5)
    I6 = pref * np.dot(wts, integrand_6)
    I7 = pref * np.dot(wts, integrand_7)
    I8 = pref * np.dot(wts, integrand_8)
    I9 = pref * np.dot(wts, integrand_9)
    I10 = pref * np.dot(wts, integrand_10)
    return I1*I2 + I3*I4 + I5*I6 + I7*I8 + 1j*I5*I9 + 1j*I5*I10 + I5*I2 + I7*I4 + I1*I6 + I3*I8 + 1j*I1*I9 + 1j*I1*I10

def demo_plot_TE():
    """
    Minimal demo plot: Re/Im of G_yy(ρ) for a fixed wavelength and layer positions.

    Edit the geometry freely; this script intentionally stays separate from te_greens.py.
    """
    # Geometry (edit)
    n_list = [1.0, 1.0, 1.0, 1.0]
    d_list = [0.0, 2.0, 1.5]

    # Source / obs (edit)
    layer_src = 0
    layer_obs = 0
    z_src = -0.35
    z_obs = -0.20

    # Wavelength -> k0
    wl = 0.650
    k0 = 2.0 * np.pi / wl

    # k_parallel integration setup (edit)
    # Typical: some multiple of k0. For evanescent contributions, you may need larger.
    k_parallel_max = 3.5 * k0
    num_k = 100  # Gauss-Legendre quadrature points

    rhos = np.linspace(0.0, 1.30, 500)

    G_rho = np.array([
        gyy_TE_rho(
            n_list, d_list,
            layer_src, z_src,
            layer_obs, z_obs,
            k0,
            rho=float(r),
            k_parallel_max=k_parallel_max,
            num_k=num_k,
        )
        for r in rhos
    ], dtype=np.complex128)

    plt.figure()
    plt.plot(rhos , np.real(G_rho), label="Re Gyy")
    plt.plot(rhos , np.imag(G_rho), label="Im Gyy")
    plt.axhline(0.0, color='k', linestyle='--', linewidth=1)
    plt.xlabel(r"$\rho$ (µm)")
    plt.ylabel(r"$G_{yy}^{TE}(\rho)$ (arb.)")
    plt.legend()
    plt.title("TE: Bessel-integrated $G_{yy}$ from te_greens.gyy_TE")
    plt.tight_layout()
    plt.show()
    # 再用同一份 G_rho 直接轉成 2D（不重算）
    plot_Grho_as_2D(-G_rho, rhos, extent_um=3.5, N=401, which="imag")

def plot_Grho_as_2D(G_rho, rhos, extent_um=3.5, N=401, which="real"):
    """
    Make a 2D map from radial data G(ρ) by coordinate transform + interpolation.

    extent_um: plot x,y in [-extent_um, +extent_um] (µm)
    N: grid points per axis
    which: "real" or "imag" or "abs"
    """
    # 1) build (x,y) grid in meters
    x = np.linspace(-extent_um, extent_um, N) 
    y = np.linspace(-extent_um, extent_um, N) 
    X, Y = np.meshgrid(x, y, indexing="xy")
    R = np.sqrt(X**2 + Y**2)

    # 2) choose which scalar to plot
    if which == "real":
        g1d = np.real(G_rho)
        label = "Re Gyy"
    elif which == "imag":
        g1d = np.imag(G_rho)
        label = "Im Gyy"
    elif which == "abs":
        g1d = np.abs(G_rho)
        label = "|Gyy|"
    else:
        raise ValueError('which must be "real", "imag", or "abs"')

    # 3) interpolate g(ρ) onto R
    # rhos is in nm already in your code
    # np.interp is fast (linear); outside range -> fill with 0
    G2 = np.interp(R.ravel(), rhos, g1d, left=0.0, right=0.0).reshape(R.shape)

    # 4) plot
    plt.figure()
    im = plt.imshow(
        G2,
        origin="lower",
        extent=[-extent_um, extent_um, -extent_um, extent_um],
        aspect="equal",
    )
    plt.colorbar(im, label=label)
    plt.xlabel("x (µm)")
    plt.ylabel("y (µm)")
    plt.title(f"TE: {label} from radial Gyy(ρ)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    demo_plot_TE()
