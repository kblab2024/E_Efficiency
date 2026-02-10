"""
k_parallel -> real-space (rho) integrator for multilayer Green's functions.

Design goal (your workflow):
    - Keep te_greens.py as the "engine" (you will keep editing it).
    - This file ONLY:
        (1) imports te_greens.gyy_TE (and later TM engines)
        (2) performs angular integration -> Bessel J0
        (3) performs k_parallel radial integral (Gauss-Legendre quadrature)
        (4) plots results
using numpy, matplotlib, and scipy.

Acceleration:
    - Bessel functions use ``scipy.special.jv`` (compiled C, accurate
      for all arguments, replaces the old power-series loops).
    - Gauss-Legendre nodes/weights use ``scipy.special.roots_legendre``.
    - When MLX is available (Apple Silicon), the integrand assembly and
      weighted dot products run on the GPU via ``accel_backend.xp``.

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
from scipy.special import roots_legendre, jv
from accel_backend import xp, to_numpy, from_numpy

# You will keep modifying te_greens.py; we import from it on purpose.
from te_greens import gyy_TE
from te_greens import dgyy_dzobs_TE
from tm_greens import gxx_TM
from tm_greens import dgxx_dzobs_TM
from tm_greens import gzx_TM


def J0_series(x: np.ndarray) -> np.ndarray:
    """
    Bessel function of the first kind J0(x).

    Delegates to ``scipy.special.jv`` (compiled C, accurate for all x).
    The function name is kept for backward compatibility.
    """
    return jv(0, np.asarray(x, dtype=np.complex128))

def J2_series(x: np.ndarray) -> np.ndarray:
    """
    Bessel function of the first kind J2(x).

    Delegates to ``scipy.special.jv`` (compiled C, accurate for all x).
    The function name is kept for backward compatibility.
    """
    return jv(2, np.asarray(x, dtype=np.complex128))

def gauss_legendre(n: int, a: float = -1.0, b: float = 1.0):
    """
    Gauss-Legendre quadrature nodes and weights on [a, b].

    Uses scipy.special.roots_legendre for nodes/weights on [-1, 1],
    then linearly transforms to [a, b].
    Returns (nodes, weights) arrays of length *n*.
    """
    xi, wi = roots_legendre(n)
    scale = (b - a) / 2.0
    shift = (a + b) / 2.0
    return shift + scale * xi, wi * scale

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
    # Green's function engines use NumPy (complex arithmetic)
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

    # --- Bessel factor (scipy, compiled C) ---
    J0 = J0_series(kps * rho)
    J2 = J2_series(kps * rho)

    # --- Move to accelerated backend for integrand assembly ---
    # On Apple Silicon with MLX this runs on the GPU;
    # otherwise xp is just numpy and from_numpy/to_numpy are no-ops.
    _kps   = from_numpy(kps)
    _wts   = from_numpy(wts)
    _J0    = from_numpy(J0)
    _J2    = from_numpy(J2)
    _Gyy   = from_numpy(Gyykp)
    _DGyy  = from_numpy(np.conj(DGyykp))
    _Gxx   = from_numpy(Gxxkp)
    _DGxx  = from_numpy(np.conj(DGxxkp))
    _Gzx_c = from_numpy(np.conj(Gzxkp))

    pref = 1.0 / np.pi
    _kps2 = _kps * _kps

    # --- integrands (accelerated element-wise ops) ---
    ig1  = _kps * _J0 * _Gyy
    ig2  = _kps * _J0 * _DGyy
    ig3  = _kps * _J2 * _Gyy
    ig4  = _kps * _J2 * _DGyy
    ig5  = _kps * _J0 * _Gxx
    ig6  = _kps * _J0 * _DGxx
    ig7  = _kps * _J2 * _Gxx
    ig8  = _kps * _J2 * _DGxx
    ig9  = _kps2 * _J0 * _Gzx_c
    ig10 = _kps2 * _J2 * _Gzx_c

    # --- Gauss-Legendre weighted sums (accelerated dot) ---
    I1  = pref * xp.dot(_wts, ig1)
    I2  = pref * xp.dot(_wts, ig2)
    I3  = pref * xp.dot(_wts, ig3)
    I4  = pref * xp.dot(_wts, ig4)
    I5  = pref * xp.dot(_wts, ig5)
    I6  = pref * xp.dot(_wts, ig6)
    I7  = pref * xp.dot(_wts, ig7)
    I8  = pref * xp.dot(_wts, ig8)
    I9  = pref * xp.dot(_wts, ig9)
    I10 = pref * xp.dot(_wts, ig10)

    # --- combine (back to Python scalars) ---
    I1, I2, I3, I4, I5 = complex(I1), complex(I2), complex(I3), complex(I4), complex(I5)
    I6, I7, I8, I9, I10 = complex(I6), complex(I7), complex(I8), complex(I9), complex(I10)
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
