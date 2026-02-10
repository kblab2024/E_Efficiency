"""
k_parallel -> real-space (rho) integrator for multilayer Green's functions.

Design goal (your workflow):
    - Keep te_greens.py as the "engine" (you will keep editing it).
    - This file ONLY:
        (1) imports te_greens.gyy_TE (and later TM engines)
        (2) performs angular integration -> Bessel J0
        (3) performs k_parallel radial integral
        (4) plots results
using only numpy and matplotlib (no scipy).

GPU acceleration:
    All k_parallel evaluations are batched into a single vectorised call.
    When CuPy is available (NVIDIA GPU), the entire batch runs on the GPU.
    Otherwise NumPy is used on the CPU.

Math (2D in-plane Fourier/Bessel transform):
    G_yy(rho) = (1/(2π)) ∫_0^{∞} k_parallel * J0(k_parallel*rho) * G_yy(k_parallel) dk_parallel

Notes:
    - In te_greens.py, q_list is dimensionless and exponentials use exp(q*k0*z).
      Here, kp has physical dimension (1/length), rho has length.
    - You choose k_parallel_max and quadrature settings.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from gpu_config import xp, to_numpy, trapz

# You will keep modifying te_greens.py; we import from it on purpose.
from te_greens import gyy_TE
from te_greens import dgyy_dzobs_TE
from tm_greens import gxx_TM
from tm_greens import dgxx_dzobs_TM
from tm_greens import gzx_TM


def J0_series(x):
    """
    Minimal, dependency-free J0(x) approximation (series, stable for small/moderate x).

    J0(x) = Σ_{m=0}^∞ (-1)^m (x^2/4)^m / (m!)^2

    This is fine for plotting / debugging. If you later want faster & more accurate:
        - use scipy.special.j0

    Works with both NumPy and CuPy arrays.
    """
    x = xp.asarray(x, dtype=xp.complex128)
    x2_over_4 = (x * x) / 4.0

    # Adaptive-ish truncation: more terms for larger |x|
    max_abs = float(xp.max(xp.abs(x)))
    if max_abs < 5:
        M = 40
    elif max_abs < 20:
        M = 80
    else:
        M = 140

    out = xp.zeros_like(x, dtype=xp.complex128)
    term = xp.ones_like(x, dtype=xp.complex128)
    out += term
    for m in range(1, M):
        term *= (-x2_over_4) / (m * m)
        out += term
    return out

def J2_series(x):
    """
    Minimal, dependency-free J2(x) approximation (series).

    J2(x) = Σ_{m=0}^∞ (-1)^m (x/2)^(2m+2) / (m!(m+2)!)

    Works with both NumPy and CuPy arrays.
    """
    x = xp.asarray(x, dtype=xp.complex128)

    max_abs = float(xp.max(xp.abs(x)))
    if max_abs < 5:
        M = 50
    elif max_abs < 20:
        M = 120
    else:
        M = 220

    out = xp.zeros_like(x, dtype=xp.complex128)

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
    use_trapz: bool = True,
) -> complex:

    # --- midpoint grid (avoid hitting branch point exactly) ---
    dk  = k_parallel_max / num_k
    kps = (np.arange(num_k, dtype=float) + 0.5) * dk
    kps_dev = xp.asarray(kps)

    # --- Batch-evaluate all Green's functions at once (GPU-accelerated) ---
    Gyykp  = gyy_TE(n_list, d_list, layer_src, z_src, layer_obs, z_obs, k0, kps_dev)
    DGyykp = dgyy_dzobs_TE(n_list, d_list, layer_src, z_src, layer_obs, z_obs, k0, kps_dev)
    Gxxkp  = gxx_TM(n_list, d_list, layer_src, z_src, layer_obs, z_obs, k0, kps_dev)
    DGxxkp = dgxx_dzobs_TM(n_list, d_list, layer_src, z_src, layer_obs, z_obs, k0, kps_dev)
    Gzxkp  = gzx_TM(n_list, d_list, layer_src, z_src, layer_obs, z_obs, k0, kps_dev)

    # --- Bessel factor ---
    J0 = J0_series(kps_dev * rho)
    J2 = J2_series(kps_dev * rho)
    pref = 1.0 / xp.asarray(np.pi)

    # --- two integrands ---
    integrand_1 = kps_dev * J0 * Gyykp
    integrand_2 = kps_dev * J0 * xp.conj(DGyykp)
    integrand_3 = kps_dev * J2 * Gyykp
    integrand_4 = kps_dev * J2 * xp.conj(DGyykp)
    integrand_5 = kps_dev * J0 * Gxxkp
    integrand_6 = kps_dev * J0 * xp.conj(DGxxkp)
    integrand_7 = kps_dev * J2 * Gxxkp
    integrand_8 = kps_dev * J2 * xp.conj(DGxxkp)
    integrand_9 = kps_dev**2 * J0 * xp.conj(Gzxkp)
    integrand_10 = kps_dev**2 * J2 * xp.conj(Gzxkp)

    # --- integrate ---
    if use_trapz:
        I1 = pref * trapz(integrand_1, kps_dev)
        I2 = pref * trapz(integrand_2, kps_dev)
        I3 = pref * trapz(integrand_3, kps_dev)  
        I4 = pref * trapz(integrand_4, kps_dev)
        I5 = pref * trapz(integrand_5, kps_dev)
        I6 = pref * trapz(integrand_6, kps_dev)
        I7 = pref * trapz(integrand_7, kps_dev)
        I8 = pref * trapz(integrand_8, kps_dev)
        I9 = pref * trapz(integrand_9, kps_dev)
        I10 = pref * trapz(integrand_10, kps_dev)
        result = I1*I2 + I3*I4 + I5*I6 + I7*I8 + 1j*I5*I9 + 1j*I5*I10 + I5*I2 + I7*I4 + I1*I6+ I3*I8 + 1j*I1*I9 + 1j*I1*I10
        return complex(to_numpy(result))

    # If you ever turn on Simpson later, you'd want an endpoint grid (not midpoint).
    # For now keep your original guard:
    if num_k % 2 == 0:
        raise ValueError("Simpson needs odd num_k. Set num_k to an odd integer or use_trapz=True.")

    h = (k_parallel_max - 0.0) / (num_k - 1)  # 你說今晚先別管它 OK
    S1 = integrand_1[0] + integrand_1[-1] + 4.0 * xp.sum(integrand_1[1:-1:2]) + 2.0 * xp.sum(integrand_1[2:-2:2])
    S2 = integrand_2[0] + integrand_2[-1] + 4.0 * xp.sum(integrand_2[1:-1:2]) + 2.0 * xp.sum(integrand_2[2:-2:2])
    S3 = integrand_3[0] + integrand_3[-1] + 4.0 * xp.sum(integrand_3[1:-1:2]) + 2.0 * xp.sum(integrand_3[2:-2:2])
    S4 = integrand_4[0] + integrand_4[-1] + 4.0 * xp.sum(integrand_4[1:-1:2]) + 2.0 * xp.sum(integrand_4[2:-2:2])
    S5 = integrand_5[0] + integrand_5[-1] + 4.0 * xp.sum(integrand_5[1:-1:2]) + 2.0 * xp.sum(integrand_5[2:-2:2])
    S6 = integrand_6[0] + integrand_6[-1] + 4.0 * xp.sum(integrand_6[1:-1:2]) + 2.0 * xp.sum(integrand_6[2:-2:2])
    S7 = integrand_7[0] + integrand_7[-1] + 4.0 * xp.sum(integrand_7[1:-1:2]) + 2.0 * xp.sum(integrand_7[2:-2:2])
    S8 = integrand_8[0] + integrand_8[-1] + 4.0 * xp.sum(integrand_8[1:-1:2]) + 2.0 * xp.sum(integrand_8[2:-2:2])
    S9 = integrand_9[0] + integrand_9[-1] + 4.0 * xp.sum(integrand_9[1:-1:2]) + 2.0 * xp.sum(integrand_9[2:-2:2])
    S10 = integrand_10[0] + integrand_10[-1] + 4.0 * xp.sum(integrand_10[1:-1:2]) + 2.0 * xp.sum(integrand_10[2:-2:2])  
    I1 = pref * (h / 3.0) * S1
    I2 = pref * (h / 3.0) * S2
    I3 = pref * (h / 3.0) * S3
    I4 = pref * (h / 3.0) * S4
    I5 = pref * (h / 3.0) * S5
    I6 = pref * (h / 3.0) * S6
    I7 = pref * (h / 3.0) * S7
    I8 = pref * (h / 3.0) * S8
    I9 = pref * (h / 3.0) * S9
    I10 = pref * (h / 3.0) * S10
    result = I1*I2 + I3*I4 + I5*I6 + I7*I8 + 1j*I5*I9 + 1j*I5*I10 + I5*I2 + I7*I4 + I1*I6 + I3*I8 + 1j*I1*I9 + 1j*I1*I10
    return complex(to_numpy(result))

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
    num_k = 2001  # odd recommended if you switch to Simpson

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
            use_trapz=True,
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
