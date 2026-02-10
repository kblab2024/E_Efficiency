"""
TM Green's function engine for multilayer structures.

Goal:
    Compute TM Green's tensor components:
        g_xx, g_zz, g_xz, g_zx

This module REUSES the multilayer geometry and reflection machinery
defined in te_greens.py, to ensure identical definitions of:

    - q_list
    - interface parameters
    - RF / RB recursion

Only TM-specific source normalization, propagation, and assembly
are implemented here.

All functions accept kp as a scalar or array for GPU-accelerated batch
evaluation over k_parallel using CuPy (GPU) or NumPy (CPU).
"""

from gpu_config import xp
import numpy as np

# ------------------------------------------------------------
# Import shared multilayer machinery (Level 1 & 2)
# ------------------------------------------------------------
from te_greens import (compute_q_list,compute_interface_params_up_down,compute_RF_all,compute_RB_all,)

# ------------------------------------------------------------
# Level 3: same-layer TM source amplitudes
# ------------------------------------------------------------
def TM_f1xf2x_same_layer(q_list, R_down, R_up, layer_src, z_src, k0, kp):
    """
    TM source amplitudes for g_xx component.

    Returns:
        f1x : downward-propagating TM amplitude
        f2x : upward-propagating TM amplitude
    """

    q  = q_list[layer_src]
    RF = R_down[layer_src]
    RB = R_up[layer_src]

    kp = xp.asarray(kp, dtype=xp.complex128)
    e2 = (kp**2 - (q * k0)**2) / (k0**2)
    ez  = xp.exp(+q * k0 * z_src)
    emz = xp.exp(-q * k0 * z_src)

    den = 2 * e2 * (k0 / q) * (1 - RF * RB)
    f1x = -(ez  + emz * RB) / den
    f2x = -(emz + ez  * RF) / den

    return f1x, f2x

def TM_df1xdf2x_same_layer(q_list, R_down, R_up, layer_src, z_src, k0, kp):
    q  = q_list[layer_src]
    RF = R_down[layer_src]
    RB = R_up[layer_src]

    kp = xp.asarray(kp, dtype=xp.complex128)
    e2 = (kp**2 - (q * k0)**2) / (k0**2)
    ez  = xp.exp(+q * k0 * z_src)   # E_+
    emz = xp.exp(-q * k0 * z_src)   # E_-

    den = 2 * e2 * (k0 / q) * (1 - RF * RB)
    df1x = -(q*k0) * (ez - RB*emz) / den
    df2x = +(q*k0) * (emz - RF*ez) / den
    return df1x, df2x


def TM_f1zf2z_same_layer(q_list, R_down, R_up, layer_src, z_src, k0, kp):
 
    q  = q_list[layer_src]
    RF = R_down[layer_src]
    RB = R_up[layer_src]
    kp = xp.asarray(kp, dtype=xp.complex128)
    e2 = (kp**2 - (q * k0)**2) / (k0**2)
    ez  = xp.exp(+q * k0 * z_src)
    emz = xp.exp(-q * k0 * z_src)
    den = (2 * k0**2 * e2 / (1j * kp)) * (1 - RF * RB)
    f1z = ( ez  - emz * RB ) / den
    f2z = (-emz + ez  * RF ) / den

    return f1z, f2z

def propagate_down_TM(f1_src, q_list, R_down, layer_src, layer_obs,k0, d_list, eps_list):

    if layer_obs <= layer_src:
        raise ValueError("propagate_down_TM requires layer_obs > layer_src")
    
    f = f1_src

    for l in range(layer_src, layer_obs):
        d_l = d_list[l] if l < len(d_list) else 0.0
        ql  = q_list[l]
        qlp = q_list[l + 1]
        eps_l  = eps_list[l]
        eps_lp = eps_list[l + 1]

        wn = (qlp / ql) * (eps_l / eps_lp)
        wp = 1.0 + wn        # M_l^+
        wq = 1.0 - wn        # M_l^-
        Rnext = R_down[l + 1]
        T = 1.0 / (wp + wq * Rnext)
        x = xp.exp(-ql * k0 * d_l)
        f = T * x * f

    return f


def propagate_up_TM(f2_src, q_list, R_up, layer_src, layer_obs,k0, d_list, eps_list):

    if layer_obs >= layer_src:
        raise ValueError("propagate_up_TM requires layer_obs < layer_src")

    f = f2_src

    # l = layer_src-1, ..., layer_obs
    for l in range(layer_src - 1, layer_obs - 1, -1):

        d_l = d_list[l] if l < len(d_list) else 0.0
        ql   = q_list[l]
        qlp1 = q_list[l + 1]
        eps_l  = eps_list[l]
        eps_lp = eps_list[l + 1]
        wn = (ql / qlp1) * (eps_lp / eps_l)
        mp = 1.0 + wn     # m^+
        mm = 1.0 - wn     # m^-
        r_l = R_up[l]
        x = xp.exp(-ql * k0 * d_l)
        T = 1.0 / (mp + mm * x * r_l * x)
        f = x * T * f

    return f


# ------------------------------------------------------------
# Level 5: assemble TM Green's function components
# ------------------------------------------------------------
def gxx_TM(n_list, d_list, layer_src, z_src,layer_obs, z_obs,k0, kp):

    eps_list = xp.array(n_list, dtype=xp.complex128)**2
    q_list   = compute_q_list(n_list, k0, kp)
    R_down = compute_RF_all(n_list, d_list, k0, kp, polarization="TM")
    R_up   = compute_RB_all(n_list, d_list, k0, kp, polarization="TM")
    f1x_src, f2x_src = TM_f1xf2x_same_layer(q_list, R_down, R_up, layer_src, z_src, k0, kp)
    q_obs   = q_list[layer_obs]
    eps_obs = eps_list[layer_obs]
    RF_obs  = R_down[layer_obs]
    RB_obs  = R_up[layer_obs]

    # -------------------------
    # Case A: same layer (paper)
    # -------------------------
    if layer_obs == layer_src:
        q   = q_list[layer_src]
        eps = eps_list[layer_src]
        RF  = R_down[layer_src]
        RB  = R_up[layer_src]

        direct = (-q / (2.0 * k0 * eps)) * xp.exp(-q * k0 * abs(z_obs - z_src))
        refl   = (xp.exp(+q * k0 * z_obs) * RF * f1x_src
                  + xp.exp(-q * k0 * z_obs) * RB * f2x_src)
        Gxx = direct + refl
        return Gxx

    # -------------------------
    # Case B: obs below src
    # -------------------------
    if layer_obs > layer_src:
        f1x_at_obs_top = propagate_down_TM(
            f1x_src, q_list, R_down,
            layer_src, layer_obs,
            k0, d_list, eps_list
        )
        dress = (xp.exp(-q_obs * k0 * z_obs) + xp.exp(+q_obs * k0 * z_obs) * RF_obs)
        Gxx = dress * f1x_at_obs_top
        return Gxx

    # -------------------------
    # Case C: obs above src
    # -------------------------
    f2x_at_obs_top = propagate_up_TM(
        f2x_src, q_list, R_up,
        layer_src, layer_obs,
        k0, d_list, eps_list
    )
    dress = (xp.exp(+q_obs * k0 * z_obs) + xp.exp(-q_obs * k0 * z_obs) * RB_obs)
    Gxx = dress * f2x_at_obs_top
    return Gxx


def gzz_TM(n_list, d_list, layer_src, z_src, layer_obs, z_obs, k0, kp):
    """
    TM Green's function component G_zz for multilayer.

    - Same-layer: use your paper-form (the one you already wrote).
    - Cross-layer: use the 'general formula' route you specified:
          Gzz(z,z') = (kp^2 / q_obs^4) * (∂z ∂z') Gxx(z,z')    (z != z')
      with the rule: ∂_{z'} acts ONLY on f1/f2 (so we propagate df1/df2).
    """

    eps_list = xp.array(n_list, dtype=xp.complex128)**2
    q_list   = compute_q_list(n_list, k0, kp)
    kp = xp.asarray(kp, dtype=xp.complex128)

    # reflections (TM)
    R_down = compute_RF_all(n_list, d_list, k0, kp, polarization="TM")  # RF_all
    R_up   = compute_RB_all(n_list, d_list, k0, kp, polarization="TM")  # RB_all

    # ------------------------------------------------------------
    # Case A: same layer (paper form you used before)
    # ------------------------------------------------------------
    if layer_obs == layer_src:
        n   = layer_src
        q   = q_list[n]
        eps = eps_list[n]
        RF  = R_down[n]
        RB  = R_up[n]

        f1z, f2z = TM_f1zf2z_same_layer(q_list, R_down, R_up,layer_src, z_src, k0, kp)

        Gzz = (
            (kp**2) / (2 * k0**3 * eps * q)
            - (1j * kp / (q * k0)) * xp.exp(+q * k0 * z_obs) * RF * f1z
            + (1j * kp / (q * k0)) * xp.exp(-q * k0 * z_obs) * RB * f2z)
        return Gzz

    # ------------------------------------------------------------
    # Case B/C: cross layer (your df-propagation route)
    # ------------------------------------------------------------
    q_obs  = q_list[layer_obs]
    RF_obs = R_down[layer_obs]
    RB_obs = R_up[layer_obs]

    # source-side df1x/df2x at z_src (your code)
    df1x_src, df2x_src = TM_df1xdf2x_same_layer(q_list, R_down, R_up,layer_src, z_src, k0, kp)

    # -------------------------
    # obs below src  (use f1-branch)
    # -------------------------
    if layer_obs > layer_src:
        df1x_at_obs_top = propagate_down_TM(df1x_src, q_list, R_down,layer_src, layer_obs,k0, d_list, eps_list)
        # A(z) = exp(-qz) + exp(+qz) RF
        # A'(z)= -q exp(-qz) + q exp(+qz) RF
        Aprime = (-q_obs* k0) * xp.exp(-q_obs * k0 * z_obs) + (q_obs * k0) * xp.exp(+q_obs * k0 * z_obs) * RF_obs
        d2Gxx = Aprime * df1x_at_obs_top          # ∂z∂z' Gxx  (per your rule)
        Gzz   = (kp**2 / (q_obs**4 * k0**4)) * d2Gxx      # your prefactor
        return Gzz

    # -------------------------
    # obs above src  (use f2-branch)
    # -------------------------
    df2x_at_obs_top = propagate_up_TM(df2x_src, q_list, R_up,layer_src, layer_obs,k0, d_list, eps_list)

    # B(z) = exp(+qz) + exp(-qz) RB
    # B'(z)= +q exp(+qz) - q exp(-qz) RB
    Bprime = (q_obs * k0) * xp.exp(+q_obs * k0 * z_obs) - (q_obs * k0) * xp.exp(-q_obs * k0 * z_obs) * RB_obs
    d2Gxx = Bprime * df2x_at_obs_top
    Gzz   = (kp**2 / (q_obs**4 * k0**4)) * d2Gxx
    return Gzz


def gxz_TM(n_list, d_list,layer_src, z_src,layer_obs, z_obs,k0, kp):
    """
    TM Green's function component G_xz.

    Conventions:
        - q_list is dimensionless: q = -sqrt(kp^2 - eps*k0^2)/k0
        - exp factors always use (q * k0 * z)
        - RF = R_down, RB = R_up
        - IMPORTANT: we assume f1z/f2z ALREADY include the prefactor (-i*kp)/(2*k0^2*eps)
          through the denominator in TM_f1zf2z_same_layer().
          Therefore cross-layer terms do NOT multiply an extra prefactor.
    """
    eps_list = xp.array(n_list, dtype=xp.complex128)**2
    kp = xp.asarray(kp, dtype=xp.complex128)

    # Level 1
    q_list = compute_q_list(n_list, k0, kp)

    # Level 2 (TM reflections)
    R_down = compute_RF_all(n_list, d_list, k0, kp, polarization="TM")  # RF_all
    R_up   = compute_RB_all(n_list, d_list, k0, kp, polarization="TM")  # RB_all

    q_obs  = q_list[layer_obs]
    RF_obs = R_down[layer_obs]
    RB_obs = R_up[layer_obs]

    # Level 3: source amplitudes for z-branch (same-layer definition at z_src)
    f1z_src, f2z_src = TM_f1zf2z_same_layer(q_list, R_down, R_up,layer_src, z_src, k0, kp)

    # ============================================================
    # Case A: same layer
    # (your corrected paper form: prefactor is already inside f1z/f2z)
    # ============================================================
    if layer_obs == layer_src:
        q = q_list[layer_src]
        RF = R_down[layer_src]
        RB = R_up[layer_src]

        sgn = float(xp.sign(z_obs - z_src))

        # NOTE: keep exactly this structure (no extra pref outside)
        Gxz = ( -sgn * xp.exp(-q * k0 * abs(z_obs - z_src)) 
                     + xp.exp(+q * k0 * z_obs) * RF * f1z_src 
                     + xp.exp(-q * k0 * z_obs) * RB * f2z_src)

        return Gxz

    # ============================================================
    # Case B: obs below src  (layer_obs > layer_src) -> use f1z branch
    # ============================================================
    if layer_obs > layer_src:
        f1z_at_obs = propagate_down_TM(f1z_src, q_list, R_down,layer_src, layer_obs,k0, d_list, eps_list)

        Gxz = (
            xp.exp(-q_obs * k0 * z_obs) * f1z_at_obs
            + xp.exp(+q_obs * k0 * z_obs) * RF_obs * f1z_at_obs)

        return Gxz

    # ============================================================
    # Case C: obs above src (layer_obs < layer_src) -> use f2z branch
    # ============================================================
    f2z_at_obs = propagate_up_TM(f2z_src, q_list, R_up,layer_src, layer_obs,k0, d_list, eps_list)

    Gxz = (
        xp.exp(+q_obs * k0 * z_obs) * f2z_at_obs
        + xp.exp(-q_obs * k0 * z_obs) * RB_obs * f2z_at_obs)

    return Gxz


def gzx_TM(n_list, d_list,layer_src, z_src,layer_obs, z_obs,k0, kp):
    """
    TM Green's function component G_zx.

    Conventions:
        q_list is dimensionless, exp uses (q*k0*z).
        RF = R_down, RB = R_up.

    Structure:
        - same-layer: keep your paper form (with sgn term)
        - cross-layer: NO sgn term, use propagation + local dressing
          based on your screenshot:
              Gxx = (e^{-qz} + e^{+qz} RF) f1(z')   for z > z'
              Gxx = (e^{+qz} + e^{-qz} RB) f2(z')   for z < z'
          and
              Gzx = -(i kp / q^2) ∂_z Gxx
    """
    eps_list = xp.array(n_list, dtype=xp.complex128)**2
    kp = xp.asarray(kp, dtype=xp.complex128)
    q_list   = compute_q_list(n_list, k0, kp)

    R_down = compute_RF_all(n_list, d_list, k0, kp, polarization="TM")
    R_up   = compute_RB_all(n_list, d_list, k0, kp, polarization="TM")

    # source amplitudes for x-branch (defined at z_src in source layer)
    f1x_src, f2x_src = TM_f1xf2x_same_layer(q_list, R_down, R_up, layer_src, z_src, k0, kp)

    q_obs  = q_list[layer_obs]
    RF_obs = R_down[layer_obs]
    RB_obs = R_up[layer_obs]

    # ============================================================
    # Case A: same layer (keep your current paper form)
    # ============================================================
    if layer_obs == layer_src:
        q  = q_list[layer_src]
        RF = R_down[layer_src]
        RB = R_up[layer_src]

        sgn = float(xp.sign(z_obs - z_src))

        gzx = ( -1j * kp / (2 * k0**2) * sgn * xp.exp(-q * k0 * abs(z_obs - z_src))
            + (1j * kp / q) * ( - xp.exp(+q * k0 * z_obs) * RF * f1x_src + xp.exp(-q * k0 * z_obs) * RB * f2x_src ))
        return gzx

    # ============================================================
    # Case B: obs below src  (layer_obs > layer_src) -> f1x branch
    # ============================================================
    if layer_obs > layer_src:
        f1x_at_obs = propagate_down_TM(f1x_src, q_list, R_down,layer_src, layer_obs,k0, d_list, eps_list)

        # From Gxx = (e^{-qz} + e^{+qz} RF) f1
        # ∂zGxx = (-q e^{-qz} + q e^{+qz} RF) f1
        # Gzx = -(i kp / q^2) ∂zGxx  -> (-i kp / q) * (-e^{-qz} + e^{+qz} RF) f1
        gzx = (-1j * kp / (q_obs*k0) ) * ( 
            - xp.exp(-q_obs * k0 * z_obs) * f1x_at_obs
            + xp.exp(+q_obs * k0 * z_obs) * RF_obs * f1x_at_obs)
        return gzx

    # ============================================================
    # Case C: obs above src (layer_obs < layer_src) -> f2x branch
    # ============================================================
    f2x_at_obs = propagate_up_TM(f2x_src, q_list, R_up,layer_src, layer_obs,k0, d_list, eps_list)

    # From Gxx = (e^{+qz} + e^{-qz} RB) f2
    # ∂zGxx = ( +q e^{+qz} - q e^{-qz} RB) f2
    # Gzx = -(i kp / q^2) ∂zGxx -> (-i kp / q) * ( +e^{+qz} - e^{-qz} RB) f2
    gzx = (-1j * kp / (q_obs*k0) ) * (
        + xp.exp(+q_obs * k0 * z_obs) * f2x_at_obs
        - xp.exp(-q_obs * k0 * z_obs) * RB_obs * f2x_at_obs)
    return gzx

def dgxx_dzobs_TM( n_list, d_list, layer_src, z_src, layer_obs, z_obs, k0, kp, tol: float = 1e-12,):
    """
    d/dz_obs of Gxx for TM polarization.

    This version *raises an error* for the same-layer cusp at z_obs == z_src
    (or |z_obs - z_src| < tol), which is appropriate when doing Poynting-flux
    surface integrals: the observation surface should not pass through the source.

    All arrays are vectorised over k_parallel.

    Parameters
    ----------
    tol : float
        If abs(z_obs - z_src) < tol (same layer), raise ValueError.
        Tune based on your z units/scale.
    """

    eps_list = xp.array(n_list, dtype=xp.complex128)**2
    kp = xp.asarray(kp, dtype=xp.complex128)
    q_list   = compute_q_list(n_list, k0, kp)

    R_down = compute_RF_all(n_list, d_list, k0, kp, polarization="TM")
    R_up   = compute_RB_all(n_list, d_list, k0, kp, polarization="TM")

    f1x_src, f2x_src = TM_f1xf2x_same_layer(q_list, R_down, R_up, layer_src, z_src, k0, kp)

    q_obs   = q_list[layer_obs]
    RF_obs  = R_down[layer_obs]
    RB_obs  = R_up[layer_obs]

    # -------------------------
    # Case A: same layer
    # -------------------------
    if layer_obs == layer_src:
        q   = q_list[layer_src]
        eps = eps_list[layer_src]
        RF  = R_down[layer_src]
        RB  = R_up[layer_src]

        a  = q * k0
        dz = (z_obs - z_src)

        if abs(dz) < tol:
            raise ValueError(
                "dgxx_dzobs_TM (same-layer): |z_obs - z_src| < tol -> cusp at the source.\n"
                "Move the observation surface away from the source location."
            )

        s = float(xp.sign(dz))  # now guaranteed to be ±1 for real dz

        # direct = (-q/(2*k0*eps)) * exp(-a*|dz|)
        # ddirect = (q^2/(2*eps)) * exp(-a*|dz|) * sign(dz)
        ddirect = (q*q/(2.0*eps)) * xp.exp(-a * abs(dz)) * s

        # refl = exp(+a z_obs)*RF*f1x_src + exp(-a z_obs)*RB*f2x_src
        # drefl = a*( exp(+a z_obs)*RF*f1x_src - exp(-a z_obs)*RB*f2x_src )
        drefl = a * (
            xp.exp(+a * z_obs) * RF * f1x_src
            - xp.exp(-a * z_obs) * RB * f2x_src
        )

        return ddirect + drefl

    # -------------------------
    # Case B: obs below src
    # -------------------------
    if layer_obs > layer_src:
        f1x_at_obs_top = propagate_down_TM(
            f1x_src, q_list, R_down,
            layer_src, layer_obs,
            k0, d_list, eps_list
        )

        a_obs = q_obs * k0
        # dress = exp(-a_obs z_obs) + exp(+a_obs z_obs)*RF_obs
        # ddress = -a_obs exp(-a_obs z_obs) + a_obs exp(+a_obs z_obs)*RF_obs
        ddress = (-a_obs) * xp.exp(-a_obs * z_obs) + (a_obs) * xp.exp(+a_obs * z_obs) * RF_obs

        return ddress * f1x_at_obs_top

    # -------------------------
    # Case C: obs above src
    # -------------------------
    f2x_at_obs_top = propagate_up_TM(
        f2x_src, q_list, R_up,
        layer_src, layer_obs,
        k0, d_list, eps_list
    )

    a_obs = q_obs * k0
    # dress = exp(+a_obs z_obs) + exp(-a_obs z_obs)*RB_obs
    # ddress = a_obs exp(+a_obs z_obs) - a_obs exp(-a_obs z_obs)*RB_obs
    ddress = (a_obs) * xp.exp(+a_obs * z_obs) + (-a_obs) * xp.exp(-a_obs * z_obs) * RB_obs

    return ddress * f2x_at_obs_top