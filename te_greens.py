"""
TE Green's function engine for multilayer structures.

Goal:
    Compute G_yy(l_obs, l_src, z_obs, z_src, k_parallel)
    using a clean 5-level architecture:

        Level 1: q list (per layer)
        Level 2: TE reflection recursion (R_down, R_up)
        Level 3: same-layer f1, f2 amplitudes (local)
        Level 4: propagation between layers (down/up)
        Level 5: assemble full TE Green's function G_yy

This is a collaborative development skeleton.

All functions accept kp as a scalar or array for GPU-accelerated batch
evaluation over k_parallel using CuPy (GPU) or NumPy (CPU).
"""


from gpu_config import xp
import numpy as np
from multilayer_reflectance import RF_multilayer, RB_multilayer  

# ------------------------------------------------------------
# Level 1: q(k_parallel) per layer
# ------------------------------------------------------------
def compute_q_list(n_list, k0, kp):
    """  Return q_list for all layers.  kp can be scalar or array."""
    eps_list = [n**2 for n in n_list]
    kp = xp.asarray(kp, dtype=xp.complex128)
    q_list = [xp.sqrt(kp**2 - eps * k0**2) / k0 for eps in eps_list]
    return q_list

def compute_interface_params_up_down(n_list, d_list, k0, kp, polarization="TE"):
    """
    Returns two lists:
        params_down[L] : for crossing interface L downward (layer L → L+1)
        params_up[L]   : for crossing interface L upward   (layer L+1 → L)

    Each params_down[L] / params_up[L] contains:
        q_up      : q in layer L      (source side)
        q_down    : q in layer L+1    (destination side)
        x         : exp(-q_up * k0 * d_list[L])  # SAME for up/down
        wp, wq    : jump coefficients for recursion

    kp can be scalar or array (vectorised over k_parallel).
    """
    N = len(n_list)
    eps = xp.array(n_list, dtype=xp.float64) ** 2
    kp2 = xp.asarray(kp, dtype=xp.complex128) ** 2
    params_down = []
    params_up = []

    for L in range(N - 1):
        q_up_layer   = xp.sqrt(kp2 - eps[L]   * k0**2) / k0
        q_down_layer = xp.sqrt(kp2 - eps[L+1] * k0**2) / k0
        xL = xp.exp(- q_up_layer * k0 * d_list[L])

        if polarization.upper() == "TE":
            wn_down = q_down_layer / q_up_layer
            wn_up   = q_up_layer   / q_down_layer
        else:   # TM
            wn_down = (q_down_layer / q_up_layer) * (eps[L] / eps[L+1])
            wn_up   = (q_up_layer   / q_down_layer) * (eps[L+1] / eps[L])

        wm = 1.0
        wp_down = wm + wn_down
        wq_down = wm - wn_down
        wp_up = wm + wn_up
        wq_up = wm - wn_up

        params_down.append({"q_up": q_up_layer,"q_down": q_down_layer,"x": xL,"wp": wp_down,"wq": wq_down})
        params_up.append({"q_up": q_up_layer,"q_down": q_down_layer,"x": xL,"wp": wp_up,"wq": wq_up})

    return params_down, params_up

def compute_RF_all(n_list, d_list, k0, kp, polarization="TE"):
    """
    RF_all[j] = reflection seen when standing at TOP of layer j, looking downward.
    Implemented using params_down (wp, wq, x), equivalent to original RF_multilayer().

    kp can be scalar or array (vectorised over k_parallel).
    """
    params_down, params_up = compute_interface_params_up_down(n_list, d_list, k0, kp, polarization)
    N = len(n_list)
    kp_arr = xp.asarray(kp, dtype=xp.complex128)
    RF_all = [xp.zeros_like(kp_arr)] * N
    r_eff = xp.zeros_like(kp_arr)

    # recursion from bottom to top
    # interfaces are L = 0...(N-2), between layer L and L+1
    for L in range(N-2, -1, -1):
        p = params_down[L]
        wp = p["wp"]
        wq = p["wq"]
        x  = p["x"]      # exponential through layer L (downward)

        # downward recursion formula:
        # r_new = x^2 * (wq + wp * r_old) / (wp + wq * r_old)
        num = wq + wp * r_eff
        den = wp + wq * r_eff
        r_eff = x * (num / den) * x   # x^2 * (bl/al)
        RF_all[L] = r_eff

    RF_all[N-1] = xp.zeros_like(kp_arr)   # bottom semi-infinite region
    return RF_all


def compute_RB_all(n_list, d_list, k0, kp, polarization="TE"):
    """
    RB_all[j] = reflection seen when standing at BOTTOM of layer j, looking upward.
    Implemented using params_up (wp, wq, x), equivalent to original RB_multilayer().

    kp can be scalar or array (vectorised over k_parallel).
    """
    params_down, params_up = compute_interface_params_up_down(n_list, d_list, k0, kp, polarization)
    N = len(n_list)
    kp_arr = xp.asarray(kp, dtype=xp.complex128)
    RB_all = [xp.zeros_like(kp_arr)] * N
    r_eff = xp.zeros_like(kp_arr)

    # recursion from top to bottom
    for L in range(0, N-1):
        p = params_up[L]
        wp = p["wp"]
        wq = p["wq"]
        x  = p["x"]      # exponential through layer L (upward)
        xr = x * r_eff * x
        num = wq + wp * xr
        den = wp + wq * xr
        r_eff = num / den
        RB_all[L+1] = r_eff

    RB_all[0] = xp.zeros_like(kp_arr)   # top semi-infinite region
    return RB_all

# ------------------------------------------------------------
# Level 3: same-layer source amplitudes f1, f2 (local)
# ------------------------------------------------------------
def TE_f1yf2y_same_layer(q_list, R_down, R_up, layer_src, z_src, k0):
    """
    Compute TE source amplitudes (f1y, f2y) inside the source layer.
    f1y : downward-propagating TE wave amplitude
    f2y : upward-propagating TE wave amplitude

    All arrays are vectorised over k_parallel.
    """
    q  = q_list[layer_src]
    RF = R_down[layer_src]
    RB = R_up[layer_src]
    ez  = xp.exp(+q * k0 * z_src)
    emz = xp.exp(-q * k0 * z_src)
    den = 2 * q * k0 * (1 - RF * RB)
    f1y = ( ez  + emz * RB ) / den
    f2y = ( emz + ez  * RF ) / den
    return f1y, f2y


# ------------------------------------------------------------
# Level 4: propagate to another layer
# ------------------------------------------------------------
def propagate_down_TE(f1_src, q_list, R_down, R_up, layer_src, layer_obs, k0, d_list):
    if layer_obs <= layer_src:
        raise ValueError("propagate_down_TE requires layer_obs > layer_src")

    f = f1_src
    for l in range(layer_src, layer_obs):
        d_l = d_list[l] if l < len(d_list) else 0.0
        ql  = q_list[l]
        qlp = q_list[l+1]
        wn = qlp / ql
        wp = 1.0 + wn          # M_l^+
        wq = 1.0 - wn          # M_l^-
        Rnext = R_down[l+1]    # R_{l+1}
        T = 1.0 / (wp + wq * Rnext)       # T_l (paper)
        x = xp.exp(-ql * k0 * d_l)        # e^{-q_l d_l}
        f = T * x * f                     # f_{l+1} = T_l e^{-q_l d_l} f_l
    return f

def propagate_up_TE(f2_src, q_list, R_up, layer_src, layer_obs, k0, d_list):  
    if layer_obs >= layer_src:
        raise ValueError("propagate_up_TE requires layer_obs < layer_src")
    
    f = f2_src
    for l in range(layer_src - 1, layer_obs - 1, -1):
        d_l = d_list[l] if l < len(d_list) else 0.0
        ql   = q_list[l]
        qlp1 = q_list[l + 1]
        x = xp.exp(-ql * k0 * d_l)
        wn = ql / qlp1
        mp = 1.0 + wn     # m^+
        mm = 1.0 - wn     # m^-
        r_l = R_up[l]
        T = 1.0 / (mp + mm * x * r_l * x)
        f = x * T * f
    return f


# ------------------------------------------------------------
# Level 5: assemble G_yy
# ------------------------------------------------------------
def gyy_TE(n_list, d_list, layer_src, z_src, layer_obs, z_obs, k0, kp):

    q_list = compute_q_list(n_list, k0, kp)
    R_down = compute_RF_all(n_list, d_list, k0, kp, polarization="TE")
    R_up   = compute_RB_all(n_list, d_list, k0, kp, polarization="TE")
    f1y_src, f2y_src = TE_f1yf2y_same_layer(q_list, R_down, R_up, layer_src, z_src, k0)
    q_obs  = q_list[layer_obs]
    RF_obs = R_down[layer_obs]
    RB_obs = R_up[layer_obs]
    Gyy = xp.zeros_like(xp.asarray(kp, dtype=xp.complex128))

    if layer_obs == layer_src:
        q  = q_list[layer_src]
        RF = R_down[layer_src]
        RB = R_up[layer_src]
        direct = (1.0 / (2.0 * q * k0)) * xp.exp(-q * k0 * abs(z_obs - z_src))
        cavity = ( xp.exp(+q * k0 * z_obs) * RF * f1y_src + xp.exp(-q * k0 * z_obs) * RB * f2y_src )
        Gyy = direct + cavity

    elif layer_obs > layer_src:
        f1y_at_obs_top = propagate_down_TE( f1y_src, q_list, R_down,layer_src, layer_obs, k0, d_list )
        Gyy = ( xp.exp(-q_obs * k0 * z_obs) + xp.exp(+q_obs * k0 * z_obs) * RF_obs) * f1y_at_obs_top

    else:
        f2y_at_obs_top = propagate_up_TE( f2y_src, q_list, R_up,layer_src, layer_obs, k0, d_list )
        Gyy = (xp.exp(+q_obs * k0 * z_obs)+ xp.exp(-q_obs * k0 * z_obs) * RB_obs) * f2y_at_obs_top

    return Gyy

def dgyy_dzobs_TE(n_list, d_list,layer_src, z_src,layer_obs, z_obs,k0, kp,tol: float = 1e-12,):
    """
    d/dz_obs of Gyy for TE polarization.

    This version raises an error for the same-layer cusp at z_obs == z_src
    (or |z_obs - z_src| < tol), which is appropriate for Poynting-flux
    surface integrals (observation surface should not pass through the source).

    All arrays are vectorised over k_parallel.
    """

    q_list = compute_q_list(n_list, k0, kp)
    R_down = compute_RF_all(n_list, d_list, k0, kp, polarization="TE")
    R_up   = compute_RB_all(n_list, d_list, k0, kp, polarization="TE")

    f1y_src, f2y_src = TE_f1yf2y_same_layer(q_list, R_down, R_up, layer_src, z_src, k0)

    q_obs  = q_list[layer_obs]
    RF_obs = R_down[layer_obs]
    RB_obs = R_up[layer_obs]

    # -------------------------
    # Case A: same layer
    # -------------------------
    if layer_obs == layer_src:
        q  = q_list[layer_src]
        RF = R_down[layer_src]
        RB = R_up[layer_src]

        a  = q * k0
        dz = (z_obs - z_src)

        if abs(dz) < tol:
            raise ValueError(
                "dgyy_dzobs_TE (same-layer): |z_obs - z_src| < tol -> cusp at the source.\n"
                "Move the observation surface away from the source location."
            )

        s = float(xp.sign(dz))  # ±1 for real dz

        # direct = (1/(2 q k0)) * exp(-a*|dz|)
        # ddirect = (1/(2 q k0)) * exp(-a|dz|) * (-a*sign(dz)) = -(1/2)*exp(-a|dz|)*sign(dz)
        ddirect = (-0.5) * xp.exp(-a * abs(dz)) * s

        # cavity = exp(+a z_obs)*RF*f1y_src + exp(-a z_obs)*RB*f2y_src
        # dcavity = a*( exp(+a z_obs)*RF*f1y_src - exp(-a z_obs)*RB*f2y_src )
        dcavity = a * (
            xp.exp(+a * z_obs) * RF * f1y_src
            - xp.exp(-a * z_obs) * RB * f2y_src
        )

        return ddirect + dcavity

    # -------------------------
    # Case B: obs below src
    # -------------------------
    if layer_obs > layer_src:
        f1y_at_obs_top = propagate_down_TE(
            f1y_src, q_list, R_down,
            layer_src, layer_obs,
            k0, d_list
        )

        a_obs = q_obs * k0
        # dress = exp(-a_obs z_obs) + exp(+a_obs z_obs)*RF_obs
        # ddress = -a_obs exp(-a_obs z_obs) + a_obs exp(+a_obs z_obs)*RF_obs
        ddress = (-a_obs) * xp.exp(-a_obs * z_obs) + (a_obs) * xp.exp(+a_obs * z_obs) * RF_obs

        return ddress * f1y_at_obs_top

    # -------------------------
    # Case C: obs above src
    # -------------------------
    f2y_at_obs_top = propagate_up_TE(
        f2y_src, q_list, R_up,
        layer_src, layer_obs,
        k0, d_list
    )

    a_obs = q_obs * k0
    # dress = exp(+a_obs z_obs) + exp(-a_obs z_obs)*RB_obs
    # ddress = a_obs exp(+a_obs z_obs) - a_obs exp(-a_obs z_obs)*RB_obs
    ddress = (a_obs) * xp.exp(+a_obs * z_obs) + (-a_obs) * xp.exp(-a_obs * z_obs) * RB_obs

    return ddress * f2y_at_obs_top


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from gpu_config import to_numpy

    n_list = [1.0, 1.5, 1.3, 1.0]  # 四層結構
    d_list = [0.0, 2000, 1500]   # 四層結構
    wls = np.linspace(500, 900, 200)
    diff_RF = []
    diff_RB = []

    for pol in ["TE", "TM"]:
        diff_RF = []
        diff_RB = []

        for wl in wls:
            k0 = 2*np.pi / wl
            kp = 0.3*k0

            # OLD version
            RF_old = [to_numpy(RF_multilayer(n_list, d_list, pol, k0, kp, j))
                      for j in range(len(n_list))]
            RB_old = [to_numpy(RB_multilayer(n_list, d_list, pol, k0, kp, j))
                      for j in range(len(n_list))]

            # NEW version
            RF_new = [to_numpy(v) for v in compute_RF_all(n_list, d_list, k0, kp, pol)]
            RB_new = [to_numpy(v) for v in compute_RB_all(n_list, d_list, k0, kp, pol)]

            diff_RF.append(np.max(np.abs(np.array(RF_old) - np.array(RF_new))))
            diff_RB.append(np.max(np.abs(np.array(RB_old) - np.array(RB_new))))

        print(f"=== polarization: {pol} ===")
        print("max |RF_old - RF_new| =", np.max(diff_RF))
        print("max |RB_old - RB_new| =", np.max(diff_RB))
        print()
