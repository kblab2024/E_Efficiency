#!/usr/bin/env python3
from gpu_config import xp
import numpy as np
import matplotlib.pyplot as plt
def r_te(qi, qj):
    """TE Fresnel reflection (using q = sqrt(kp**2 - eps*k0**2)/k0)."""
    return (qi - qj) / (qi + qj)

def r_tm(qi, qj, ei, ej):
    """TM Fresnel reflection (using q = sqrt(kp**2 - eps*k0**2)/k0)."""
    return (qi/ei - qj/ej) / (qi/ei + qj/ej)

def RF_multilayer(n_list, d_list, polarization, k0, kp, target_layer):
    """
    RF: reflection looking downward starting from target_layer.
    n_list: [n1, n2, ..., nN] (N layers, last is substrate)
    d_list: [d1, d2, ..., d_{N-1}] (thicknesses, last has no thickness)
    target_layer: index where you站著往下看 (0-based)

    kp can be a scalar or an array (vectorised over k_parallel).
    """
    e_list = [n**2 for n in n_list]
    kp2 = xp.asarray(kp, dtype=xp.complex128)**2
    r_eff = xp.zeros_like(kp2)

    # 從最底層往上到 target_layer
    for L in range(len(n_list)-1, target_layer, -1):
        eps_up   = e_list[L-1]
        eps_down = e_list[L]

        q_up   = xp.sqrt(kp2 - eps_up   * k0**2) / k0
        q_down = xp.sqrt(kp2 - eps_down * k0**2) / k0
        xL     = xp.exp(-q_up * k0 * d_list[L-1])

        if polarization.upper() == "TE":
            wn = q_down / q_up
        else:  # TM
            wn = (q_down / q_up) * (eps_up / eps_down)

        wm = 1.0
        wp = wm + wn
        wq = wm - wn

        al = (wp + r_eff * wq) / 2.0
        bl = (wq + r_eff * wp) / 2.0
        r_eff = xL * (bl / al) * xL

    return r_eff


def RB_multilayer(n_list, d_list, polarization, k0, kp, target_layer):
    """
    RB: reflection looking upward starting from target_layer.
    n_list: [n1, n2, ..., nN] (N layers, last is substrate)
    d_list: [d1, d2, ..., d_{N-1}] (thicknesses, last has no thickness)
    target_layer: index where you站著往上看 (0-based)

    kp can be a scalar or an array (vectorised over k_parallel).
    """
    e_list = [n**2 for n in n_list]
    kp2 = xp.asarray(kp, dtype=xp.complex128)**2
    r_eff = xp.zeros_like(kp2)

    # 從最上層往下到 target_layer
    for L in range(0, target_layer):
        eps_up   = e_list[L]
        eps_down = e_list[L+1]

        q_up   = xp.sqrt(kp2 - eps_up   * k0**2) / k0
        q_down = xp.sqrt(kp2 - eps_down * k0**2) / k0
        xL     = xp.exp(-q_up * k0 * d_list[L])

        if polarization.upper() == "TE":
            wn = q_up / q_down
        else:  # TM
            wn = (q_up / q_down) * (eps_down / eps_up)

        wm = 1.0
        wp = wm + wn
        wq = wm - wn

        bl = (wq + wp * xL * r_eff * xL)
        al = (wp + wq * xL * r_eff * xL)
        r_eff = bl / al

    return r_eff


def main():
    n_list = [1.0, 1.5, 1.0]
    d_list = [0.0, 2000e-9]  # thicknesses (最後一層無厚度)

    wl_min, wl_max, points = 400e-9, 1000e-9, 500
    wl = np.linspace(wl_min, wl_max, points)
    k0 = 2*np.pi / wl
    theta = 30.0  # normal incidence
    kp = n_list[0] * k0 * np.sin(theta)

    R_RF = np.empty(points, dtype=float)
    R_RB = np.empty(points, dtype=float)
    RF = np.empty(points, dtype=float)
    RB = np.empty(points, dtype=float)

    for i in range(points):
        # RF: 從頂層往下
        R_RF[i] = np.abs(RF_multilayer(n_list, d_list, "TE", k0[i], kp[i], target_layer=0))**2
        R_RB[i] = np.abs(RB_multilayer(n_list, d_list, "TM", k0[i], kp[i], target_layer=2))**2
        e1, e2, e3 = n_list[0]**2, n_list[1]**2, n_list[2]**2
        q1 = -np.sqrt(kp[i]**2 - e1*k0[i]**2 + 0j) / k0[i]
        q2 = -np.sqrt(kp[i]**2 - e2*k0[i]**2 + 0j) / k0[i]
        q3 = -np.sqrt(kp[i]**2 - e3*k0[i]**2 + 0j) / k0[i]
        RB[i] = np.abs(r_te(q2, q1))**2                       # n2|n1 (at z=0)
        RF[i] = np.abs(r_te(q2, q3) * np.exp(-2*q2*k0[i]*d_list[1]))**2  # n2|n3 folded back to z=0

    plt.figure(figsize=(9,5))
    plt.plot(wl*1e9, R_RF, label="RF from top")
    plt.plot(wl*1e9, R_RB, label="RB from bottom")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Reflectance")
    plt.title("multi-layer reflectance")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
