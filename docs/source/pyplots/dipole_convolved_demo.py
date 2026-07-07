import matplotlib.pylab as plt
import numpy as np

import litebird_sim as lbs


n_samples = 180
phi = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)

pointings = np.zeros((1, n_samples, 3))
pointings[0, :, 0] = np.pi / 2.0  # Equatorial scan
pointings[0, :, 1] = phi
pointings[0, :, 2] = 0.0

beta = 1.0e-3
velocity = np.tile([lbs.C_LIGHT_KM_OVER_S * beta, 0.0, 0.0], (n_samples, 1))
frequency_ghz = np.array([100.0])

beam_alm = lbs.gauss_beam_to_alm(
    lmax=64,
    mmax=64,
    fwhm_rad=np.deg2rad(60.0),
    psi_pol_rad=None,
)
s_params = lbs.BeamSParams.from_beam_alm(beam_alm)

tod_pencil = np.zeros((1, n_samples))
lbs.add_dipole(
    tod_pencil,
    pointings,
    velocity,
    t_cmb_k=lbs.T_CMB_K,
    frequency_ghz=frequency_ghz,
    dipole_type=lbs.DipoleType.QUADRATIC_FROM_LIN_T,
)

tod_convolved = np.zeros((1, n_samples))
lbs.add_dipole(
    tod_convolved,
    pointings,
    velocity,
    t_cmb_k=lbs.T_CMB_K,
    frequency_ghz=frequency_ghz,
    dipole_type=lbs.DipoleType.QUADRATIC_FROM_LIN_T,
    s_params=s_params,
)

scan_angle_deg = np.rad2deg(phi)
diff_uk = (tod_convolved[0] - tod_pencil[0]) * 1.0e6

_, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)

axes[0].plot(scan_angle_deg, tod_pencil[0] * 1.0e3, label="Pencil beam")
axes[0].plot(
    scan_angle_deg,
    tod_convolved[0] * 1.0e3,
    label=f"60 deg Gaussian beam, S_z={s_params.s_vec[2]:.3f}",
)
axes[0].set_ylabel("Signal [mK]")
axes[0].legend()

axes[1].plot(scan_angle_deg, diff_uk, color="tab:red")
axes[1].set_xlabel("Scan longitude [deg]")
axes[1].set_ylabel("Convolved - pencil [uK]")

plt.tight_layout()
