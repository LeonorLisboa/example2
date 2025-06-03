import numpy as np
import numpy.testing as npt

from mycosmo.cosmology import critical_density, hubble


class TestCosmology:
    fid_cosmo = {
        "H0": 70,
        "omega_m_0": 0.3,
        "omega_k_0": 0.0,
        "omega_lambda_0": 0.7,
    }
    H_tolerance = 0.01
    rho_tolerance = 1e-28  # density in SI units ~10^-26, so small tolerance

    z_range = np.array([0.0, 0.5, 1.0])
    H_expect = np.array([70, 91.60, 123.24])

    def test_hubble(self):
        H_vals = hubble(self.z_range, self.fid_cosmo)

        npt.assert_allclose(
            H_vals,
            self.H_expect,
            atol=self.H_tolerance,
            err_msg=(
                "The H(z) differs from expected values by more than "
                f"{self.H_tolerance} decimal places."
            ),
        )

    def test_critical_density(self):
        from mycosmo.constants import G, Mpc

        H_vals = hubble(self.z_range, self.fid_cosmo) * 1e3 / Mpc  # convert H to s⁻¹
        rho_expect = (3 * H_vals**2) / (8 * np.pi * G)

        rho_vals = critical_density(self.z_range, self.fid_cosmo)

        npt.assert_allclose(
            rho_vals,
            rho_expect,
            atol=self.rho_tolerance,
            err_msg=(
                f"Critical density calculation deviates from expected values "
                f"by more than {self.rho_tolerance} kg/m³."
            ),
        )
