import numpy as np
from astropy import units

def planck(x,T,x_type="frequency"):
    """
    Planck intensity function.
    
    Arguments:
    - x (vector of length N or a single scalar value): a vector of either frequencies (Hz, s^-1), wavlengths (cm), or wavenumbers (cm^-1)
    - T (real scalar): temperature value in Kelvin
    - x_type (string): "frequency", "wavelength", or "wave number" (default is "frequency")

    Outputs:
    - B (same type as x): vector of calculated intensity in cgs per specified frequency, wavelength, or wave number in x. 
    """
    
    if x_type not in ['frequency','wavelength','wave number']:
        raise ValueError(f'{x_type} is not a valid entry. Valid entries for x_type include "frequency" (default), "wavelength", or "wave number"')
    
    assert T > 0, "Temperature must be T > 0 Kelvin"

    from astropy.constants import h, c, k_B

    # setting units
    h = h.cgs # planck constant [erg s]
    k_B = k_B.cgs # boltzmann constant [erg K^-1]
    c = c.cgs # speed of light in vacuum [cm s^-1]
    T = T * units.K # temperature in Kelvin

    if x_type == 'frequency': # eqn. 6.9 from Gray et. al 2022 (4th ed.)
        nu = x.copy() * units.Hz
        B = (2*h*nu**3)/(c**2)
        B = B/(np.exp((h*nu)/(k_B*T)) - 1)

    if x_type == 'wavelength': # converted to cgs wavelength units (lam = c/nu) from eqn. 6.9 from Gray et. al 2022 (4th ed.)
        lam = x.copy() * units.cm
        B = (2*h*c**2)/(lam**5)
        B = B/(np.exp((h*c)/(lam*k_B*T)) - 1)

    if x_type == 'wave number': # converted to cgs wave number units (nu_tilde = 1/lam = nu/c) from eqn. 6.9 from Gray et. al 2022 (4th ed.)
        nu_tilde = x.copy() / (1*units.cm)
        B = 2*h*(c**2)*(nu_tilde**3)
        B = B/(np.exp((h*c*nu_tilde)/(k_B*T)) - 1)

    return B