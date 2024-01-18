import numpy as np
import astropy.units as u

def planck(x,T=5000,x_type="frequency"):
    """
    Planck intensity function.
    
    Parameters:
    - x (vector of length N or a single scalar value): a vector of either frequencies (Hz, s^-1), wavlengths (cm), or wavenumbers (cm^-1)
    - T (real scalar): temperature value in Kelvin (defualt is 5000 K)
    - x_type (string): "frequency", "wavelength", or "wave number" (default is "frequency")

    Returns:
    - B (same type as x): vector of calculated intensity in erg/cm^2/Hz/s per specified frequency, wavelength, or wave number in x. 
    """
    
    if x_type not in ['frequency','wavelength','wave number']:
        raise ValueError(f'{x_type} is not a valid entry. Valid entries for x_type include "frequency" (default), "wavelength", or "wave number"')
    
    assert T > 0, "Temperature must be T > 0 Kelvin"

    from astropy.constants import h, c, k_B

    # setting units
    h = h.cgs # planck constant [erg s]
    k_B = k_B.cgs # boltzmann constant [erg K^-1]
    c = c.cgs # speed of light in vacuum [cm s^-1]
    
    try:
        u.quantity_input(T)
    except:
        T = T * u.K # temperature in Kelvin

    if x_type == 'frequency': 
        nu = x.copy().to(u.Hz)

    if x_type == 'wavelength':
        nu = c/(x.copy().to(u.cm))
        
    if x_type == 'wave number':
        nu = c*(x.copy().to(1/u.cm))

        
    B = (2*h*nu**3)/(c**2) # eqn. 6.9 from Gray et. al 2022 (4th ed.)
    B = B/(np.exp((h*nu)/(k_B*T)) - 1)

    # if x_type == 'wavelength': # converted to cgs wavelength units (lam = c/nu) from eqn. 6.9 from Gray et. al 2022 (4th ed.)
    #     lam = x.copy() * u.cm
    #     B = (2*h*c)/(lam**3)
    #     B = B/(np.exp((h*c)/(lam*k_B*T)) - 1)

    # if x_type == 'wave number': # converted to cgs wave number units (nu_tilde = 1/lam = nu/c) from eqn. 6.9 from Gray et. al 2022 (4th ed.)
    #     nu_tilde = x.copy() #/ (1*units.cm)
    #     # nu_tilde = nu_tilde.to(1/u.cm)
    #     B = 2*h*c*(nu_tilde**3)
    #     B = B/(np.exp((h*c*nu_tilde)/(k_B*T)) - 1)

    B = B / (u.sr*u.Hz*u.s)

    return B

#######################

def integrate_box(x, y):
    """
    Numerical integration using the midpoint method.

    Parameters:
    - x: Array of x values.
    - y: Array of y values.

    Returns:
    - The approximate integral.
    """

    assert isinstance(x, np.ndarray), "The x variable is not a numpy array."
    assert isinstance(y, np.ndarray), "The y variable is not a numpy array."
    assert len(x) == len(y), "x and y must be of the same size."

    dx = np.diff(x) # does NOT assume equal spacing in x
    dy = (y[:-1] + y[1:]) / 2.0
    int = np.sum(dx*dy)

    return int

#######################

def integrate_bounds(func,a,b,n=50,*args,**kwargs):
    """
    Numerical integration of a given function using the midpoint method given a set of bounds.

    Parameters:
    - func: Function to be integrated.
    - a, b: lower and upper limit of integration.
    - n: (optional) number of steps to integrate over (default 50). 

    Returns:
    - The approximate integral.
    """

    assert isinstance(a, (int,float,u.Quantity)), "Lower integration limit must be a number."
    assert isinstance(b, (int,float,u.Quantity)), "Upper integration limit must be a number."
    assert isinstance(n, int), "Number of steps must be an integer."
    assert n > 0, "Number of steps must greater than 0."
    
    x = np.linspace(a,b,n)
    y = func(x,*args,**kwargs)

    return integrate_box(x,y)