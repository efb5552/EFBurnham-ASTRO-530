import numpy as np
import astropy.units as u
from matplotlib import pyplot as plt

def planck(x,T,x_type="frequency"):
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
    
    # assert T > 0, "Temperature must be T > 0 Kelvin"

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

def integrate_bounds(func,a,b,dx,*args,**kwargs):
    """
    Numerical integration of a given function using the midpoint method given a set of bounds and calculates the integral with logarithmically spaced values.

    Parameters:
    - func: Function to be integrated of form func(x,**kwargs).
    - a, b: lower and upper limit of integration.
    - dx: size of bins/slices to be integrated over. 

    Returns:
    - The approximate integral.
    """
    type='linear'

    assert isinstance(a, (int,float,u.Quantity)), "Lower integration limit must be a number."
    assert isinstance(b, (int,float,u.Quantity)), "Upper integration limit must be a number."
    assert type in ['linear','log']
    # assert isinstance(dx, int), "Number of steps must be an integer."
    # assert dx > 0, "Size of steps must greater than 0."
    
    if type == 'linear':
        n = int(round(abs(b-a)/dx,0))
        x = np.linspace(a,b,n)
    if type == 'log':
        n = int(round((np.log10(b)-np.log10(a))/dx,0))
        x = np.logspace(np.log10(a),np.log10(b),n)
    y = func(x,*args,**kwargs)

    return integrate_box(x,y)

#######################

def precision(truth,calc):
    """
    Determined the precision of a calculation when provided the true value.
    
    Parameters:
    - truth (scalar): true value
    - calc (scalar or array): calculated value(s)

    Returns:
    - Absolute value of precision.
    """
    return abs(1-(calc/truth))

#######################

def plot_precisions_bdx(func,truth,file_name,*args,**kwargs):
    """
    Calculates and plots the precision of a provided numerical integrator function with respect to some true value, where only the upper limit and the slice size are being integrated over.
    
    Parameters:
    - func (function): function with the arguments func(a,b,dx,*args,**kwargs)
    - truth: true value in which precision is based off of
    - file_name (str): directory in Plots/ and name of .svg file that the image is being saved to. If this field is not specified, the file will not be saved. (Ex: file_name="HW1/image" will be saved to Plots/HW1/image.svg)
    """

    b_log_range=[0,2.5]
    dx_log_range=[-1,-5]
    num_vals=50

    a = 0 # lower bound
    b_values = np.logspace(b_log_range[0],b_log_range[1],num_vals)
    dx_values = np.logspace(dx_log_range[0],dx_log_range[1],num_vals)
    values = np.array([b_values,dx_values])
    default_values = [np.max(b_values), np.min(dx_values)] # Default values for b, and dx (lowest values from test range)
    
    prec_arrs = []
    func_calcs_all = []
    for i, var_values in enumerate(values): # parsing through 3 variables (a, b, n)
        prec_arr = []
        func_calcs = []
        b, dx = default_values
        for j, val in enumerate(var_values): # parsing through each value in the variable space
    
            if i == 0: # setting default values
                b = val.copy()
            elif i == 1:
                dx = val.copy()
                
            func_calc = func(a,b,dx,*args,**kwargs) # evaluating the integral for each value
            prec_arr.append(precision(truth,func_calc))
        prec_arrs.append(prec_arr)

    fig = plt.figure(figsize=(17,8)) # plotting results
    grid = plt.GridSpec(1,2,wspace=0,hspace=0.4)
    
    for i, var in enumerate(prec_arrs):
        ax = fig.add_subplot(grid[0,i])

        ax.set_ylim(10e-18,100)
    
        ax.loglog(values[i],prec_arrs[i],lw=5,c=['cornflowerblue','indigo'][i])
        ax.scatter(values[i],prec_arrs[i],s=50,c=['cornflowerblue','indigo'][i])
        ax.set_xlabel([r"Upper Limit","Size of Bins"][i])
        a
        ax.grid()
        if i > 0:
            ax.set_yticklabels([])
        if i == 0:
            ax.set_ylabel("Precision")

    plt.savefig(f"../Plots/{file_name}.pdf")
    plt.show()

#######################

def plot_precisions_abdx(func,truth,file_name,*args,**kwargs):
    """
    Calculates and plots the precision of a provided numerical integrator function with respect to some true value, where the lower limit, the upper limit, and the slice size are being integrated over.
    
    Parameters:
    - func (function): function with the arguments func(a,b,dx,*args,**kwargs)
    - truth: true value in which precision is based off of
    - file_name (str): directory in Plots/ and name of .svg file that the image is being saved to. If this field is not specified, the file will not be saved. (Ex: file_name="HW1/image" will be saved to Plots/HW1/image.svg)
    """
    a_log_range = [0,-20]
    b_log_range=[0,2.5]
    dx_log_range=[-1,-5]
    num_vals=25

    a_values = np.logspace(a_log_range[0],a_log_range[1],num_vals)
    b_values = np.logspace(b_log_range[0],b_log_range[1],num_vals)
    dx_values = np.logspace(dx_log_range[0],dx_log_range[1],num_vals)
    values = np.array([a_values,b_values,dx_values])
    default_values = [np.min(a_values),np.max(b_values),np.min(dx_values)] # Default values for b, and dx (lowest values from test range)
    
    prec_arrs = []
    func_calcs_all = []
    for i, var_values in enumerate(values): # parsing through 3 variables (a, b, n)
        prec_arr = []
        func_calcs = []
        a, b, dx = default_values
        for j, val in enumerate(var_values): # parsing through each value in the variable space
    
            if i == 0: # using default values for variables not being actively tested
                a = val.copy()
            elif i == 1:
                b = val.copy()
            elif i == 2:
                dx = val.copy()
                
            func_calc = func(a,b,dx,*args,**kwargs) # evaluating the integral for each value
            prec_arr.append(precision(truth,func_calc))
        prec_arrs.append(prec_arr)

    fig = plt.figure(figsize=(25,7)) # plotting results
    grid = plt.GridSpec(1,3,wspace=0,hspace=0.4)
    
    for i, var in enumerate(prec_arrs):
        ax = fig.add_subplot(grid[0,i])

        ax.set_ylim(10e-18,100)
    
        ax.loglog(values[i],prec_arrs[i],lw=4,c=['lightblue','cornflowerblue','indigo'][i])
        ax.scatter(values[i],prec_arrs[i],s=50,c=['lightblue','cornflowerblue','indigo'][i])
        ax.set_xlabel([r"Lower Limit",r"Upper Limit","Size of Bins"][i])
        ax.grid()
        if i > 0:
            ax.set_yticklabels([])
        if i == 0:
            ax.set_ylabel("Precision")

    plt.savefig(f"../Plots/{file_name}.pdf")
    plt.show()

def H_nu_tau(tau,S_nu,*args,**kwargs):

    from scipy.integrate import simps 
    from scipy.special import expn
    
    b = 10e2 # limits set from HW3 for maximum precision
    a = 10e-18
    dx = 10e-5

    if tau != 0.0 and tau > a:
        num = int(round((np.log10(b)-np.log10(tau))/dx,0))
        t = np.logspace(np.log10(tau),np.log10(b),num)
        term1 = 0.5*simps(S_nu(t,*args,**kwargs)*expn(2,t-tau),x=t) # integrate from tau to infty
        
        num = int(round(((np.log10(tau)-np.log10(a))/dx),0))
        t = np.logspace(np.log10(a),np.log10(tau),num)
        t[0] = 0
        term2 = -0.5*simps(S_nu(t,*args,**kwargs)*expn(2,tau-t),x=t) # integrate from 0 to tau
        H = term1 + term2

    else:
        num = int(round(b/dx,0))
        t = np.linspace(tau,b,num)
        H = 0.5*simps(S_nu(t,*args,**kwargs)*expn(2,t),x=t)

    return H      

def partition_function(species,T):
    from scipy.optimize import curve_fit
    import pandas as pd
    import numpy as np

    partition_functions = pd.read_csv("partition_functions_T.csv",delimiter=",") # modified table 

    if species == 'H-': # only depends on ionization energy, filled valence shell
        chi = 0.755 # eV
        U = 1*10**(-(chi*(5040/T)))
    elif species not in partition_functions.columns: # table is missing fully ionized elements, which have a constant partition function
        U = 1.
    else:
        def exponential_decay(x, a, b, c):
            return a * np.exp(-b * x) + c
        
        if species in partition_functions.columns[:-1]:
            y_values = np.array(list(partition_functions[species][:-1].values))  # Extracting values from DataFrame column
            thetas = partition_functions['theta'].values
            nan_mask = ~np.isnan(y_values)  # Create mask to remove NaN values
            y = y_values[nan_mask]
            x_values = thetas[:-1][nan_mask]
            x = np.array(x_values, dtype=float)
    
            popt, pcov = curve_fit(exponential_decay, x, y) # fitting the points
            x_new = 5040/T
            logU = exponential_decay(x_new, *popt) # returns log value of U
            U = 10**logU
    return U