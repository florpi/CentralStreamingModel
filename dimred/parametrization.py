import numpy as np
import matplotlib.pyplot as plt 
from scipy.optimize import curve_fit
import inspect

def chi2dof(observed, expected, n_params):
    dof = len(observed) - n_params
    chi2 = np.sum((observed - expected)**2 / abs(expected))
    print(chi2)
    return chi2/dof

def fit(fitting_function, n_params, x, y,  p0):

    popt, pcov = curve_fit(fitting_function, x, y, p0) 

    chi = chi2dof(y, fitting_function(x, *popt),  n_params) 

    return popt, chi 


def plot(fitting_function, x, y,  color, label, parameter_name, limit = 5, p0=None):

    threshold = x > limit
    x_ = x[threshold]
    y_ = y[threshold]
    n_params = len(inspect.getfullargspec(fitting_function)[0])-1


    popt, chi = fit(fitting_function, n_params, x_, y_,  p0) 

    plt.plot(x, y, color=color, linestyle='', 
            marker='o', markersize=3,label=label)

    plt.plot(x, fitting_function(x, *popt), color=color)

    #plt.text(60, np.min(y) + 0.15*np.min(y), r'$\chi^2_\nu $ = %.2f'%chi, fontdict=dict(color=color))

    plt.xlabel('r[Mpc/h]')
    plt.ylabel(parameter_name)

    return fitting_function(x, *popt), chi, popt

