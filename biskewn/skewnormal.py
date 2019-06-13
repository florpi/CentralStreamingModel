import numpy as np
from scipy.special import gamma
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.optimize import fsolve

#https://arxiv.org/pdf/0911.2342.pdf equation 1

#*************************** PDF DEFINITION *****************************#

def symmetric_pdf(v, vr_c, w_r, w_t):
	
	mean = [0., vr_c]
	omega = np.asarray([[w_t, 0.], [0., w_r]])

	return multivariate_normal.pdf(v, mean = mean, cov = omega)

def symmetric_cdf(v, vr_c, w_r, alpha):

	alpha_rv = norm(loc = 0., scale = 1.)
	argument = (v[:,1] - vr_c)/np.sqrt(w_r) * alpha

	return alpha_rv.cdf(argument)


def skewnormal(v, w_r, w_t, vr_c, alpha):

	w_r = np.abs(w_r)
	w_t = np.abs(w_t)

	sym_pdf = symmetric_pdf(v, vr_c, w_r, w_t)
	sym_cdf = symmetric_cdf(v, vr_c, w_r, alpha)

	return 2 * sym_pdf * sym_cdf
	
#*************************** MOMENTS METHOD ********************************#


def get_alpha(skewness):
	func = lambda delta : skewness - (4. - np.pi)/2 *(delta * np.sqrt(2/np.pi))**3 / (1 - 2*delta**2/np.pi)**(3/2)

	delta_initial_guess = 0.7
	delta_solution = fsolve(func, delta_initial_guess)
	delta_solution = delta_solution[0]
	alpha = delta_solution / np.sqrt(1-delta_solution**2)

	return delta_solution, alpha

def get_w(sigma, delta):

    return sigma/np.sqrt(1 - 2*delta**2/np.pi)

def get_vc(mean, w, delta):
    
    return mean - w * delta *np.sqrt(2./np.pi)

def skewnormal_given_moments(v, mean_r, std_r, std_t, skewness):

	delta, alpha = get_alpha(skewness)
	
	w_r = get_w(std_r, delta )
	w_t = get_w(std_t, 0.)

	vr_c = get_vc(mean_r, w_r, delta)

	return skewnormal(v, w_r**2, w_t**2, vr_c, alpha)
