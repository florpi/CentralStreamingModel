import numpy as np
import sympy as sym
from scipy.interpolate import interp1d


def m_1(r, a,c,d):

	return a * r**(0.5) + c*r + d


def c_2(r, a, b,c):

	return a * r**b  + c

def xi(r, a, b):
	return (r/a)**(-b)

def first_mu_prime(s, mu):
	
	return (1 - mu**2)/s

def first_m1_dot(s, popt_m_r):

	a, b, c = popt_m_r

	return 0.5 * a/np.sqrt(s) + b

def first_m1_prime(s, mu, popt_m_r):
    
	return first_mu_prime(s,mu) * m_1(s, *popt_m_r) + \
            mu**2 * first_m1_dot(s, popt_m_r)

def second_mu_prime(s, mu):

	return -3 * (1 - mu**2) * mu/s**2
    
def second_s_prime(s,mu):
	
	return (1 - mu**2)/s

def second_m1_dot(s, popt_m_r):

    a, b, c = popt_m_r

    return -1./4.* a * s**(-3./2.)


def second_m1_prime(s, mu, popt_m_r):

	return second_mu_prime(s, mu) * m_1(s, *popt_m_r)  + \
            2 * first_mu_prime(s, mu) * mu * first_m1_dot(s, popt_m_r) + \
            mu * (second_s_prime(s, mu) * first_m1_dot(s, popt_m_r) + \
                  mu**2 * second_m1_dot(s, popt_m_r))


def m2(s, mu, popt_m_r, popt_c_r, popt_c_t):
    
	return mu**2 * c_2(s, *popt_c_r) + (1-mu**2) * c_2(s, *popt_c_t) +\
            mu **2 *m_1(s, *popt_m_r)**2
      
def first_dot_c(s, popt_c):

    a, b, c = popt_c    

    return a * b * s **(b-1)

    
def second_dot_c(s, popt_c):

    a, b, c = popt_c    

    return a * b * (b-1) * s **(b-2)

def first_c2_prime(s, mu, popt_c_r, popt_c_t):
    
    return 2 * mu * first_mu_prime(s, mu)

def first_c2_prime(s, mu, popt_c_r, popt_c_t):
	    
	return (c_2(s, *popt_c_r) - c_2(s, *popt_c_t)) * 2 * mu * first_mu_prime(s, mu) + \
		mu**3 * first_dot_c(s, popt_c_r) + mu * (1 - mu**2) * first_dot_c(s, popt_c_t)


def second_c2_prime(s, mu, popt_c_r, popt_c_t):
    
	return (c_2(s, *popt_c_r) - c_2(s, *popt_c_t))* (2 * first_mu_prime(s, mu)**2 \
                                                                 + 2 * mu *second_mu_prime(s,mu)) + \
            (first_dot_c(s, popt_c_r) - first_dot_c(s, popt_c_t)) * 4 *mu**2 * first_mu_prime(s, mu) + \
            mu**2 * (second_s_prime(s,mu) *first_dot_c(s, popt_c_r) + mu**2 * second_dot_c(s, popt_c_r) ) +\
            (1-mu**2) * (second_s_prime(s,mu) *first_dot_c(s, popt_c_t) + mu**2 * second_dot_c(s, popt_c_t) )
            
def first_m2_prime(s, mu, popt_m_r, popt_c_r, popt_c_t):
	    
	return first_c2_prime(s, mu, popt_c_r, popt_c_t) + \
	2 * mu * m_1(s, *popt_m_r) *  first_m1_prime(s, mu, popt_m_r)

def second_m2_prime(s, mu, popt_m_r, popt_c_r, popt_c_t):
    
    return second_c2_prime(s, mu, popt_c_r, popt_c_t) + \
            2 * first_m1_prime(s, mu, popt_m_r)**2 + \
            2 * mu * m_1(s, *popt_m_r) * second_m1_prime(s, mu, popt_m_r)

def first_xi_dot(s, popt_xi):

    a, b = popt_xi
    
    return - b * (s/a)**(-b) / s
    
def second_xi_dot(s, popt_xi):

    a, b = popt_xi
    
    return b * (b + 1) * (s/a)**(-b)/s**2

def first_xi_prime(s, mu, popt_xi):
    return mu * first_xi_dot(s, popt_xi)

def second_xi_prime(s, mu, popt_xi):
    return second_s_prime(s,mu) * first_xi_dot(s, popt_xi)  + \
            mu**2 * second_xi_dot(s, popt_xi)

def legendre(mu, multipole_order):
    
    if multipole_order == 0:
        return 1
    
    elif multipole_order == 2:
        return 1./2. * (3 * mu**2 - 1)
    
    elif multipole_order == 4:
        return 1./8. * (35 *mu**4 - 30 *  mu**2 + 3)
    
    else:    
        raise ValueError('Multipole order not implemented.')    
    
def s_mu_approximation(mu, s, order, popt_xi, popt_m_r, popt_c_r, popt_c_t, tpcf_dict, derivatives ):
    
    tpcf = interp1d(tpcf_dict['r'], tpcf_dict['tpcf'],
                       kind = 'linear', bounds_error = False,
                        fill_value = (tpcf_dict['tpcf'][0], 0.))

    if not derivatives:
        
        approx = tpcf(s) -first_m1_prime(s, mu, popt_m_r)

        if order == 2:
            approx +=  0.5 * second_m2_prime(s, mu, popt_m_r,
                                             popt_c_r, popt_c_t)
            
    else:
        
        approx = (1. + tpcf(s)) * (1. - first_m1_prime(s, mu, popt_m_r)) \
                - first_xi_prime(s, mu, popt_xi) * mu * m_1(s, *popt_m_r)\
                - 1.

        if order == 2:
            
            approx = (1. + tpcf(s)) * ( 1. - first_m1_prime(s, mu, popt_m_r) + \
                0.5 * second_m2_prime(s, mu, popt_m_r,
                                    popt_c_r, popt_c_t) ) \
                + first_xi_prime(s, mu, popt_xi) * (- mu * m_1(s, *popt_m_r) \
                +  first_m2_prime(s, mu, popt_m_r, popt_c_r, popt_c_t)) \
                + 0.5 * second_xi_prime(s, mu, popt_xi) * m2(s, mu, popt_m_r, popt_c_r, popt_c_t)\
                - 1.
        
    return approx


def multipoles_approximation(mu, s, order, multipole_order, popt_xi, popt_m_r, popt_c_r, popt_c_t,
					tpcf_dict, derivatives):
    
     return (2 * multipole_order + 1)/2. * \
                s_mu_approximation(mu, s, order, 
						popt_xi, popt_m_r, popt_c_r, popt_c_t,
						tpcf_dict, derivatives) * legendre(mu, multipole_order)


def approximation(s, order, popt_xi, popt_m_r, popt_c_r, popt_c_t, tpcf_dict, derivatives = False):
    
    x = sym.Symbol('x')

    s_c = 0.5* (s[1:] + s[:-1])
    
    mono_approx = np.zeros((len(s_c)))
    quad_approx = np.zeros((len(s_c)))
    hexa_approx = np.zeros((len(s_c)))

    for i, s_value in enumerate(s_c):
        mono_approx[i] = sym.integrate(multipoles_approximation(x, s_value, order, 0, 
										popt_xi, popt_m_r, popt_c_r, popt_c_t, tpcf_dict, derivatives),
                                       (x, -1, 1))
        quad_approx[i] = sym.integrate(multipoles_approximation(x, s_value, order, 2,
										popt_xi, popt_m_r, popt_c_r, popt_c_t, tpcf_dict, derivatives),
                                       (x, -1, 1))
        hexa_approx[i] = sym.integrate(multipoles_approximation(x, s_value, order, 4, 
										popt_xi, popt_m_r, popt_c_r, popt_c_t, tpcf_dict, derivatives),
                                       (x, -1, 1))

    return mono_approx, quad_approx, hexa_approx

