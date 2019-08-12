import numpy as np
import sympy as sym
from scipy.interpolate import interp1d


# ************ FITTING THE FIRST MOMENT ********************
def m_1(r, m_r_spl):

	return	m_r_spl(r)

def first_m1_dot(r, m_r_spl):

	return m_r_spl.derivative()(r)


def second_m1_dot(r, m_r_spl):

	return m_r_spl.derivative(n = 2)(r)




# ************ FITTING THE SECOND MOMENT ********************
def c_2(r, c_spl):

	return c_spl(r)

def first_dot_c(r, c_spl):

	return c_spl.derivative()(r)

	
def second_dot_c(r, c_spl):

	return c_spl.derivative( n = 2 )(r)


#****** FITTING THE REAL SPACE CORRELATION FUNCTION *********
def xi(r, a, b):
	return (r/a)**(-b)

def first_xi_dot(s, popt_xi):

	a, b = popt_xi
	
	return - b * (s/a)**(-b) / s
	
def second_xi_dot(s, popt_xi):

	a, b = popt_xi
	
	return b * (b + 1) * (s/a)**(-b)/s**2

'''

def xi(r, xi_spl):

	return xi_spl(r)

def first_xi_dot(r, xi_spl):

	return xi_spl.derivative()(r)
	
def second_xi_dot(r, xi_spl):

	return xi_spl.derivative( n = 2 )(r)
'''

# *********** PRIME DERIVATIVES ************************

def first_mu_prime(s, mu):
	
	return (1 - mu**2)/s

def first_m1_prime(s, mu, m_r_spl):
	
	return first_mu_prime(s,mu) * m_1(s, m_r_spl) + \
			mu**2 * first_m1_dot(s, m_r_spl)

def second_mu_prime(s, mu):

	return -3 * (1 - mu**2) * mu/s**2
	
def second_s_prime(s,mu):
	
	return (1 - mu**2)/s


def second_m1_prime(s, mu, m_r_spl):

	return second_mu_prime(s, mu) * m_1(s, m_r_spl)  + \
			2 * first_mu_prime(s, mu) * mu * first_m1_dot(s, m_r_spl) + \
			mu * (second_s_prime(s, mu) * first_m1_dot(s, m_r_spl) + \
				  mu**2 * second_m1_dot(s, m_r_spl))


def m2(s, mu, m_r_spl, c_r_spl, c_t_spl):
	
	return mu**2 * c_2(s, c_r_spl) + (1-mu**2) * c_2(s, c_t_spl) +\
			mu **2 *m_1(s, m_r_spl)**2
	  
def first_c2_prime(s, mu, c_r_spl, c_t_spl):
	
	return 2 * mu * first_mu_prime(s, mu)

def first_c2_prime(s, mu, c_r_spl, c_t_spl):
		
	return (c_2(s, c_r_spl) - c_2(s, c_t_spl)) * 2 * mu * first_mu_prime(s, mu) + \
		mu**3 * first_dot_c(s, c_r_spl) + mu * (1 - mu**2) * first_dot_c(s, c_t_spl)


def second_c2_prime(s, mu, c_r_spl, c_t_spl):
	
	return (c_2(s, c_r_spl) - c_2(s, c_t_spl))* (2 * first_mu_prime(s, mu)**2 \
																 + 2 * mu *second_mu_prime(s,mu)) + \
			(first_dot_c(s, c_r_spl) - first_dot_c(s, c_t_spl)) * 4 *mu**2 * first_mu_prime(s, mu) + \
			mu**2 * (second_s_prime(s,mu) *first_dot_c(s, c_r_spl) + mu**2 * second_dot_c(s, c_r_spl) ) +\
			(1-mu**2) * (second_s_prime(s,mu) *first_dot_c(s, c_t_spl) + mu**2 * second_dot_c(s, c_t_spl) )
			
def first_m2_prime(s, mu, m_r_spl, c_r_spl, c_t_spl):
		
	return first_c2_prime(s, mu, c_r_spl, c_t_spl) + \
	2 * mu * m_1(s, m_r_spl) *	first_m1_prime(s, mu, m_r_spl)

def second_m2_prime(s, mu, m_r_spl, c_r_spl, c_t_spl):
	
	return second_c2_prime(s, mu, c_r_spl, c_t_spl) + \
			2 * first_m1_prime(s, mu, m_r_spl)**2 + \
			2 * mu * m_1(s, m_r_spl) * second_m1_prime(s, mu, m_r_spl)

'''
def first_xi_prime(s, mu, xi_spl):
	return mu * first_xi_dot(s, xi_spl)

def second_xi_prime(s, mu, xi_spl):
	return second_s_prime(s,mu) * first_xi_dot(s, xi_spl)  + \
			mu**2 * second_xi_dot(s, xi_spl)
'''

def first_xi_prime(s, mu, popt_xi):
	return mu * first_xi_dot(s, popt_xi)

def second_xi_prime(s, mu, popt_xi):
	return second_s_prime(s,mu) * first_xi_dot(s, popt_xi)	+ \
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
	
'''
def s_mu_approximation(mu, s, order, xi_spl, m_r_spl, c_r_spl, c_t_spl, tpcf_dict, derivatives ):
	
	tpcf = interp1d(tpcf_dict['r'], tpcf_dict['tpcf'],
					   kind = 'linear', bounds_error = False,
						fill_value = (tpcf_dict['tpcf'][0], 0.))

	if not derivatives:
		
		approx = tpcf(s) -first_m1_prime(s, mu, m_r_spl)

		if order == 2:
			approx +=  0.5 * second_m2_prime(s, mu, m_r_spl,
											 c_r_spl, c_t_spl)
			
	else:
		
		approx = (1. + tpcf(s)) * (1. - first_m1_prime(s, mu, m_r_spl)) \
				- first_xi_prime(s, mu, xi_spl) * mu * m_1(s, m_r_spl)\
				- 1.

		if order == 2:
			
			approx = (1. + tpcf(s)) * ( 1. - first_m1_prime(s, mu, m_r_spl) + \
				0.5 * second_m2_prime(s, mu, m_r_spl,
									c_r_spl, c_t_spl) ) \
				+ first_xi_prime(s, mu, xi_spl) * ( - mu * m_1(s, m_r_spl) \
				+  first_m2_prime(s, mu, m_r_spl, c_r_spl, c_t_spl) ) \
				+ 0.5 * second_xi_prime(s, mu, xi_spl) * m2(s, mu, m_r_spl, c_r_spl, c_t_spl)\
				- 1.
		
	return approx


def multipoles_approximation(mu, s, order, multipole_order, xi_spl, m_r_spl, c_r_spl, c_t_spl,
					tpcf_dict, derivatives):
	
	 return (2 * multipole_order + 1)/2. * \
				s_mu_approximation(mu, s, order, 
						xi_spl, m_r_spl, c_r_spl, c_t_spl,
						tpcf_dict, derivatives) * legendre(mu, multipole_order)


def approximation(s, order, xi_spl, m_r_spl, c_r_spl, c_t_spl, tpcf_dict, derivatives = False):
	
	x = sym.Symbol('x')

	s_c = 0.5* (s[1:] + s[:-1])
	
	mono_approx = np.zeros((len(s_c)))
	quad_approx = np.zeros((len(s_c)))
	hexa_approx = np.zeros((len(s_c)))

	for i, s_value in enumerate(s_c):
		mono_approx[i] = sym.integrate(multipoles_approximation(x, s_value, order, 0, 
										xi_spl, m_r_spl, c_r_spl, c_t_spl, tpcf_dict, derivatives),
									   (x, -1, 1))
		quad_approx[i] = sym.integrate(multipoles_approximation(x, s_value, order, 2,
										xi_spl, m_r_spl, c_r_spl, c_t_spl, tpcf_dict, derivatives),
									   (x, -1, 1))
		hexa_approx[i] = sym.integrate(multipoles_approximation(x, s_value, order, 4, 
										xi_spl, m_r_spl, c_r_spl, c_t_spl, tpcf_dict, derivatives),
									   (x, -1, 1))

	return mono_approx, quad_approx, hexa_approx
'''
def s_mu_approximation(mu, s, order, popt_xi, m_r_spl, c_r_spl, c_t_spl, tpcf_dict, derivatives ):
	
	tpcf = interp1d(tpcf_dict['r'], tpcf_dict['tpcf'],
					   kind = 'linear', bounds_error = False,
						fill_value = (tpcf_dict['tpcf'][0], 0.))

	if not derivatives:
		
		approx = tpcf(s) -first_m1_prime(s, mu, m_r_spl)

		if order == 2:
			approx +=  0.5 * second_m2_prime(s, mu, m_r_spl,
											 c_r_spl, c_t_spl)
			
	else:
		
		approx = (1. + xi(s, *popt_xi)) * (1. - first_m1_prime(s, mu, m_r_spl)) \
				- first_xi_prime(s, mu, popt_xi) * mu * m_1(s, m_r_spl)\
				- 1.

		if order == 2:

			
			'''
			approx = xi(s, *popt_xi) * (1 - first_m1_prime(s, mu, m_r_spl) ) - first_m1_prime(s, mu, m_r_spl) \
					+ 0.5 * second_m2_prime(s, mu, m_r_spl, c_r_spl, c_t_spl)\
				+ first_xi_prime(s, mu, popt_xi) * ( - mu * m_1(s, m_r_spl)) \
			 	+ 0.5 * second_xi_prime(s, mu, popt_xi) * m2(s, mu, m_r_spl, c_r_spl, c_t_spl)
				#+	first_m2_prime(s, mu, m_r_spl, c_r_spl, c_t_spl) ) 
	
			
			'''
			approx = (1. + xi(s, *popt_xi)) * ( 1. - first_m1_prime(s, mu, m_r_spl) + \
				0.5 * second_m2_prime(s, mu, m_r_spl,
									c_r_spl, c_t_spl) ) \
				- 1. \
				+ first_xi_prime(s, mu, popt_xi) * ( - mu * m_1(s, m_r_spl) \
				+	first_m2_prime(s, mu, m_r_spl, c_r_spl, c_t_spl) ) \
			 	+ 0.5 * second_xi_prime(s, mu, popt_xi) * m2(s, mu, m_r_spl, c_r_spl, c_t_spl)\
		
	return approx
			

def multipoles_approximation(mu, s, order, multipole_order, popt_xi, m_r_spl, c_r_spl, c_t_spl,
					tpcf_dict, derivatives):
	
	 return (2 * multipole_order + 1)/2. * \
				s_mu_approximation(mu, s, order, 
						popt_xi, m_r_spl, c_r_spl, c_t_spl,
						tpcf_dict, derivatives) * legendre(mu, multipole_order)


def approximation(s, order, popt_xi, m_r_spl, c_r_spl, c_t_spl, tpcf_dict, derivatives = False):
	
	x = sym.Symbol('x')

	s_c = 0.5* (s[1:] + s[:-1])
	
	mono_approx = np.zeros((len(s_c)))
	quad_approx = np.zeros((len(s_c)))
	hexa_approx = np.zeros((len(s_c)))

	for i, s_value in enumerate(s_c):
		mono_approx[i] = sym.integrate(multipoles_approximation(x, s_value, order, 0, 
										popt_xi, m_r_spl, c_r_spl, c_t_spl, tpcf_dict, derivatives),
									   (x, -1, 1))
		quad_approx[i] = sym.integrate(multipoles_approximation(x, s_value, order, 2,
										popt_xi, m_r_spl, c_r_spl, c_t_spl, tpcf_dict, derivatives),
									   (x, -1, 1))
		hexa_approx[i] = sym.integrate(multipoles_approximation(x, s_value, order, 4, 
										popt_xi, m_r_spl, c_r_spl, c_t_spl, tpcf_dict, derivatives),
									   (x, -1, 1))

	return mono_approx, quad_approx, hexa_approx

