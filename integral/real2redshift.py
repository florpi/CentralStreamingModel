from scipy.interpolate import interp1d, interp2d
import numpy as np
from joblib import Parallel, delayed
import itertools
from scipy.integrate import simps, quadrature, quad

# https://arxiv.org/abs/1710.09379 (Eq 22)



def integrate(r_parallel, integrand):

		# due to discontiniuty at 0, need to integrate by parts

		left = r_parallel < 0.
		right = r_parallel > 0.

		int_left = simps(integrand[...,left], r_parallel[...,left], axis = -1)

		int_right = simps(integrand[...,right], r_parallel[...,right], axis = -1)

		pi_sigma = int_left + int_right - 1.

		return pi_sigma

def integrate_full(r_parallel, integrand):

		int_ = simps(integrand, r_parallel, axis=-1)

		pi_sigma = int_ - 1.

		return pi_sigma

def compute_integrand_s_mu(s, mu, twopcf_dict, projected_pairwise_pdf, 
		s_perp_full, s_paral_full, vlos_bins, n_cores = 1):

		s_c = 0.5 * ( s[1:] + s[:-1] )
		mu_c = 0.5 * ( mu[1:] + mu[:-1] )

		binwidth = s_paral_full[1] - s_paral_full[0]
		initial = -69. - binwidth/2.
		final = 69. + binwidth/2.
		y = np.arange(initial, final, binwidth)

		# Refine integrand near zero (where correlation function dominates)
		epsilon = 0.0001
		y = np.append(y, epsilon)
		y = np.append(y, -epsilon)
		y = np.sort(y)

		# Interpolate tpcf
		interp_twopcf = interp1d(twopcf_dict['r'], twopcf_dict['tpcf'], kind = 'linear', 
							fill_value = (-1., twopcf_dict['tpcf'][-1]), bounds_error=False)

		def define_integrand(args):

				s_bin, mu_bin = args

				integrand = np.zeros(len(y))
				pdf_contribution = np.zeros(len(y))

				for k, y_value in enumerate(y):

							
						s_perp_c = s_c[s_bin] * np.sqrt(1 - mu_c[mu_bin]**2)
						s_paral_c = s_c[s_bin] * mu_c[mu_bin]

						r = np.sqrt(s_perp_c**2 + y_value**2)

						y_bin = np.digitize(np.abs(y_value), s_paral_full - binwidth/2) - 1 

						selected_pairwise_pdf = projected_pairwise_pdf[:, y_bin, :]
						interp_pairwise_pdf = interp2d(vlos_bins, s_perp_full,   
												selected_pairwise_pdf, kind='linear')

						vlos = (s_paral_c - y_value) * np.sign(y_value)


						if ( (vlos > np.min(vlos_bins)) and (vlos < np.max(vlos_bins) ) ):
								p = interp_pairwise_pdf(vlos, s_perp_c)
						else:
								p = 0.

						if (r < twopcf_dict['r'][-1]):
								integrand[k] = (1. + interp_twopcf(r)) * p

						else:
								integrand[k] = p
						pdf_contribution[k] = p

				return [pdf_contribution, integrand]


		# Parallelise the integrand computation

		dim1 = s_c.shape[0]
		dim2 = mu_c.shape[0]

		arg_instances = list((i,j) for i,j in itertools.product(range(dim1), range(dim2)))


		combined_result = Parallel(n_jobs = n_cores, verbose = 0 ) (map(delayed(define_integrand), arg_instances))

		pdf_contribution = [i[0] for i in combined_result]
		result_integrand = [i[1] for i in combined_result]

		pdf_contribution = np.asarray(pdf_contribution)
		pdf_contribution= pdf_contribution.reshape((s_c.shape[0], mu_c.shape[0], y.shape[0]))

		result_integrand = np.asarray(result_integrand)
		result_integrand = result_integrand.reshape((s_c.shape[0], mu_c.shape[0], y.shape[0]))
		
		return y, result_integrand, pdf_contribution


