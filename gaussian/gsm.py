from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal
import numpy as np

# Get moments measured from simulation -> interpolate them
def interpolate_parameters(r, discrete_mean, discrete_std_r, discrete_std_t):
	
	mean  = interp1d(r, discrete_mean, kind = 'linear', bounds_error = False, 
						fill_value = (discrete_mean[0], discrete_mean[-1]))
	std_r = interp1d(r, discrete_std_r, kind='linear', bounds_error = False,
						fill_value = (discrete_std_r[0], discrete_std_r[-1]))
	std_t = interp1d(r, discrete_std_t, kind='linear', bounds_error = False,
						fill_value = (discrete_std_t[0], discrete_std_t[-1]))

	return mean, std_r, std_t



def gaussian_los_pdf( mean, std_r, std_t):

	def gaussian_pdf(r_perp, r_parallel, v_los):

		r = np.sqrt(r_parallel**2 + r_perp**2)

		mu = r_parallel/r

		sigma_sq = mu**2 * std_r(r)**2 + (1 - mu**2) * std_t(r)**2	


		return 1./np.sqrt(2. * np.pi * sigma_sq) * np.exp(-(v_los - mu *mean(r))**2/2./sigma_sq)

	return gaussian_pdf
		

def gaussian_rt_pdf(v,  mean_r, std_r, std_t):

	mean = np.asarray([0., mean_r])

	omega = np.asarray([[std_t**2, 0.], [0., std_r**2]])

	return multivariate_normal.pdf(v, mean = mean, cov = omega)
	

	
