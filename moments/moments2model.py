import numpy as np
from scipy.stats import norm
from halotools.mock_observables import tpcf_multipole
from CentralStreamingModel.integral import real2redshift as real2red
from CentralStreamingModel.skewt import skewt as st
from CentralStreamingModel.skewt import skewt_moments




class Model:

	def __init__(self, rm, projected_moments, model):

		self.rm = rm

		self.s = np.arange(0., 50., 1.)
		self.s_c = 0.5 * (self.s[1:] + self.s[:-1])
		self.mu = np.linspace(0., 1., 60)
		self.mu_c = 0.5 * (self.mu[1:] + self.mu[:-1])

		mean = projected_moments[...,0]
		sigma = np.sqrt(projected_moments[...,1])


		if model == 'measured':

			self.jointpdf_los = self.rm.jointpdf_los
			self.color = 'black'

		elif model  == 'gaussian':

			self.jointpdf_los = self.moments2gaussian(mean, sigma)
			self.color = 'forestgreen'

		elif model == 'st':
			
			gamma1 = projected_moments[..., 2]/sigma**3
			gamma2 = projected_moments[..., 3]/sigma**4 - 3.

			self.jointpdf_los = self.moments2st(mean, sigma, gamma1, gamma2)
			self.color = 'indianred'

		self.multipoles(self.s, self.mu)

	def moments2gaussian(self, mean, sigma):

		return norm.pdf(self.rm.v_los[np.newaxis, np.newaxis, :],
				loc  = mean[...,np.newaxis],
				scale = sigma[..., np.newaxis])

	def moments2st(self, mean, sigma, gamma1, gamma2):

		st_los = np.zeros_like(self.rm.jointpdf_los)

		for i, rperp in enumerate(self.rm.r_perp):
			for j, rpar in enumerate(self.rm.r_parallel):

				v_c, w, alpha, nu = skewt_moments.moments2parameters(mean[i,j], sigma[i,j], gamma1[i,j], gamma2[i,j])

				st_los[i,j,:] = st.skewt_pdf(self.rm.v_los, w, v_c, alpha, nu)

		return st_los


	def multipoles(self, s, mu):

		rparallel, self.integrand, pdf_contribution = real2red.compute_integrand_s_mu(s,
				                    mu, self.rm.tpcf_dict, self.jointpdf_los,
									self.rm.r_perp, self.rm.r_parallel, self.rm.v_los)

		
		self.s_mu = real2red.integrate(rparallel, self.integrand)

		self.mono = tpcf_multipole(self.s_mu, mu, order = 0)
		self.quad = tpcf_multipole(self.s_mu, mu, order = 2)
		self.hexa = tpcf_multipole(self.s_mu, mu, order = 4)


		self.s_c = 0.5 * (s[1:] + s[:-1])
		self.mu_c = 0.5 * (mu[1:] + mu[:-1])

		

				






