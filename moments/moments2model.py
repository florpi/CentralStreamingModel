import numpy as np
from scipy.stats import norm
from halotools.mock_observables import tpcf_multipole
from CentralStreamingModel.integral import real2redshift as real2red
from CentralStreamingModel.skewt import skewt as st
from CentralStreamingModel.skewt import skewt_moments
from scipy.optimize import curve_fit




class Model:

	def __init__(self, rm, projected_moments, model, p0 = None):

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

		elif model  == 'bf-gaussian':

			self.jointpdf_los = self.bfgaussian()
			self.color = 'darkseagreen'

		elif model == 'st':
			
			gamma1 = projected_moments[..., 2]/sigma**3
			gamma2 = projected_moments[..., 3]/sigma**4 - 3.

			self.params, self.jointpdf_los = self.moments2st(mean, sigma, gamma1, gamma2, p0)
			self.color = 'royalblue'

		elif model == 'bf-st':
			
			self.params, self.jointpdf_los = self.bfst()
			self.color = 'indianred'

	
		self.multipoles(self.s, self.mu)

	def moments2gaussian(self, mean, sigma):

		return norm.pdf(self.rm.v_los[np.newaxis, np.newaxis, :],
				loc  = mean[...,np.newaxis],
				scale = sigma[..., np.newaxis])
	
	def bfgaussian(self):

		bf_gauss = np.zeros_like(self.rm.jointpdf_los)

		for i, rperp in enumerate(self.rm.r_perp):
			for j, rpar in enumerate(self.rm.r_parallel):

				popt, pcov = curve_fit(gaussian, self.rm.v_los, self.rm.jointpdf_los[i,j,:]) 
						        
				bf_gauss[i,j,:] = norm.pdf(self.rm.v_los, loc = popt[0], scale = popt[1])

		return bf_gauss



	def moments2st(self, mean, sigma, gamma1, gamma2, p0 = None):

		st_los = np.zeros_like(self.rm.jointpdf_los)
		params = np.zeros((self.rm.jointpdf_los.shape[0], self.rm.jointpdf_los.shape[1], 4))

		for i, rperp in enumerate(self.rm.r_perp):
			for j, rpar in enumerate(self.rm.r_parallel):

				#v_c, w, alpha, nu = skewt_moments.moments2parameters(mean[i,j], sigma[i,j], gamma1[i,j], gamma2[i,j])
				if p0 is not None:
					params[i,j,...] = skewt_moments.moments2parameters(mean[i,j], sigma[i,j], gamma1[i,j], gamma2[i,j], p0 = (p0[i,j,-2], p0[i,j,-1]))
				else:
					params[i,j,...] = skewt_moments.moments2parameters(mean[i,j], sigma[i,j], gamma1[i,j], gamma2[i,j])

				st_los[i,j,:] = st.skewt_pdf(self.rm.v_los, params[i,j,1], params[i,j,0], params[i,j,2], params[i,j,-1])

		return params, st_los

	def bfst(self):

		bf_st = np.zeros_like(self.rm.jointpdf_los)
		params = np.zeros((self.rm.jointpdf_los.shape[0], self.rm.jointpdf_los.shape[1], 4))

		for i, rperp in enumerate(self.rm.r_perp):
			for j, rpar in enumerate(self.rm.r_parallel):

				popt, pcov = curve_fit(st.skewt_pdf, self.rm.v_los, self.rm.jointpdf_los[i,j,:],
						p0 = [5., 2., -0.2, 30.]) 
						        
				params[i,j ,...] = popt
				bf_st[i,j,:] = st.skewt_pdf(self.rm.v_los, *popt)

		return params, bf_st



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

		


def gaussian(v, mean, std):
	    return norm.pdf(v, loc = mean, scale = std)
