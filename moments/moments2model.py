import numpy as np
from scipy.stats import norm
from halotools.mock_observables import tpcf_multipole
from CentralStreamingModel.integral import real2redshift as real2red
from CentralStreamingModel.skewt import skewt as st
from CentralStreamingModel.skewt import skewt_moments
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d, interp2d
from CentralStreamingModel.projection import generating_moments
from scipy.integrate import simps
from scipy.interpolate import RegularGridInterpolator
from CentralStreamingModel.pearson import pearson as ps
from CentralStreamingModel.pearson import pearson_moments





class Model:

	def __init__(self, rm, expectations,  model, direct = None, p0 = None):

		self.rm = rm
		self.expectations = expectations

		if rm.tracer == 'halos':

			fill_value = ( -1., self.rm.tpcf_dict['tpcf'][-1])

		else:

			fill_value = (self.rm.tpcf_dict['tpcf'][0], self.rm.tpcf_dict['tpcf'][-1])

		self.tpcf = interp1d(self.rm.tpcf_dict['r'], self.rm.tpcf_dict['tpcf'], kind = 'linear',
                fill_value = fill_value, bounds_error = False)
	
		self.s = np.arange(0., 50., 1.)
		self.s_c = 0.5 * (self.s[1:] + self.s[:-1])
		self.mu =  np.sort(1 - np.geomspace(0.0001, 1., 60))
		self.mu_c = 0.5 * (self.mu[1:] + self.mu[:-1])


		mode = 'continuous'
		if model == 'measured':

			self.interpolator= self.interpolate_los_pdf()
			self.jointpdf_los = self.measured_los_pdf()
			self.color = 'black'

		elif model  == 'gaussian':

			if direct is not None:
				self.mean_r = direct[0]
				self.c_r = direct[1]
				self.c_t = direct[2]
				self.jointpdf_los = self.moments2gaussian_direct()

			else:
				self.jointpdf_los = self.moments2gaussian()
			self.color = 'forestgreen'

		elif model  == 'bf-gaussian':

			self.jointpdf_los = self.bfgaussian()
			self.color = 'darkseagreen'

		elif model == 'st':
			
			self.jointpdf_los = self.moments2st()
			self.color = 'royalblue'

		elif model == 'bf-st':
			
			self.jointpdf_los = self.bfst()
			self.color = 'indianred'

		elif model == 'pearson':
			
			self.jointpdf_los = self.moments2pearson()
			self.color = 'yellow'


		self.multipoles(self.s, self.mu, mode)

	def interpolate_los_pdf(self):

		return RegularGridInterpolator((self.rm.r_perp, self.rm.r_parallel, self.rm.v_los), 
							self.rm.jointpdf_los,
				            bounds_error = False, 
							fill_value = 0.)

	def measured_los_pdf(self):

	
		def function_los(v_los, r_perp, r_parallel):

			v_bins = -1 * np.ones_like(v_los)

			v_los_c = self.rm.v_los - 0.5 * (self.rm.v_los[1] - self.rm.v_los[0])
			r_perp_c = self.rm.r_perp - 0.5 * (self.rm.r_perp[1] - self.rm.r_perp[0])
			r_par_c = self.rm.r_parallel - 0.5 * (self.rm.r_parallel[1] - self.rm.r_parallel[0])

			valid_mask = (v_los < np.max(v_los_c)) & (v_los > np.min(v_los_c))

			v_bins = np.digitize(v_los, v_los_c) - 1

			r_perp_bins = np.digitize(r_perp, r_perp_c) - 1

			r_par_bins = np.digitize(np.abs(r_parallel), r_par_c) - 1



			output = self.rm.jointpdf_los[r_perp_bins, r_par_bins, v_bins]

			# Filter output for bounds
			#output[valid_mask] = np.zeros_like(output[valid_mask])
			filtered = np.where(valid_mask, output, 0.0)

			return filtered 

		return function_los

		'''
		v_los = v_los.flatten()
		r_perp = r_perp.flatten()
		r_parallel = r_parallel.flatten()

		points = np.meshgrid(v_los, r_perp, r_parallel)
		flat = np.array([m.flatten() for m in points])

		out_array = self.interpolator(flat.T)


		return out_array.reshape(*points[0].shape)
		'''


	def moments2gaussian(self):
		
		def function_los(vlos, rperp, rparallel):

			r = np.sqrt(rperp** 2 + rparallel** 2)
			mu = rparallel/r
			
			mean, std, gamma1, gamma2 = generating_moments.project(self.expectations, r, mu)

			return norm.pdf(vlos, loc = mean, scale = std)

		return function_los

	def moments2gaussian_direct(self):
		
		def function_los(vlos, rperp, rparallel):

			r = np.sqrt(rperp** 2 + rparallel** 2)
			mu = rparallel/r
			
			mean = mu * self.mean_r(r)
			std = np.sqrt( self.c_r(r) * mu ** 2 + self.c_t(r) * (1 - mu**2))

			return norm.pdf(vlos, loc = mean, scale = std)

		return function_los

	
	
	def bfgaussian(self):


		popt = np.zeros((self.rm.r_perp.shape[0], 
							self.rm.r_parallel.shape[0], 2))

		for i, rperp in enumerate(self.rm.r_perp):
			for j, rpar in enumerate(self.rm.r_parallel):

				popt[i,j, :], pcov = curve_fit(gaussian, self.rm.v_los, self.rm.jointpdf_los[i,j,:]) 
						        

		print('Found optimal parameters')

		#self.interpolators = []

		#for i in range(popt.shape[-1]):
		#	self.interpolators.append(interp2d(self.rm.r_parallel, self.rm.r_perp, popt[..., i], fill_value = popt[0,0,i]))

		def function_los(vlos, rperp, rparallel):

			r_perp_c = self.rm.r_perp - 0.5 * (self.rm.r_perp[1] - self.rm.r_perp[0])
			r_par_c = self.rm.r_parallel - 0.5 * (self.rm.r_parallel[1] - self.rm.r_parallel[0])

			r_perp_bins = np.digitize(rperp, r_perp_c) - 1
			r_par_bins = np.digitize(np.abs(rparallel), r_par_c) - 1

			mean = popt[r_perp_bins, r_par_bins, 0]
			std = popt[r_perp_bins, r_par_bins, 1]

			#mean = self.interpolators[0](rparallel[0, :], rperp[:,0])
			#std = self.interpolators[1](rparallel[0,:], rperp[:,0])

			return norm.pdf(vlos, loc =  mean, scale = std)

		return function_los 


	def moments2st(self): 


		s = np.sqrt(self.rm.r_perp.reshape(-1, 1)**2 + self.rm.r_parallel.reshape(1, -1)**2)

		mu = self.rm.r_parallel/s

		#mean, std, gamma1, gamma2 = generating_moments.project(self.expectations, s, mu)

		mean = simps(self.rm.v_los * self.rm.jointpdf_los, self.rm.v_los, axis = -1)

		std = np.sqrt(simps( (self.rm.v_los - mean[..., np.newaxis])**2 * self.rm.jointpdf_los, self.rm.v_los, axis = -1))

		gamma1 = simps( (self.rm.v_los - mean[..., np.newaxis])**3 * self.rm.jointpdf_los, self.rm.v_los, axis = -1)/std**3

		gamma2 = simps( (self.rm.v_los - mean[..., np.newaxis])**4 * self.rm.jointpdf_los, self.rm.v_los, axis = -1)/std**4 - 3.



		self.params = np.zeros((self.rm.r_perp.shape[0],
						self.rm.r_parallel.shape[0],
						4))

		for i, rperp in enumerate(self.rm.r_perp):
			for j, rpar in enumerate(self.rm.r_parallel):

				
				self.params[i,j,:] = skewt_moments.moments2parameters(
																mean[i,j], std[i,j], gamma1[i,j], gamma2[i,j]
																)

		print('Found params from moments')




		def function_los(vlos, r_perp, r_parallel):

			r_perp_c = self.rm.r_perp - 0.5 * (self.rm.r_perp[1] - self.rm.r_perp[0])
			r_par_c = self.rm.r_parallel - 0.5 * (self.rm.r_parallel[1] - self.rm.r_parallel[0])

			r_perp_bins = np.digitize(r_perp, r_perp_c) - 1
			r_par_bins = np.digitize(np.abs(r_parallel), r_par_c) - 1

			w = self.params[r_perp_bins, r_par_bins, 1]
			v_c = self.params[r_perp_bins, r_par_bins, 0]
			gamma1 = self.params[r_perp_bins, r_par_bins, 2]
			gamma2 = self.params[r_perp_bins, r_par_bins, 3]

			return st.skewt_pdf(vlos, w, v_c, gamma1, gamma2)

		return function_los

	'''
	def moments2st(self): 

		def function_los(vlos, rperp, rparallel):
			r = np.sqrt(rperp** 2 + rparallel** 2)
			mu = rparallel/r



			mean, std, gamma1, gamma2 = generating_moments.project(self.expectations, r, mu)

			v_c, w, alpha, nu = [np.zeros_like(mean) for _ in range(4)]

			for i, rp in enumerate(range(mean.shape[0])):
				for j, rpar in enumerate(range(mean.shape[1])):

					v_c[i,j], w[i,j], alpha[i,j], nu[i,j] = skewt_moments.moments2parameters(
																mean[i,j], std[i,j], gamma1[i,j], gamma2[i,j]
																)

			return st.skewt_pdf(vlos, w, v_c, alpha, nu)

		return function_los
	'''

	def moments2pearson(self): 


		s = np.sqrt(self.rm.r_perp.reshape(-1, 1)**2 + self.rm.r_parallel.reshape(1, -1)**2)

		mu = self.rm.r_parallel/s

		#mean, std, gamma1, gamma2 = generating_moments.project(self.expectations, s, mu)

		mean = simps(self.rm.v_los * self.rm.jointpdf_los, self.rm.v_los, axis = -1)

		std = np.sqrt(simps( (self.rm.v_los - mean[..., np.newaxis])**2 * self.rm.jointpdf_los, self.rm.v_los, axis = -1))

		gamma1 = simps( (self.rm.v_los - mean[..., np.newaxis])**3 * self.rm.jointpdf_los, self.rm.v_los, axis = -1)/std**3

		gamma2 = simps( (self.rm.v_los - mean[..., np.newaxis])**4 * self.rm.jointpdf_los, self.rm.v_los, axis = -1)/std**4 - 3.



		self.params = np.zeros((self.rm.r_perp.shape[0],
						self.rm.r_parallel.shape[0],
						4))

		for i, rperp in enumerate(self.rm.r_perp):
			for j, rpar in enumerate(self.rm.r_parallel):

				
				self.params[i,j,:] = pearson_moments.moments2parameters(
																mean[i,j], std[i,j], gamma1[i,j], gamma2[i,j], p0 = (3.,3.)
																)

		print('Found params from moments')


		self.params[0,0,:] = self.params[1,1,:]



		def function_los(vlos, r_perp, r_parallel):

			r_perp_c = self.rm.r_perp - 0.5 * (self.rm.r_perp[1] - self.rm.r_perp[0])
			r_par_c = self.rm.r_parallel - 0.5 * (self.rm.r_parallel[1] - self.rm.r_parallel[0])

			r_perp_bins = np.digitize(r_perp, r_perp_c) - 1
			r_par_bins = np.digitize(np.abs(r_parallel), r_par_c) - 1

			lamda = self.params[r_perp_bins, r_par_bins, 0]
			a = self.params[r_perp_bins, r_par_bins, 1]
			m = self.params[r_perp_bins, r_par_bins, 2]
			n = self.params[r_perp_bins, r_par_bins, 3]

			n[n> 500] = 500

			print(np.sum(n > 500))


			return ps.pearson(vlos, lamda, a, m, n)

		return function_los



	def bfst(self):

		self.popt = np.zeros((self.rm.r_perp.shape[0], 
							self.rm.r_parallel.shape[0], 4))


		for i, rperp in enumerate(self.rm.r_perp):
			for j, rpar in enumerate(self.rm.r_parallel):

				self.popt[i, j, :], pcov = curve_fit(st.skewt_pdf, self.rm.v_los, self.rm.jointpdf_los[i,j,:])
						#p0 = [5., 2., -0.2, 30.]) 
						        
		print('Found optimal parameters')

		#self.interpolators = []

		#for i in range(self.popt.shape[-1]):
		#	self.interpolators.append(interp2d(self.rm.r_parallel, self.rm.r_perp, self.popt[..., i]))

		def function_los(vlos, rperp, rparallel):

			r_perp_c = self.rm.r_perp - 0.5 * (self.rm.r_perp[1] - self.rm.r_perp[0])
			r_par_c = self.rm.r_parallel - 0.5 * (self.rm.r_parallel[1] - self.rm.r_parallel[0])

			r_perp_bins = np.digitize(rperp, r_perp_c) - 1
			r_par_bins = np.digitize(np.abs(rparallel), r_par_c) - 1

			w = self.popt[r_perp_bins, r_par_bins, 0]
			v_c = self.popt[r_perp_bins, r_par_bins, 1]
			gamma1 = self.popt[r_perp_bins, r_par_bins, 2]
			gamma2 = self.popt[r_perp_bins, r_par_bins, 3]


			#w = self.interpolators[0](rparallel[0,:], rperp[:,0])
			#v_c = self.interpolators[1](rparallel[0,:], rperp[:,0])
			#gamma1 = self.interpolators[2](rparallel[0,:], rperp[:,0])
			#gamma2 = self.interpolators[3](rparallel[0,:], rperp[:,0])

			return st.skewt_pdf(vlos, w, v_c, gamma1, gamma2)

		return function_los 

	def multipoles(self, s, mu, mode):

		if mode == 'discrete':
			rparallel, self.integrand, pdf_contribution = real2red.compute_integrand_s_mu(s,
										mu, self.rm.tpcf_dict, self.jointpdf_los,
										self.rm.r_perp, self.rm.r_parallel, self.rm.v_los)

			
			self.s_mu = real2red.integrate(rparallel, self.integrand)
		else:
			self.s_mu = real2red.simps_integrate(self.s, self.mu, self.tpcf, self.jointpdf_los)

		self.mono = tpcf_multipole(self.s_mu, mu, order = 0)
		self.quad = tpcf_multipole(self.s_mu, mu, order = 2)
		self.hexa = tpcf_multipole(self.s_mu, mu, order = 4)


		self.s_c = 0.5 * (s[1:] + s[:-1])
		self.mu_c = 0.5 * (mu[1:] + mu[:-1])

		


def gaussian(v, mean, std):
	    return norm.pdf(v, loc = mean, scale = std)

