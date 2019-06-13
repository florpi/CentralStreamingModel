import numpy as np
from halotools.mock_observables import tpcf_multipole
from CentralStreamingModel.utils.read_probabilities import VD, VD_los
from CentralStreamingModel.integral.real2redshift  import compute_integrand, compute_integrand_continuous, integrate, integrate_quad, integrate_quad_normalised
from CentralStreamingModel.tpcfs.tpcf_tools import get_multipoles, get_wedges
from CentralStreamingModel.gaussian import gsm
from CentralStreamingModel.biskewt import skewtfit
from CentralStreamingModel.projection.wrap_projection import skewt_los_pdf
import pickle


class RedshiftSpace:

	def __init__(self, box, boxsize, snapshot, tracer = 'halos', extra = None):

		self.tracer = tracer
		self.box = box
		self.boxsize = int(boxsize)
		self.snapshot = snapshot
	
		# Load measured real space 2pcf
		
		if self.tracer == 'halos':
			tpcf_filename = f"/raid/c-cuesta/tpcfs/real/box{100 + self.box}_s{self.snapshot:03d}.pickle"
		elif self.tracer == 'galaxies':
			tpcf_filename = f"/raid/c-cuesta/tpcfs/gals_real_tpcf_box{box}.pickle"
		else:
			tpcf_filename = None

		with open(tpcf_filename, "rb") as input_file:

			twopcf_dict = pickle.load(input_file)
		# Load measured line of sight pairwise velocity PDF

		self.measured = Measured(self.box, self.boxsize, self.snapshot, self.tracer, extra)

		self.measured.tpcf_dict = twopcf_dict
		self.r = twopcf_dict['r']
		self.measured.tpcf = twopcf_dict['tpcf']





class Measured():

	def __init__(self, box, boxsize, snapshot, tracer, extra):


		self.box = box
		self.tracer = tracer
		self.snapshot = snapshot
		self.pairwise_pdf_los = VD_los(self.tracer, box, boxsize, snapshot, extra)
		self.pairwise_pdf_rt = VD(self.tracer, box, boxsize, snapshot)

		self.r_parallel = self.pairwise_pdf_los.r.r
		self.r_perp = self.pairwise_pdf_los.r.t
		self.v_los = self.pairwise_pdf_los.v
		#threshold = (self.v_los > -10.) & (self.v_los < 10.)	

		self.r = self.pairwise_pdf_rt.r
		self.v_r = self.pairwise_pdf_rt.v.r
		self.v_t = self.pairwise_pdf_rt.v.t


		#self.jointpdf_los = self.pairwise_pdf_los.jointpdf[:,:, threshold]
		self.jointpdf_los = self.pairwise_pdf_los.jointpdf
		self.jointpdf_rt = self.pairwise_pdf_rt.jointpdf
		self.mean_r = self.pairwise_pdf_rt.mean.r
		self.std_r = self.pairwise_pdf_rt.std.r
		self.std_t = self.pairwise_pdf_rt.std.t

		#self.v_los = self.v_los[threshold]

		self.color = 'black'

		# Read pi_sigma, s_mu and multipoles
		self.read_redshift_tpcf()


	def read_redshift_tpcf(self):

		if self.tracer == 'halos':
			#s_tpcf_filename = f"/raid/c-cuesta/tpcfs/redshift_tpcf_box{self.box}_r2.pickle"
			s_tpcf_filename = f"/raid/c-cuesta/tpcfs/redshift/box{100 + self.box}_s{self.snapshot:03d}.pickle"
		elif self.tracer == 'galaxies':
			s_tpcf_filename = f"/raid/c-cuesta/tpcfs/gals_redshift_tpcf_box{self.box}.pickle"
		else:
			s_tpcf_filename = None

		with open(s_tpcf_filename, "rb") as input_file:
		    redshift_direct_measurement = pickle.load(input_file)

		self.s_c = redshift_direct_measurement['r']
		self.pi_sigma= redshift_direct_measurement['pi_sigma']
		self.s_mu =redshift_direct_measurement['s_mu']
		self.mu = redshift_direct_measurement['mu_bins']

		mu = np.linspace(0., 1., 60)

		self.monopole = tpcf_multipole(self.s_mu, mu, order=0)
		self.quadrupole = tpcf_multipole(self.s_mu, mu, order=2)
		self.hexadecapole = tpcf_multipole(self.s_mu, mu, order=4)

		self.wedges_bins = np.linspace(0.,1, 6)
		wedges = get_wedges(self.s_c, self.s_mu, mu, self.wedges_bins)

		for i, wedge in enumerate(wedges):
			setattr(self, f'wedge_{i}', wedge)



class components:

    def __init__(self, los, rt):
        self.los = los 
        self.rt = rt

class mean_error:

	def __init__(self, mean, std):

		self.mean = mean
		self.std = std


class MeanRedshiftSpace:
	def __init__(self, boxsize, snapshot, per_box_list):

		self.per_box_list = per_box_list

		self.measured = allModels()

		self.measured.color = 'black'

		list_attributes = ['jointpdf_los', 'jointpdf_rt', 's_mu', 'pi_sigma',
								'tpcf', 'monopole', 'quadrupole', 'hexadecapole',
								'mean_r', 'std_r', 'std_t', 's_c']

		self.n_wedges = 5

		for i in range(self.n_wedges):
			list_attributes.append(f'wedge_{i}')

		self.compute_mean_error(self.measured, 'measured', list_attributes)


		#r_c = 0.5 * (per_box_list[0].r[1:] + per_box_list[0].r[:-1])
		self.mean_tpcf_dict = {'r': per_box_list[0].r, 'tpcf': self.measured.tpcf.mean}


		print('Initiating Streaming')	
		self.streaming = Streaming(per_box_list[0].measured.v_los, per_box_list[0].measured.r_perp,
					per_box_list[0].measured.r_parallel, self.measured.jointpdf_los.mean,
				 self.mean_tpcf_dict)

		print('Finishied Streaming')	

		#truncate = np.max(per_box_list[0].measured.v_los)
		#truncate = None
		truncate = 20.

		#self.skewt = Skewt(per_box_list[0].measured.r, per_box_list[0].measured.v_r, per_box_list[0].measured.v_t,
		#								self.measured.jointpdf_rt.mean, self.mean_tpcf_dict, truncate)


		print('Initiating gaussian')	

		self.gaussian = Gaussian(per_box_list[0].measured.r, self.measured.mean_r.mean, self.measured.std_r.mean,
								self.measured.std_t.mean, self.mean_tpcf_dict, truncate) 

		print('Finished gaussian')

	def compute_mean_error(self, model_instance, model, list_attributes):

		for attribute in list_attributes:


			mean = np.mean([getattr(getattr(box_redshift_space, model),
			attribute) for box_redshift_space in self.per_box_list], axis =0 )
			std = np.std([getattr(getattr(box_redshift_space,model),
			attribute) for box_redshift_space in self.per_box_list],axis =0 )

			setattr(model_instance, attribute, mean_error(mean,std))

class allModels:

	def __init__(self):

		self.jointpdf_los = None
		self.jointpdf_rt = None

		self.pi_sigma = None
		self.s_mu = None
		self.monopole = None
		self.quadrupole = None
		self.hexadecapole = None


	def compute_pi_sigma_discrete(self, s_paral_full, real_tpcf_dict):

		self.s = np.arange(0., 50., 1.)
		self.s_c = 0.5 * (self.s[1:] + self.s[:-1])

		self.int_r_parallel, self.integrand, self.pdf_contribution = compute_integrand(self.s, self.s, 
			real_tpcf_dict, self.jointpdf_los, s_paral_full, self.v_los)

	
		self.pi_sigma = integrate(self.int_r_parallel, self.integrand)

	def compute_pi_sigma_continuous(self, real_tpcf_dict, truncate, integration):

		self.s = np.arange(0., 50., 1.)
		self.s_c = 0.5 * (self.s[1:] + self.s[:-1])

		if integration == 'quad':
			# Just to save the integrand
			self.int_r_parallel, self.integrand, self.pdf_contribution = compute_integrand_continuous(self.s, self.s, 
				real_tpcf_dict, self.jointpdf_los, truncate )


			self.pi_sigma = integrate_quad(self.s, self.s, real_tpcf_dict, self.jointpdf_los, truncate)

		if integration == 'quad_norm':
			# Just to save the integrand
			#self.int_r_parallel, self.integrand, self.pdf_contribution = compute_integrand_continuous(self.s, self.s, 
			#	real_tpcf_dict, self.jointpdf_los, truncate )

			self.pi_sigma = integrate_quad_normalised(self.s, self.s, real_tpcf_dict, self.jointpdf_los, truncate)

		else:	
			self.int_r_parallel, self.integrand, self.pdf_contribution = compute_integrand_continuous(self.s, self.s, 
				real_tpcf_dict, self.jointpdf_los, truncate )

			self.pi_sigma = integrate(self.int_r_parallel, self.integrand)


	def compute_s_mu(self):

		n_mu_bins = 60 
		mu_bins = np.linspace(0.,1.,n_mu_bins)

		self.s_mu, self.monopole, self.quadrupole, self.hexadecapole = \
					get_multipoles(self.s, self.pi_sigma, self.s, mu_bins)


		self.n_wedges = 5
		self.wedges_bins = np.linspace(0.,1, self.n_wedges + 1)

		wedges = get_wedges(self.s_c, self.s_mu, mu_bins, self.wedges_bins)

		for i, wedge in enumerate(wedges):
			setattr(self, f'wedge_{i}', wedge)


class Streaming(allModels):

	def __init__(self, v_los, r_perp, r_parallel, mean_pdf_los, mean_real_tpcf):
		allModels.__init__(self)

		self.jointpdf_los = mean_pdf_los
		self.v_los = v_los

		self.compute_pi_sigma_discrete(r_parallel, mean_real_tpcf)
		self.compute_s_mu()

		self.color = 'gray'


class Gaussian(allModels):

	def __init__(self, r, mean_r, std_r, std_t, mean_real_tpcf, truncate, integration = 'simpson'):
		# Due to Gaussian quadrature simpson integral seems accurate enough

		mean, std_r, std_t = gsm.interpolate_parameters(r, mean_r, std_r, std_t)
		
		self.jointpdf_los = gsm.gaussian_los_pdf(mean, std_r, std_t)
		#	remember it's a function

		self.compute_pi_sigma_continuous( mean_real_tpcf, truncate, integration)
			
		self.compute_s_mu()
		self.color = 'forestgreen'



class Skewt(allModels):

	def __init__(self, r, v_r, v_t, mean_jointpdf_rt, mean_real_tpcf, truncate, integration = 'simpson', popt = None, color = None):

		# get parameters fitting mean_joint_pdf_rt

		if popt is None:
			self.popt, self.pcov = skewtfit.radial_tangential_skewtfit(r, v_r, v_t, mean_jointpdf_rt, log=False)
		
		else:
			self.popt = popt

		print('Found popt')
		
		self.jointpdf_los = skewt_los_pdf(r, self.popt)
		print('Computing integral')
		self.compute_pi_sigma_continuous( mean_real_tpcf, truncate, integration)

		print('Finished streaming integral')
	

		self.compute_s_mu()
		if color:
			self.color = color

		else:
			self.color = 'indianred'

