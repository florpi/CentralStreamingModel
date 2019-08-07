from halotools.mock_observables import rp_pi_tpcf, tpcf, s_mu_tpcf, tpcf_multipole
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import pickle
import h5py

def compute_real_tpcf(tracer, box, r_bins, boxsize = 1024., num_threads = 16,
		m_min = 0., m_max =1.e17, gravity = 'GR', downsampling = None, extra = None):


	hdf5_dir = '/cosma6/data/dp004/dc-cues1/simulations/RSD/'
	filename = f'{tracer}_1024_b{box}.h5'
	if extra:
		filename = f'{tracer}_1024_b{box}_{extra}.h5'
	if tracer == 'particles':
		filename = f'{tracer}_1024.h5'

	with h5py.File( hdf5_dir + filename , 'r') as f:

		pos = f['GroupPos'][:]
		if tracer != 'particles':
			m200c = f['GroupMass'][:]
			threshold = (m200c > m_min) & (m200c < m_max)
			pos = pos[threshold,...]

	if downsampling:

		print(f'There are {pos.shape[0]} particles')
		idx = np.random.randint(low = 0, high = pos.shape[0],
				size = int(downsampling * pos.shape[0]))
		pos = pos[idx, :]
		print(f'Using {pos.shape[0]} particles')

	real_tpcf = tpcf(pos, r_bins, period = boxsize, num_threads = num_threads)

	del pos 

	r_bins_c = 0.5*(r_bins[1:] + r_bins[:-1])

	tpcf_dict = {'r': r_bins_c, 'tpcf': real_tpcf}

	saveto = f'/cosma6/data/dp004/dc-cues1/simulations/RSD/tpcfs/real/{tracer}_b{box:1d}'

	if m_min != 0.:
		saveto += '_m2'

	if m_max != 1.e17:
		saveto += '_m1'

	if downsampling is not None:
		saveto += f'_d{downsampling}'

	if extra:
		saveto += '_' + extra

	with open(saveto + '.pickle', 'wb') as fp:
		print(f'Saving file to {saveto}')
		pickle.dump(tpcf_dict, fp, protocol = pickle.HIGHEST_PROTOCOL)


def compute_redshift_tpcf(tracer, box,	r_bins, mu_bins = None,
						  m_min = 0., m_max = 1.e17, num_threads = 16, 
						  los_direction=2, boxsize=1024., save = True,
						  gravity = 'GR', downsampling = None, extra = None):

	lamda = 0.6844
	matter = 1. - lamda

	hdf5_dir = '/cosma6/data/dp004/dc-cues1/simulations/RSD/'
	filename = f'{tracer}_1024_b{box}.h5'
	if extra:
		filename = f'{tracer}_1024_b{box}_{extra}.h5'

	if tracer == 'particles':
		filename = f'{tracer}_1024.h5'

	with h5py.File( hdf5_dir + filename , 'r') as f:

		pos = f['GroupPos'][:]
		vel = f['GroupVel'][:]

		if tracer != 'particles':
			m200c = f['GroupMass'][:]
			threshold = (m200c > m_min) & (m200c < m_max)
			pos = pos[threshold, :]
			vel = vel[threshold, :]

	if downsampling:

		print(f'There are {pos.shape[0]} particles')
		idx = np.random.randint(low = 0, high = pos.shape[0],
				size = int(downsampling * pos.shape[0]))
		pos = pos[idx, :]
		vel = vel[idx, :]
		print(f'Using {pos.shape[0]} particles')


	s_pos = pos.copy()
	
	s_pos[:, los_direction] += vel[:, los_direction]/100. # to Mpc/h
	

	s_pos[:, los_direction][s_pos[:, los_direction]  > boxsize] -= boxsize

	s_pos[:, los_direction][s_pos[:, los_direction]  < 0.] += boxsize

	# rp_pi_tpcf always los = z
	if(los_direction != 2): 
		s_pos_old = s_pos.copy()
		s_pos[:,2] = s_pos_old[:,los_direction]
		s_pos[:,los_direction] = s_pos_old[:,2]
	
	r_bins_c = 0.5*(r_bins[1:] + r_bins[:-1])

	del pos, vel


	pi_sigma = rp_pi_tpcf(s_pos, r_bins, r_bins, period=boxsize,
				estimator=u'Landy-Szalay', num_threads=num_threads)
		
	s_mu = s_mu_tpcf(s_pos, r_bins, mu_bins, period=boxsize,
				estimator=u'Landy-Szalay', num_threads= num_threads)
		
	del m200c, s_pos, vel
	mu_bins_c = 0.5 * (mu_bins[1:] + mu_bins[:-1])

	mono = tpcf_multipole(s_mu, mu_bins, order = 0)
	quad = tpcf_multipole(s_mu, mu_bins, order = 2)
	hexa = tpcf_multipole(s_mu, mu_bins, order = 4)

	tpcf_dict = {'r': r_bins_c, 'mu_bins': mu_bins_c, 'pi_sigma': pi_sigma, 's_mu': s_mu,
					'mono': mono, 'quad': quad, 'hexa': hexa}
	saveto = f'/cosma6/data/dp004/dc-cues1/simulations/RSD/tpcfs/redshift/{tracer}_b{box:1d}'

	if m_min != 0.:
		saveto += '_m2'

	if m_max != 1.e17:
		saveto += '_m1'


	if downsampling is not None:
		saveto += f'_d{downsampling}'

	if extra:
		saveto += '_' + extra


	if save == True:
		with open(saveto + '.pickle', 'wb') as fp:
			 pickle.dump(tpcf_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
	else:
		 return tpcf_dict
			


def pi_sigma2s_mu(s_center, pi_sigma, s_bins_center, mu_bins_center):

	s_mu = np.zeros((s_bins_center.shape[0], mu_bins_center.shape[0]))
	pi_sigma_interpolation = interp2d(s_center, s_center, pi_sigma, kind='linear')

	for i,s in enumerate(s_bins_center):
		for j,mu in enumerate(mu_bins_center):
			r_parallel = mu * s 
			r_perpendicular = np.sqrt(s**2 - r_parallel**2)

			s_mu[i,j] = pi_sigma_interpolation(r_parallel, r_perpendicular)

	return s_mu 
	  
def get_multipoles(s, pi_sigma, s_bins, mu_bins):

	s_center = 0.5*(s[1:] + s[:-1])
	s_bins_center = 0.5*(s_bins[1:] + s_bins[:-1])

	mu_bins_center = 0.5*(mu_bins[1:] + mu_bins[:-1])


	s_mu = pi_sigma2s_mu(s_center, pi_sigma, s_bins_center, mu_bins_center)

	mono = tpcf_multipole(s_mu, mu_bins, order =0) 
	quad = tpcf_multipole(s_mu, mu_bins, order =2) 
	hexa = tpcf_multipole(s_mu, mu_bins, order =4) 

	return s_mu, mono, quad, hexa


def get_wedges(s_c, tpcf_s_mu, mu_bins, wedges_bins):


	mu_bins_center = 0.5 * (mu_bins[1:] + mu_bins[:-1])
	mu_wedges_center = 0.5 * (wedges_bins[1:] + wedges_bins[:-1])

	delta_mu = mu_bins[1:] - mu_bins[:-1]

	wedge_idx = np.digitize(mu_bins_center, wedges_bins) - 1

	wedge_width = wedges_bins[1:] - wedges_bins[:-1]

	wedges = [1/wedge_width[i] * np.sum(delta_mu[wedge_idx ==i] * tpcf_s_mu[:, wedge_idx == i], axis=-1) for i in range(len(wedges_bins)-1)]

	return wedges


if __name__ == "__main__":

	r_bins = np.logspace(-0.4, np.log10(150.), 100)
	r_bins_c = 0.5*(r_bins[1:] + r_bins[:-1])
	n_mu_bins = 120
	mu_bins = np.linspace(0.,1.,n_mu_bins)

	n_boxes = 15

	for box in range(1, n_boxes+1):

		compute_real_tpcf(box, r_bins, num_threads = 16, m_min = 1.e13, m_max =1.e17)
		r_bins = np.arange(0., 50., 1.)
		r_bins[0] = 0.0001

		compute_redshift_tpcf(box, r_bins, mu_bins = None,
						  m_min = 1.e13, m_max = 1.e17, num_threads = 16, 
						  los_direction=2)


