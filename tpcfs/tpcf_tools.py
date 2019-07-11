from halotools.mock_observables import rp_pi_tpcf, tpcf, s_mu_tpcf, tpcf_multipole
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
import pickle
import h5py

def compute_real_tpcf(tracer, box, r_bins, boxsize = 1024., num_threads = 16, m_min = 0., m_max =1.e17, gravity = 'GR'):


	hdf5_dir = '/cosma6/data/dp004/dc-cues1/simulations/RSD/'
	filename = f'{tracer}_1024_b{box}.h5'
	# if tracer == 'halos':

		# simulation_path = f'/madfs/data/FR/Baojiu/fr_data/B1024-PM1024/gadget_data/2015/halo_catalogues/{gravity}/box{box}/38/'
		# data_filename = f'Rockstar_M200c_{gravity}_B{box}_B1024_NP1024_S38.dat'
		# m200c,x,y,z,vx,vy,vz,pid = np.loadtxt(simulation_path+ data_filename ,unpack=True, usecols = [2,8,9,10,11,12,13,-1])

		# m_min = 8.e12
		# threshold = (m200c > m_min) & (m200c < m_max) & (pid == -1) 
	
	# elif tracer == 'galaxies':
		# simulation_path = '/cosma5/data/dp004/dc-hern1/GR_tests/catalogues/HOD_fixed_n_xi/'
		# data_filename = f'{gravity}_HOD_z00_B{box:1d}.txt'

		# x, y, z, vx, vy, vz, r200c, rs, m200c, type_id= np.loadtxt(simulation_path+ data_filename ,unpack=True)
		# threshold = (m200c > m_min) & (m200c < m_max)

	# pos = np.stack([x,y,z],axis=-1)
	# pos = pos[threshold,:]
	# print(pos.shape)
	
	with h5py.File( hdf5_dir + filename , 'r') as f:

		pos = f['GroupPos'][:]
	
	real_tpcf = tpcf(pos, r_bins, period = boxsize, num_threads = num_threads)

	r_bins_c = 0.5*(r_bins[1:] + r_bins[:-1])

	tpcf_dict = {'r': r_bins_c, 'tpcf': real_tpcf}


	with open(f'/cosma6/data/dp004/dc-cues1/simulations/RSD/tpcfs/real/{tracer}_b{box:1d}.pickle', 'wb') as fp:
		pickle.dump(tpcf_dict, fp, protocol = pickle.HIGHEST_PROTOCOL)


def compute_redshift_tpcf(tracer, box,	r_bins, mu_bins = None,
						  m_min = 0., m_max = 1.e17, num_threads = 16, 
						  los_direction=2, boxsize=1024., save = True, gravity = 'GR'):

	lamda = 0.6844
	matter = 1. - lamda

	hdf5_dir = '/cosma6/data/dp004/dc-cues1/simulations/RSD/'
	filename = f'{tracer}_1024_b{box}.h5'

	# if tracer == 'halos':

		# simulation_path = f'/madfs/data/FR/Baojiu/fr_data/B1024-PM1024/gadget_data/2015/halo_catalogues/{gravity}/box{box}/38/'
		# data_filename = f'Rockstar_M200c_{gravity}_B{box}_B1024_NP1024_S38.dat'


		# m200c,x,y,z,vx,vy,vz,pid = np.loadtxt(simulation_path+ data_filename ,unpack=True, usecols = [2,8,9,10,11,12,13,-1])
		# m_min = 8.e12
		# threshold = (m200c > m_min) & (m200c < m_max) & (pid == -1) 
	
	# elif tracer == 'galaxies':
		# simulation_path = '/cosma5/data/dp004/dc-hern1/GR_tests/catalogues/HOD_fixed_n_xi/'
		# data_filename = f'{gravity}_HOD_z00_B{box:1d}.txt'

		# x, y, z, vx, vy, vz, r200c, rs, m200c, type_id= np.loadtxt(simulation_path+ data_filename ,unpack=True)
		# threshold = (m200c > m_min) & (m200c < m_max)

	# pos = np.stack([x,y,z],axis=-1)
	# vel = np.stack([vx,vy,vz],axis=-1)

	# threshold = (m200c > m_min) & (m200c < m_max)

	# pos = pos[threshold,:]
	#vel = vel[threshold,:]

	with h5py.File( hdf5_dir + filename , 'r') as f:

		pos = f['GroupPos'][:]
		vel = f['GroupVel'][:]
	
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


	pi_sigma = rp_pi_tpcf(s_pos, r_bins, r_bins, period=boxsize,
				estimator=u'Landy-Szalay', num_threads=num_threads)
		
	s_mu = s_mu_tpcf(s_pos, r_bins, mu_bins, period=boxsize,
				estimator=u'Landy-Szalay', num_threads= num_threads)
		
	mu_bins_c = 0.5 * (mu_bins[1:] + mu_bins[:-1])

	mono = tpcf_multipole(s_mu, mu_bins, order = 0)
	quad = tpcf_multipole(s_mu, mu_bins, order = 2)
	hexa = tpcf_multipole(s_mu, mu_bins, order = 4)

	tpcf_dict = {'r': r_bins_c, 'mu_bins': mu_bins_c, 'pi_sigma': pi_sigma, 's_mu': s_mu,
					'mono': mono, 'quad': quad, 'hexa': hexa}

	if save == True:
		with open(f'/cosma6/data/dp004/dc-cues1/simulations/RSD/tpcfs/redshift/{tracer}_b{box:1d}.pickle', 'wb') as fp:
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


