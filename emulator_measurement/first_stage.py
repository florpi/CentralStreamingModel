from CentralStreamingModel.tpcfs import tpcf_tools
import numpy as np
import sys
import time

t_init = time.time()
# Common properties among all cosmologies

box = int(sys.argv[1])


data_dir = f'/raid/nbody/baorsd/run{box:03d}/halo_catalog/'
boxsize = 2000. # Mpc/h 
snapshot = 11 # Redshift approx 0.5

if box == 15:
	filename_root = f'R115_S{snapshot:03d}'

else:
	filename_root = f'S{snapshot:03d}_cen_rockstar'

print(f'Computing first stage for simulation @ {data_dir + filename_root}')

n_threads = 16 

m200c = np.fromfile(data_dir + filename_root + '_mass.bin', dtype = np.float32)

# i) Measure minimum halo mass given target number density

target_number_density = 3.e-4

n_objects = int(target_number_density * boxsize**3)
m_halo_min = np.sort(m200c)[-n_objects]

# ii) Measure real and redshift space two point correlation functions,

r_bins = np.logspace(-0.4, np.log10(150.), 300)
tpcf_tools.compute_real_tpcf(data_dir + filename_root, box, snapshot,r_bins, num_threads = n_threads, m_min = m_halo_min)

'''
r_bins = np.arange(0., 50., 1.)
r_bins[0] = 0.0001
n_mu_bins = 60
mu_bins = np.linspace(0.,1.,n_mu_bins)


tpcf_tools.compute_redshift_tpcf(data_dir + filename_root, box, snapshot,
			 r_bins, mu_bins, m_min = m_halo_min, num_threads = n_threads)

'''

print(f'Measuring first stage took {time.time() - t_init} seconds')
