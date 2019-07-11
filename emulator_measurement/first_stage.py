from CentralStreamingModel.tpcfs import tpcf_tools
import numpy as np
import sys
import time

t_init = time.time()
# Common properties among all cosmologies

tracer = sys.argv[1]
box = int(sys.argv[2])



boxsize = 1024. # Mpc/h 

print(f'Computing first stage for simulation: {tracer}, box number {box}') 

n_threads = 16 


# i) Measure minimum halo mass given target number density

#target_number_density = 3.e-4
#n_objects = int(target_number_density * boxsize**3)
#m_halo_min = np.sort(m200c)[-n_objects]

# ii) Measure real and redshift space two point correlation functions,

if tracer == 'halos':
	r_bins = np.logspace(-0.4, np.log10(150.), 300)
else:
	r_bins = np.logspace(-1., np.log10(150.), 300)

tpcf_tools.compute_real_tpcf(tracer, box, r_bins, num_threads = n_threads)

r_bins = np.arange(0., 50., 1.)
#r_bins = np.concatenate((np.arange(0.,10.,0.5), np.arange(10.,50.,1)))
r_bins[0] = 0.0001
n_mu_bins = 60
mu_bins = np.linspace(0.,1.,n_mu_bins)


tpcf_tools.compute_redshift_tpcf(tracer, box, 
			 r_bins, mu_bins,  num_threads = n_threads)


print(f'Measuring first stage took {time.time() - t_init} seconds')
