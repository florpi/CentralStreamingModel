
from models import RedshiftSpace
import models as md
import pickle
import matplotlib.pyplot as plt
import numpy as np
import importlib
import seaborn as sns
from CentralStreamingModel.utils import plot_tools as pt
sns.set_context('talk')
import time

plots_root = 'plots/halos_together_'
n_boxes = range(1,15)
boxsize = 2000
snapshot = 20

rs = []
for i, box in enumerate(n_boxes):
    rs.append(RedshiftSpace(box, boxsize, snapshot))

print('Read all boxes')

t1 = time.time()
mean_rs = md.MeanRedshiftSpace( boxsize, snapshot, rs)

print(f'It took {time.time() - t1} seconds.')



s_c = rs[0].measured.s_c
levels = np.arange(-4, 1, 0.3)

mean_rs.skewt_simps = md.Skewt(mean_rs.per_box_list[0].measured.r,
                            mean_rs.per_box_list[0].measured.v_r, mean_rs.per_box_list[0].measured.v_t,
                             mean_rs.measured.jointpdf_rt.mean, mean_rs.mean_tpcf_dict, truncate = 20,
                            integration = 'simpsons', color = 'indianred')

mean_rs.skewt_norm20 = md.Skewt(mean_rs.per_box_list[0].measured.r,
                            mean_rs.per_box_list[0].measured.v_r, mean_rs.per_box_list[0].measured.v_t,
                             mean_rs.measured.jointpdf_rt.mean, mean_rs.mean_tpcf_dict,
                            truncate = 20., integration = 'quad_norm', color = 'blue')

mean_rs.skewt_norm15 = md.Skewt(mean_rs.per_box_list[0].measured.r,
                            mean_rs.per_box_list[0].measured.v_r, mean_rs.per_box_list[0].measured.v_t,
                             mean_rs.measured.jointpdf_rt.mean, mean_rs.mean_tpcf_dict,
                            truncate = 15., integration = 'quad_norm', color = 'yellow')

mean_rs.skewt_norm17 = md.Skewt(mean_rs.per_box_list[0].measured.r,
                            mean_rs.per_box_list[0].measured.v_r, mean_rs.per_box_list[0].measured.v_t,
                             mean_rs.measured.jointpdf_rt.mean, mean_rs.mean_tpcf_dict,
                            truncate = 17., integration = 'quad_norm', color = 'brown')

pt.plot_attribute_residual( mean_rs, ['streaming','gaussian', 'skewt_simps', 'skewt_norm20',
							'skewt_norm15', 'skewt_norm17'],
                    'monopole',  r'$s^2\xi_0$ (s)', 
               r'$\frac{\xi_{0,model} - \xi_{0, measured}}{\sigma_{0, measured}}$',
				save = f'{plots_root}monopole_sigma.png')

pt.plot_attribute_residual( mean_rs, ['streaming','gaussian', 'skewt_simps', 'skewt_norm20',
							'skewt_norm15', 'skewt_norm17'],
                    'quadrupole',  r'$s^2\xi_2$ (s)', 
               r'$\frac{\xi_{2,model} - \xi_{2, measured}}{\xi_{2, measured}}$',
				save = f'{plots_root}quadrupole_sigma.png')

pt.plot_attribute_residual( mean_rs, ['streaming','gaussian', 'skewt_simps', 'skewt_norm20',
							'skewt_norm15', 'skewt_norm17'],
                    'hexadecapole',  r'$s^2\xi_4$ (s)', 
               r'$\frac{\xi_{4,model} - \xi_{4, measured}}{\sigma_{4, measured}}$',
				save = f'{plots_root}hexadecapole_sigma.png')

n_wedges = 5

for wedge in range(n_wedges):
    
    pt.plot_attribute_residual( mean_rs, ['streaming','gaussian', 'skewt_simps', 'skewt_norm20',
							'skewt_norm15', 'skewt_norm17'],
                    f'wedge_{wedge}', r'$s^2\xi_w$ (s)',
        r'$\frac{\xi_{w,model} - \xi_{w, measured}}{\sigma_{w, measured}}$',
        title =f'{mean_rs.skewt.wedges_bins[wedge]:.2f} < mu < {mean_rs.skewt.wedges_bins[wedge+1]:.2f}',
		save = f'{plots_root}w{wedge}_sigma.png')


