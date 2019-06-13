
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

plots_root = 'plots/halos_infty_t20_'
n_boxes = range(1,16)
boxsize = 2000
snapshot = 20

rs = []
for i, box in enumerate(n_boxes):
    rs.append(RedshiftSpace(box, boxsize, snapshot))
t1 = time.time()
mean_rs = md.MeanRedshiftSpace( boxsize, snapshot, rs)
print(f'It took {time.time() - t1} seconds.')


s_c = rs[0].measured.s_c
levels = np.arange(-4, 1, 0.3)


colors = [mean_rs.measured.color] * len(levels)
plt.contour(s_c, s_c, 
            np.log10(mean_rs.measured.pi_sigma.mean).T, 
            levels=levels, colors=colors)

colors = [mean_rs.streaming.color] * len(levels)
plt.contour(mean_rs.streaming.s_c, mean_rs.streaming.s_c,
            np.log10(mean_rs.streaming.pi_sigma).T,
           levels=levels, colors=colors, linestyles='dashed')

colors = [mean_rs.gaussian.color] * len(levels)

plt.contour(mean_rs.gaussian.s_c, mean_rs.gaussian.s_c,
            np.log10(mean_rs.gaussian.pi_sigma).T,
           levels=levels, colors=colors)


colors = [mean_rs.skewt.color] * len(levels)

plt.contour(mean_rs.skewt.s_c, mean_rs.skewt.s_c,
            np.log10(mean_rs.skewt.pi_sigma).T,
           levels=levels, colors=colors)

#plt.xlim(0,30)

#plt.ylim(0,30)

plt.xlabel('$s_\perp$ [Mpc/h]')
plt.ylabel('$s_\parallel$ [Mpc/h]')
plt.savefig(f'{plots_root}pi_sigma.png', dpi = 200, bbox_inches = 'tight')

pt.plot_attribute_residual( mean_rs, ['streaming','gaussian', 'skewt'],
                    'monopole',  r'$s^2\xi_0$ (s)', 
               r'$\frac{\xi_{0,model} - \xi_{0, measured}}{\sigma_{0, measured}}$',
				save = f'{plots_root}monopole_sigma.png')

pt.plot_attribute_percent( mean_rs, ['streaming','gaussian', 'skewt'],
                    'monopole',  r'$s^2\xi_0$ (s)', 
               r'$\frac{\xi_{0,model} - \xi_{0, measured}}{\xi_{0, measured}}$',
				save = f'{plots_root}monopole_percent.png')

pt.plot_attribute_residual( mean_rs, ['streaming','gaussian', 'skewt'],
                    'quadrupole',  r'$s^2\xi_2$ (s)', 
               r'$\frac{\xi_{2,model} - \xi_{2, measured}}{\xi_{2, measured}}$',
				save = f'{plots_root}quadrupole_sigma.png')

pt.plot_attribute_percent( mean_rs, ['streaming','gaussian', 'skewt'],
                    'quadrupole',  r'$s^2\xi_2$ (s)', 
               r'$\frac{\xi_{2,model} - \xi_{2, measured}}{\xi_{2, measured}}$',
				save = f'{plots_root}quadrupole_percent.png')

pt.plot_attribute_residual( mean_rs, ['streaming','gaussian', 'skewt'],
                    'hexadecapole',  r'$s^2\xi_4$ (s)', 
               r'$\frac{\xi_{4,model} - \xi_{4, measured}}{\sigma_{4, measured}}$',
				save = f'{plots_root}hexadecapole_sigma.png')

pt.plot_attribute_percent( mean_rs, ['streaming','gaussian', 'skewt'],
                    'hexadecapole',  r'$s^2\xi_4$ (s)', 
               r'$\frac{\xi_{4,model} - \xi_{4, measured}}{\xi_{4, measured}}$',
				save = f'{plots_root}hexadecapole_percent.png')

n_wedges = 5

for wedge in range(n_wedges):
    
    pt.plot_attribute_residual( mean_rs, ['streaming', 'gaussian', 'skewt'],
                    f'wedge_{wedge}', r'$s^2\xi_w$ (s)',
        r'$\frac{\xi_{w,model} - \xi_{w, measured}}{\sigma_{w, measured}}$',
        title =f'{mean_rs.skewt.wedges_bins[wedge]:.2f} < mu < {mean_rs.skewt.wedges_bins[wedge+1]:.2f}',
		save = f'{plots_root}w{wedge}_sigma.png')

    pt.plot_attribute_percent( mean_rs, ['streaming', 'gaussian', 'skewt'],
                    f'wedge_{wedge}', r'$s^2\xi_w$ (s)',
        r'$\frac{\xi_{w,model} - \xi_{w, measured}}{\xi_{w, measured}}$',
        title =f'{mean_rs.skewt.wedges_bins[wedge]:.2f} < mu < {mean_rs.skewt.wedges_bins[wedge+1]:.2f}',
		save = f'{plots_root}w{wedge}_percent.png')

