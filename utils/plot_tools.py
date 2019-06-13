import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter
from operator import attrgetter


def compute_mean_error(list_boxes, attribute, component):

    mean = np.mean([getattr(getattr(box, attribute), component) for box in list_boxes], axis=0)
    std = np.std([getattr(getattr(box, attribute), component) for box in list_boxes], axis=0)
    return mean, std 

def plot_mean_attribute(list_tracers, list_colors, labels,
    attribute, component, ylabel, title=None, save = None):
    
    
    for i, tracer in enumerate(list_tracers):
        mean, std = compute_mean_error(tracer, attribute, component)
        if labels is not None:
            plt.errorbar(getattr(tracer[0], 'r')[:-1], mean[:-1], yerr= std[:1], color=list_colors[i], label = labels[i])
        else:
            plt.errorbar(getattr(tracer[0], 'r')[:-1], mean[:-1], yerr= std[:-1], color=list_colors[i])

    plt.xlabel('r [Mpc/h]')
    if labels is not None:
        plt.legend()
    plt.ylabel(ylabel)


def jointplot(x, y, jointpdf, log=False):
    '''

    Plots the joint PDF of two random variables together with its marginals
        Args:
            x and y, random variables,
            jointpdf, their joint PDF

    '''
    nullfmt = NullFormatter()         # no labels
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    if(log == True):
        axScatter.contour(x, y, np.log10(jointpdf))
    else:
        axScatter.contour(x, y, jointpdf)


    axHistx.plot(y, abs(y[1] - y[0]) * np.sum(jointpdf, axis=0))
    axHisty.plot(abs(x[1] - x[0]) * np.sum(jointpdf, axis=-1), x)

    axScatter.set_xlabel(r'$v_r$ [Mpc/h]')
    axScatter.set_ylabel(r'$v_t$ [Mpc/h]')

    axHistx.set_ylabel('Marginal radial PDF')
    axHisty.set_xlabel('Marginal tangential PDF')

    plt.show()



def bestfit_jointplot(x, y, jointpdf, bestfitpdf, log=False, log_marginals=False, save=None):
    '''

    Plots the joint PDF of two random variables together with its marginals
        Args:
            x and y, random variables,
            jointpdf, their joint PDF

    '''
    nullfmt = NullFormatter()         # no labels
    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure()
    plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    vmin = np.min([jointpdf, bestfitpdf])
    vmax =  np.max([jointpdf, bestfitpdf])
    levels = np.linspace(vmin, vmax, 10)

    # the scatter plot:
    if log:
        if(vmin != 0):
            vmin = np.log10(vmin)
        else:
            vmin = np.log10(np.min(bestfitpdf)) # this one should never be == 0.
        vmax = np.log10(vmax)
        levels = np.log10(np.logspace(vmin, vmax, 5))
        cont = axScatter.contour(x, y, np.log10(jointpdf),vmin=vmin, vmax=vmax, levels=levels)
        axScatter.contour(x, y, np.log10(bestfitpdf),linestyles='dashed', vmin=vmin, vmax=vmax, levels=levels )
    else:
        cont = axScatter.contour(x, y, jointpdf, vmin=vmin, vmax=vmax, levels=levels)
        axScatter.contour(x, y, bestfitpdf, linestyles='dashed',vmin=vmin, vmax=vmax, levels=levels)
    if log_marginals:
        axHistx.semilogy(y, abs(y[1] - y[0]) * np.sum(jointpdf, axis=0), label='Measured')
        axHistx.semilogy(y, abs(y[1] - y[0]) * np.sum(bestfitpdf, axis=0), linestyle='--', color='purple', label='Best fit')
        axHisty.semilogx(abs(x[1] - x[0]) * np.sum(jointpdf, axis=-1), x, label='Measured' )
        axHisty.semilogx(abs(x[1] - x[0]) * np.sum(bestfitpdf, axis=-1), x, linestyle='--', color='purple', label='Best fit')


    else:
        axHistx.plot(y, abs(y[1] - y[0]) * np.sum(jointpdf, axis=0), label='Measured')
        axHistx.plot(y, abs(y[1] - y[0]) * np.sum(bestfitpdf, axis=0), linestyle='--', color='purple', label='Best fit')
        axHisty.plot(abs(x[1] - x[0]) * np.sum(jointpdf, axis=-1), x, label='Measured' )
        axHisty.plot(abs(x[1] - x[0]) * np.sum(bestfitpdf, axis=-1), x, linestyle='--', color='purple', label='Best fit')

    axScatter.set_xlabel(r'$v_r$ [Mpc/h]')
    axScatter.set_ylabel(r'$v_t$ [Mpc/h]')

    axHistx.set_ylabel('Radial ')
    axHisty.set_xlabel('Tangential ')

    if save:
        print('Saving plot...')
        plt.savefig(save)
        plt.close()
    else:
        plt.show()


def plot_attribute_residual(instance, models, attribute, ylabel, res_ylabel, title = None, save = None):

	fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = True, squeeze = True,
						gridspec_kw = {'height_ratios':[4,1]})


	if title is not None:
		plt.suptitle(title)


	get_attributes = attrgetter(f'measured.{attribute}.mean', f'measured.{attribute}.std')

	measured_mean, measured_std = get_attributes(instance)

	get_att = attrgetter('measured.s_c.mean')
	x_measured = get_att(instance)

	ax1.errorbar(x_measured, x_measured*x_measured*measured_mean,
				yerr = x_measured*x_measured*measured_std,
				label = 'measured', color = instance.measured.color,
				linestyle = '', marker = 'o', markersize = 3)


	for model in models:

		get_attributes = attrgetter(f'{model}.{attribute}', f'{model}.color')
		model_value, model_color = get_attributes(instance)

		get_att = attrgetter(f'{model}.s_c')
		x = get_att(instance)

		ax1.plot(x, x*x*model_value, label = model, color = model_color, linewidth = 2)

		step = int(len(x)/len(x_measured))
		ax2.plot(x_measured, (model_value[step-1::step] - measured_mean)/measured_std,
				color = model_color, linewidth = 2)

	ax2.fill_between(x, -1., 1., facecolor = 'yellow', alpha = 0.5)

	ax1.legend(bbox_to_anchor = [1., 1.])
	ax2.axhline(y = 0., linestyle='dashed', color='gray')
	ax2.set_ylim(-5.,5.)


	ax2.set_xlabel('s [Mpc/h]')
	ax1.set_ylabel(ylabel)
	ax2.set_ylabel(res_ylabel)

	plt.subplots_adjust(wspace = 0, hspace = 0)

	if save is not None:
		plt.savefig(save, dpi = 200, bbox_to_inches = 'tight')
		
def plot_attribute_percent(instance, models, attribute, ylabel, res_ylabel, title = None, save = None):

	fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = True, squeeze = True,
						gridspec_kw = {'height_ratios':[4,1]})


	if title is not None:
		plt.suptitle(title)


	get_attributes = attrgetter(f'measured.{attribute}.mean', f'measured.{attribute}.std')

	measured_mean, measured_std = get_attributes(instance)

	get_att = attrgetter('measured.s_c.mean')
	x_measured = get_att(instance)

	ax1.errorbar(x_measured, x_measured*x_measured*measured_mean,
				yerr = x_measured*x_measured*measured_std,
				label = 'measured', color = instance.measured.color,
				linestyle = '', marker = 'o', markersize = 3)


	for model in models:

		get_attributes = attrgetter(f'{model}.{attribute}', f'{model}.color')
		model_value, model_color = get_attributes(instance)

		get_att = attrgetter(f'{model}.s_c')
		x = get_att(instance)

		ax1.plot(x, x*x*model_value, label = model, color = model_color, linewidth = 2)

		step = int(len(x)/len(x_measured))
		ax2.plot(x_measured, (model_value[step-1::step] - measured_mean)/model_value[step-1::step],
				color = model_color, linewidth = 2)

	ax2.fill_between(x, -0.01, 0.01, facecolor = 'yellow', alpha = 0.5)

	ax1.legend()
	ax2.axhline(y = 0., linestyle='dashed', color='gray')
	ax2.set_ylim(-0.05,0.05)


	ax2.set_xlabel('s [Mpc/h]')
	ax1.set_ylabel(ylabel)
	ax2.set_ylabel(res_ylabel)

	plt.subplots_adjust(wspace = 0, hspace = 0)
	
	if save is not None:
		plt.savefig(save, dpi = 200, bbox_to_inches = 'tight')
	
