from scipy.integrate import simps
from joblib import Parallel, delayed
from scipy.interpolate import interp1d, interp2d
import numpy as np
import itertools

def vt_integrand(vt, vlos, theta, vr_bins, vr_limit, pdf, interpolate):

    vr = (vlos - vt * np.sin(theta))/np.cos(theta)
    threshold = (vr > -vr_limit) & (vr < vr_limit )
    pdf_thresh = np.zeros_like(vt)

    if interpolate == False:
        vr_bin = (np.zeros_like(vr) - 1).astype(int)

        wr_bin = abs(vr_bins[1] - vr_bins[0])
        vr_bin[threshold] = np.digitize(vr[threshold], vr_bins - wr_bin) - 1 

        pdf_thresh[threshold] = np.diagonal(pdf[:,vr_bin])[threshold]

    else:
        if(vr[0] > vr[1]):
            pdf_thresh = np.diagonal(pdf(vr[::-1], vt)[:,::-1])
        else:
            pdf_thresh = np.diagonal(pdf(vr, vt))

    return pdf_thresh/np.cos(theta)

def vr_integrand(vr, vlos, theta, vt_bins, vt_limit, pdf, interpolate):    

    vt =(vlos - vr * np.cos(theta))/np.sin(theta)
    threshold = (vt > -vt_limit) & (vt < vt_limit)

    pdf_thresh = np.zeros_like(vr)
    if interpolate == False:
        vt_bin = (np.zeros_like(vt) - 1).astype(int)

        wt_bin = abs(vt_bins[1] - vt_bins[0])
        vt_bin[threshold] = np.digitize(vt[threshold], vt_bins - wt_bin) - 1 

        pdf_thresh[threshold] = np.diagonal(pdf[vt_bin,:])[threshold]

    else:
        # CAREFUL !! Unexpected behaviour from scipy interp2d, arguments should be ordered
        # + confusion that it returns the transpose
        pdf_thresh = np.diagonal(pdf(vr, vt[::-1])[::-1,:])

    return pdf_thresh/np.sin(theta)

def projection_integrand(rperp, rparallel, vlos, r_bins, vr, vt, jointpdf, interpolate):

    r = np.sqrt(rperp**2 + rparallel**2)
    wr_bins = abs(r_bins[1] - r_bins[0])
    rbin = np.digitize(r, r_bins - wr_bins) - 1 

    theta = np.arctan(rperp/rparallel)

    pdf = jointpdf[rbin,...]

    if interpolate == True:
        pdf = interp2d(vt, vr, pdf, kind='linear')
    
    if (rparallel >= rperp):
        integrand = vt_integrand(vt, vlos, theta, vr, np.max(vr), pdf, interpolate)
        return integrand, vt

    else:
        integrand = vr_integrand(vr,vlos, theta,vt,  np.max(vt), pdf, interpolate)
        return integrand, vr

def projection(rperp_bins, rparallel_bins, vlos_bins, r_bins, vr, vt, jointpdf, interpolate=False):

    los_pdf = np.zeros((rperp_bins.shape[0], rparallel_bins.shape[0], vlos_bins.shape[0]))

    for i, rperp in enumerate(rperp_bins):
        for j, rpar in enumerate(rparallel_bins):
            for k, vlos in enumerate(vlos_bins):
                integrand, v = projection_integrand(rperp, rpar, vlos, r_bins, vr, vt , jointpdf, interpolate)
                los_pdf[i,j,k] = simps(integrand, v)

    return los_pdf

def projection_parallel(rperp_bins, rparallel_bins, vlos_bins, r_bins, vr, vt, jointpdf, interpolate=False, n_cores=10):

    def projection_parallel_inloop(args):
        i,j = args
        los_pdf = np.zeros((vlos_bins.shape[0]))

        for k, vlos in enumerate(vlos_bins):
            integrand, v = projection_integrand(rperp_bins[i], rparallel_bins[j], vlos, r_bins, vr, vt , jointpdf, interpolate)
            los_pdf[k] = simps(integrand, v)

        return los_pdf

    dim1 = rperp_bins.shape[0]
    dim2 = rparallel_bins.shape[0]

    arg_instances = list((i,j) for i,j in itertools.product(range(dim1), range(dim2)))
    los_pdf = Parallel(n_jobs=n_cores, verbose=0)(map(delayed(projection_parallel_inloop), arg_instances))
    los_pdf = np.asarray(los_pdf).reshape((rperp_bins.shape[0], rparallel_bins.shape[0], vlos_bins.shape[0]))
    return los_pdf


