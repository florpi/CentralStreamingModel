from CentralStreamingModel.utils.read_probabilities import VD
from CentralStreamingModel.utils import fit_tools as fit
from CentralStreamingModel.biskewt import skewt as st
import numpy as np

def radial_tangential_read_skewtfit(tracer, boxsize, boxnumber, snapshot, log):

    #print('Fitting a biskewt distribution to measured radial and tangential velocity pdfs ...')
    n_parameters = 5

    velocity_statistics = VD(tracer, boxnumber, boxsize, snapshot)

	

    v = np.array(np.meshgrid(velocity_statistics.v.r, velocity_statistics.v.t)).T.reshape(-1,2)
    rbins = np.arange(0,np.max(velocity_statistics.r),velocity_statistics.wr,dtype=int)
    popt = np.zeros((len(rbins),n_parameters))
    pcov = np.zeros((len(rbins),n_parameters, n_parameters))

    for rbin,rbin_value in enumerate(rbins):

        initial_guess = np.asarray([1.,1.,1.,1.,1.])

        if log == True:

            mask = np.where(velocity_statistics.jointpdf[rbin].reshape(-1) != 0.)[0]

            reshaped_pdf = velocity_statistics.jointpdf[rbin].reshape(-1)

            popt[rbin,:], pcov[rbin,:,:] = fit.fit_pdf(st.log_skewt, v[mask],
                                       np.log10(reshaped_pdf[mask]), initial_guess)
        else:

            reshaped_pdf = velocity_statistics.jointpdf[rbin].reshape(-1)

            popt[rbin,:], pcov[rbin,:,:] = fit.fit_pdf(st.skewt, v,
                                       reshaped_pdf, initial_guess)


    return popt, pcov

def radial_tangential_skewtfit(r, v_r, v_t, jointpdf_rt, log):

    #print('Fitting a biskewt distribution to measured radial and tangential velocity pdfs ...')
    n_parameters = 5

    v = np.array(np.meshgrid(v_r, v_t)).T.reshape(-1,2)
    popt = np.zeros((len(r),n_parameters))
    pcov = np.zeros((len(r),n_parameters, n_parameters))

    for rbin,rbin_value in enumerate(r):

        initial_guess = np.asarray([1.,1.,1.,1.,1.])
        
        reshaped_pdf = jointpdf_rt[rbin].reshape(-1)

        if log == True:

            mask = np.where(jointpdf_rt[rbin].reshape(-1) != 0.)[0]


            popt[rbin,:], pcov[rbin,:,:] = fit.fit_pdf(st.log_skewt, v[mask],
                                       np.log10(reshaped_pdf[mask]), initial_guess)
        else:


            popt[rbin,:], pcov[rbin,:,:] = fit.fit_pdf(st.skewt, v,
                                       reshaped_pdf, initial_guess)

    return popt, pcov

