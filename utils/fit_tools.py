import numpy as np
import matplotlib.pyplot as plt 
import scipy.optimize as opt 
from CentralStreamingModel.biskewt import skewt as st

def KL_divergence(x,p,q):
    ''' 
    Measures the ammount of information lost by approximating a PDF
    https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained
    Inputs: x random variable array
            Observed Probability distribution p
            Approximated probability distribution q
    Output: kullback-leibler divergence 
    '''
    # Get rid off zero values
    _p = p[(p > 0.) & (q>0.)]
    _q = q[(p>0.) & (q>0.)]
    _x = x[(p>0.) & (q>0.)]
    deltax = abs(x[1]-x[0])

    return np.sum(deltax *_p * np.log(_p/_q))

def biKL_divergence(x,y,p,q):
    ''' 
    Measures the ammount of information lost by approximating a bivariate PDF
    https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained
    Inputs: v bivariate random variable array
            Observed Bivariate Probability distribution p
            Approximated probability distribution q
    Output: kullback-leibler divergence 
    '''
    # Get rid off zero values
    _p = p[(p > 0.) & (q>0.)]
    _q = q[(p>0.) & (q>0.)]
    delta_x = abs(x[1]-x[0])
    delta_y = abs(y[1]-y[0])

    return delta_x * np.sum(delta_y*np.sum(_p * np.log(_p/_q),axis=-1),axis=-1)


def fit_pdf(model_pdf, x, pdf,initial_guess):
    ''' 
    Inputs: model_pdf, function you want to fit, 
            x, array of random variable,
            pdf, array containing the observed values for the pdf
            initial_guess, first guess for the model pdf parameters
            
    Outputs: Best fit parameters of the model pdf
    '''
    
    popt, pcov = opt.curve_fit(model_pdf, x, pdf, p0=initial_guess)
    
    return popt, pcov

def fit_pdf_r(measured_pdf,r, vr, vt, log=True):

    v = np.array(np.meshgrid(vr, vt)).T.reshape(-1,2)
    rbins = np.arange(0,np.max(r),abs(r[1]-r[0]),dtype=int)
    popt = np.zeros((len(rbins),5))

    for rbin, rbin_value in enumerate(rbins):

        initial_guess = np.ones(5)

        if log==True:
            mask = np.where(measured_pdf[rbin].reshape(-1) != 0.)[0]

            reshaped_pdf = measured_pdf[rbin].reshape(-1)

            popt[rbin,:], pcov = fit_pdf(st.log_skewt, v[mask],
                    np.log10(reshaped_pdf[mask]), initial_guess)

        else:

            reshaped_pdf = measured_pdf[rbin].reshape(-1)

            popt[rbin,:], pcov = fit_pdf(st.skewt, v,
                                       reshaped_pdf, initial_guess)
    return popt


def Pearson_likelihood(params, x, measured_pdf):
    '''

    Computes minus log likelihood for a Pearson distriution (least square)
    Args: x, random variable array,
        measured_pdf, array containing the observed values for the pdf
        params, parameters that define a Pearson distribution (note the normalization constant is not included)
    Outputs: sum of squared difference between fit and measured pdf

    '''

    nu, a, lamda, m = params
    k = pearson.normalization(params)
    if((abs(nu) > 1000.) or (abs(a) > 1000.) or (abs(lamda) > 1000.) or (abs(m)>1000.)):
        return 1e8
    else:
        return np.sum((np.log(measured_pdf) - np.log(pearson.PearsonIV(x, *params, k)))**2)

def log_fit_pearson(x, measured_pdf):
    '''
    Minimizes the minus log likelihood to obtain the Pearson parameters of the best fit distribution.
    Args: x, random variable array,
        measured_pdf, array containing the observed values for the pdf

    Outputs: array of best fit parameters, together with normalization constant
    '''
    measured_moments = moments_from_distribution(x, measured_pdf)
    piv, p0 = pearson.Pearson_from_moments(x, measured_moments, 'analytic')
    p0 = p0[:-1] # Normalization constant is not fitted
    result = opt.minimize(Pearson_likelihood, p0, args=(x,measured_pdf))
    return result['x'], pearson.normalization(result['x'])


def moments_from_distribution(v,pdf):
    '''
    Computes first four central moments given a PDF
    Inputs: v, random variable
            pdf, probability distribution
            
    Outputs:  array of first four central moments
    '''
    wv = abs(v[1]-v[0])
    # check that the PDF is normalized
    #assert abs(1- wv * np.sum(pdf)) <= .15
    mean = wv * np.sum(v*pdf)
    m2 = wv * np.sum((v-mean)**2 * pdf)
    m3 = wv * np.sum((v-mean)**3 * pdf)
    m4 = wv * np.sum((v-mean)**4 * pdf)

    sb1 = m3 / m2**(3/2)
    b2 = m4/m2**2
    return np.array([mean,m2,sb1,b2])


