import numpy as np
from scipy.integrate import simps
from scipy.special import gamma
from scipy.stats import t, norm
from scipy.optimize import curve_fit

def skewt(v, v_c, w, alpha, nu):
    
    arg = alpha/w * (v - v_c) * ((nu + 1)/(((v - v_c)/w)**2 + nu))**0.5
    
    return 2/w * t.pdf((v-v_c)/w, scale = 1, df = nu) * t.cdf(arg, df = nu+1, scale = 1)

def truncated_skewt(v,v_los, v_c, w, alpha, nu):
    
    pdf = skewt(v, v_c, w, alpha, nu)
    if hasattr(w, "__len__"):
        norm = []
        for w_value in w:
            norm.append(simps(skewt(v_los, v_c,w_value, alpha, nu), v_los))    
    else:
        norm = simps(skewt(v_los, v_c,w, alpha, nu), v_los)
    return pdf/norm

def gaussian_from_moments(r, v, vr_c, w, skewness, kappa):

	st_pdf = np.zeros((len(r), len(v)))
	gauss_pdf = np.zeros_like(st_pdf)
	
	mean = np.zeros((len(r)))
	std = np.zeros_like(mean)

	for i, r_value in enumerate(r):

		st_pdf[i] = truncated_skewt(v,v,  vr_c, w(r_value), skewness, kappa)
		mean[i] = simps(st_pdf[i,:] * v, v)
		std[i] = np.sqrt(simps(st_pdf[i] * (v - mean[i])**2, v))

		gauss_pdf[i] = norm.pdf(v, loc = mean[i], scale = std[i])

	return mean, std, st_pdf, gauss_pdf

def norm_pdf(v, loc, scale):

	return norm.pdf(v, loc = loc, scale = scale)


def gaussian_best_fit(r, v, vr_c, w, skewness, kappa):


	st_pdf = np.zeros((len(r), len(v)))
	gauss_pdf = np.zeros_like(st_pdf)
	
	mean = np.zeros((len(r)))
	std = np.zeros_like(mean)

	for i, r_value in enumerate(r):

		st_pdf[i] = truncated_skewt(v,v, vr_c, w(r_value), skewness, kappa)

		popt, pcov = curve_fit(norm_pdf, v, st_pdf[i, :])

		mean[i], std[i] = popt

		gauss_pdf[i] = norm.pdf(v, loc = mean[i], scale = std[i])

	return mean, std, st_pdf, gauss_pdf

def params2moments(r, v_c, w, alpha, kappa):

	b = (kappa/np.pi)**0.5 * gamma((kappa-1)/2.)/gamma(kappa/2.)
	delta = alpha/np.sqrt(1 + alpha**2)
	factor = kappa/(kappa - 2) - delta**2*b**2

	mean = v_c + (w(r)) * delta * b
	std = np.sqrt(factor) * w(r)

	return mean, std


def gsm_params( mean_interp, std_interp, truncate = False):

	def gsm(r, v, truncate = False):
	    return norm.pdf(v, loc = mean_interp(abs(r)), scale = std_interp(abs(r)))

	return gsm

def stsm_params(v_los, vr_c, w, skewness, kappa, truncate):

	def stsm(r,v, truncate):

		pdf_contribution = np.zeros_like(v)

		if truncate:
			threshold = (v > np.min(v_los)) & (v < np.max(v_los))
			
			pdf_contribution[threshold] = truncated_skewt( v, v_los, vr_c,w(abs(r)), skewness,
														  kappa)[threshold]

		else:
			pdf_contribution = skewt(v, vr_c,w(abs(r)), skewness, kappa)

		return pdf_contribution

	return stsm

def integrand_minus(r_parallel, pdf, s):

    v = (s - r_parallel) * np.sign(r_parallel)

    return pdf(r_parallel, v, truncate=True)

def integrate(integrand,r_parallel):
    left = r_parallel < 0.
    right = r_parallel > 0.
    
    return simps(integrand[left], r_parallel[left]) + simps(integrand[right], r_parallel[right])
