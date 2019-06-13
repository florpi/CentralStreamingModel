from scipy.interpolate import interp1d
import numpy as np
from joblib import Parallel, delayed
import itertools
from scipy.integrate import simps, quadrature, quad

# https://arxiv.org/abs/1710.09379 (Eq 22)


def compute_integrand(s_perp, s_paral, twopcf_dict, projected_pairwise_pdf, s_paral_full, vlos_bins, n_cores = 1):

    s_perp_c = 0.5 * (s_perp[1:] + s_perp[:-1])
    s_paral_c = 0.5 * (s_paral[1:] + s_paral[:-1])

    binwidth = s_paral_full[1] - s_paral_full[0]
    initial = -69. - binwidth/2.
    final = 69. + binwidth/2.
    y = np.arange(initial, final, binwidth)

    # Refine integrand near zero (where correlation function dominates)
    epsilon = 0.0001
    y = np.append(y, epsilon)
    y = np.append(y, -epsilon)
    y = np.sort(y)

    # Interpolate tpcf
    interp_twopcf = interp1d(twopcf_dict['r'], twopcf_dict['tpcf'], kind = 'cubic', 
							fill_value = (-1., twopcf_dict['tpcf'][-1]), bounds_error=False)

    def define_integrand(args):

        s_perp_bin, s_paral_bin = args

        integrand = np.zeros(len(y))
        pdf_contribution = np.zeros(len(y))

        for k, y_value in enumerate(y):

            r = np.sqrt(s_perp_c[s_perp_bin]**2 + y_value**2)

            y_bin = np.digitize(np.abs(y_value), s_paral_full - binwidth/2) - 1 

            selected_pairwise_pdf = projected_pairwise_pdf[s_perp_bin, y_bin, :]
            interp_pairwise_pdf = interp1d(vlos_bins, selected_pairwise_pdf, kind='linear')

            vlos = (s_paral_c[s_paral_bin] - y_value) * np.sign(y_value)


            if ( (vlos > np.min(vlos_bins)) and (vlos < np.max(vlos_bins) ) ):
                p = interp_pairwise_pdf(vlos)
            else:
                p = 0.

            if (r < twopcf_dict['r'][-1]):
                integrand[k] = (1. + interp_twopcf(r)) * p

            else:
                integrand[k] = p
            pdf_contribution[k] = p

        return [pdf_contribution, integrand]


    # Parallelise the integrand computation

    dim1 = s_perp_c.shape[0]
    dim2 = s_paral_c.shape[0]

    arg_instances = list((i,j) for i,j in itertools.product(range(dim1), range(dim2)))


    combined_result = Parallel(n_jobs = n_cores, verbose = 0 ) (map(delayed(define_integrand), arg_instances))

    pdf_contribution = [i[0] for i in combined_result]
    result_integrand = [i[1] for i in combined_result]

    pdf_contribution = np.asarray(pdf_contribution)
    pdf_contribution= pdf_contribution.reshape((s_perp_c.shape[0], s_paral_c.shape[0], y.shape[0]))

    result_integrand = np.asarray(result_integrand)
    result_integrand = result_integrand.reshape((s_perp_c.shape[0], s_paral_c.shape[0], y.shape[0]))
    
    return y, result_integrand, pdf_contribution

def compute_integrand_continuous(s_perp, s_paral, twopcf_dict, function_los_pdf, truncate, n_cores = 1):

    s_perp_c = 0.5 * (s_perp[1:] + s_perp[:-1])
    s_paral_c = 0.5 * (s_paral[1:] + s_paral[:-1])

    y = np.arange(-69.5, 69.5, 1.)

    # Refine integrand near zero (where correlation function dominates)
    epsilon = 0.0001
    y = np.append(y, epsilon)
    y = np.append(y, -epsilon)
    y = np.sort(y)

    # Interpolate tpcf
    interp_twopcf = interp1d(twopcf_dict['r'], twopcf_dict['tpcf'], kind = 'cubic',
							fill_value = (-1., twopcf_dict['tpcf'][-1]), bounds_error=False)

    def skewt_quad(vlos, rperp, rpar):
        return function_los_pdf(rperp, rpar,vlos)

    def define_integrand(args):

        s_perp_bin, s_paral_bin = args

        # Refine integrand peak
        integrand = np.zeros(len(y))
        pdf_contribution = np.zeros(len(y))

        for k, y_value in enumerate(y):

            r = np.sqrt(s_perp_c[s_perp_bin]**2 + y_value**2)

            vlos = (s_paral_c[s_paral_bin] - y_value) * np.sign(y_value)

            if truncate:
                if abs(vlos) < truncate :
                    p = function_los_pdf(s_perp_c[s_perp_bin], abs(y_value), vlos)
                else:
                    p = 0.
            else:
                p = function_los_pdf(s_perp_c[s_perp_bin], abs(y_value), vlos)


            if (r < twopcf_dict['r'][-1]):
                integrand[k] = (1. + interp_twopcf(r)) * p
                pdf_contribution[k] = p

            else:
                integrand[k] = p
                pdf_contribution[k] = p

        return [pdf_contribution, integrand]


    # Parallelise the integrand computation

    dim1 = s_perp_c.shape[0]
    dim2 = s_paral_c.shape[0]

    arg_instances = list((i,j) for i,j in itertools.product(range(dim1), range(dim2)))

    combined_result = Parallel(n_jobs = n_cores, verbose = 0 ) (map(delayed(define_integrand), arg_instances))

    pdf_contribution = [i[0] for i in combined_result]
    result_integrand = [i[1] for i in combined_result]

    pdf_contribution = np.asarray(pdf_contribution)
    pdf_contribution= pdf_contribution.reshape((s_perp_c.shape[0], s_paral_c.shape[0], y.shape[0]))

    result_integrand = np.asarray(result_integrand)
    result_integrand = result_integrand.reshape((s_perp_c.shape[0], s_paral_c.shape[0], y.shape[0]))
    
    return y, result_integrand, pdf_contribution



def integrate(r_parallel, integrand):

    # due to discontiniuty at 0, need to integrate by parts

    left = r_parallel < 0.
    right = r_parallel > 0.

    int_left = simps(integrand[...,left], r_parallel[...,left], axis = -1)

    int_right = simps(integrand[...,right], r_parallel[...,right], axis = -1)

    pi_sigma = int_left + int_right - 1.

    return pi_sigma

def integrate_full(r_parallel, integrand):

    int_ = simps(integrand, r_parallel, axis=-1)

    pi_sigma = int_ - 1.

    return pi_sigma

def integrate_simpson(s_perp, s_paral, twopcf_dict, function_los_pdf, truncate, n_cores = 1):

    s_perp_c = 0.5 * (s_perp[1:] + s_perp[:-1])
    s_paral_c = 0.5 * (s_paral[1:] + s_paral[:-1])

    y = np.arange(-69.5, 69.5, 1.)

    # Refine integrand near zero (where correlation function dominates)
    epsilon = 0.0001
    y = np.append(y, epsilon)
    y = np.append(y, -epsilon)
    y = np.sort(y)

    # Interpolate tpcf
    interp_twopcf = interp1d(twopcf_dict['r'], twopcf_dict['tpcf'], kind = 'cubic',
							fill_value = (-1., twopcf_dict['tpcf'][-1]), bounds_error=False)

    def define_integrand(args):

        s_perp_bin, s_paral_bin = args

        integrand = np.zeros(len(y))

        for k, y_value in enumerate(y):

            r = np.sqrt(s_perp_c[s_perp_bin]**2 + y_value**2)


            vlos = (s_paral_c[s_paral_bin] - y_value) * np.sign(y_value)

            if truncate:
                if abs(vlos) < truncate :
                    p = function_los_pdf(s_perp_c[s_perp_bin], abs(y_value), vlos)
                else:
                    p = 0.
            else:
                p = function_los_pdf(s_perp_c[s_perp_bin], abs(y_value), vlos)

            if (r < twopcf_dict['r'][-1]):
                integrand[k] = (1. + interp_twopcf(r)) * p

            else:
                integrand[k] = p

        return integrand


    # Parallelise the integrand computation


    dim1 = s_perp_c.shape[0]
    dim2 = s_paral_c.shape[0]

    arg_instances = list((i,j) for i,j in itertools.product(range(dim1), range(dim2)))

    result_integrand = Parallel(n_jobs = n_cores, verbose = 0 ) (map(delayed(define_integrand), arg_instances))

    result_integrand = np.asarray(result_integrand)
    result_integrand = result_integrand.reshape((s_perp_c.shape[0], s_paral_c.shape[0], y.shape[0]))

    return y, result_integrand



def integrate_quad(s_perp, s_paral, twopcf_dict, function_los_pdf, truncate, n_cores = 1):

    s_perp_c = 0.5 * (s_perp[1:] + s_perp[:-1])
    s_paral_c = 0.5 * (s_paral[1:] + s_paral[:-1])


    # Interpolate tpcf
    interp_twopcf = interp1d(twopcf_dict['r'], twopcf_dict['tpcf'], kind = 'cubic',
							fill_value = (-1., twopcf_dict['tpcf'][-1]), bounds_error=False)

    def integrate(args):
        
        s_perp_bin, s_paral_bin = args


        def define_integrand(y_value):

            r = np.sqrt(s_perp_c[s_perp_bin]**2 + y_value**2)


            vlos = (s_paral_c[s_paral_bin] - y_value) * np.sign(y_value)

            '''
            p = np.zeros_like(vlos)
            if truncate:
                threshold = (vlos > -truncate) & (vlos < truncate)

                p[threshold] = function_los_pdf(s_perp_c[s_perp_bin], abs(y_value[threshold]), vlos[threshold])

            else:
            '''
            p = function_los_pdf(s_perp_c[s_perp_bin], abs(y_value), vlos)

            return (1. + interp_twopcf(r)) * p



        # Avoid discontinuity due to sign function
        result_left = quadrature(define_integrand, -70., 0.)
        result_right = quadrature(define_integrand, 0., 70.)

        result = result_left[0] + result_right[0]

        return  result - 1.


    # Parallelise the integrand computation


    dim1 = s_perp_c.shape[0]
    dim2 = s_paral_c.shape[0]

    arg_instances = list((i,j) for i,j in itertools.product(range(dim1), range(dim2)))

    result_integrand = Parallel(n_jobs = n_cores, verbose = 0 ) (map(delayed(integrate), arg_instances))

    result_integrand = np.asarray(result_integrand)
    result_integrand = result_integrand.reshape((s_perp_c.shape[0], s_paral_c.shape[0]))

    return result_integrand


def integrate_quad_normalised(s_perp, s_paral, twopcf_dict, function_los_pdf, truncate, n_cores = 1):

    s_perp_c = 0.5 * (s_perp[1:] + s_perp[:-1])
    s_paral_c = 0.5 * (s_paral[1:] + s_paral[:-1])


    # Interpolate tpcf
    interp_twopcf = interp1d(twopcf_dict['r'], twopcf_dict['tpcf'], kind = 'cubic',
							fill_value = (-1., twopcf_dict['tpcf'][-1]), bounds_error=False)

    def skewt_quad(vlos, rperp, rpar):
        return function_los_pdf(rperp, rpar,vlos)


    def integrate(args):
        
        s_perp_bin, s_paral_bin = args


        def define_integrand(y_value):

            r = np.sqrt(s_perp_c[s_perp_bin]**2 + y_value**2)


            vlos = (s_paral_c[s_paral_bin] - y_value) * np.sign(y_value)

            p = function_los_pdf(s_perp_c[s_perp_bin], abs(y_value), vlos)/quad(skewt_quad, -truncate, truncate,
                                                        args = (s_perp_c[s_perp_bin], abs(y_value)))[0]


            return (1. + interp_twopcf(r)) * p



        # Avoid discontinuity due to sign function
        result_left = quad(define_integrand, -70., 0.)
        result_right = quad(define_integrand, 0., 70.)

        result = result_left[0] + result_right[0]

        return  result - 1.


    # Parallelise the integrand computation


    dim1 = s_perp_c.shape[0]
    dim2 = s_paral_c.shape[0]

    arg_instances = list((i,j) for i,j in itertools.product(range(dim1), range(dim2)))

    result_integrand = Parallel(n_jobs = n_cores, verbose = 0 ) (map(delayed(integrate), arg_instances))

    result_integrand = np.asarray(result_integrand)
    result_integrand = result_integrand.reshape((s_perp_c.shape[0], s_paral_c.shape[0]))

    return result_integrand




