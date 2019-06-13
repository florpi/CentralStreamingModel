import numpy as np
from scipy.special import gamma
from scipy.stats import t

#https://arxiv.org/pdf/0911.2342.pdf equation 26

def get_Q(v, vr_c, w_r, w_t):
    return v[:,0]*v[:,0]/w_t + (v[:,1] - vr_c)**2/w_r 


def t2(Q, dof, w_r, w_t ):

    determinant = w_r * w_t 

    prefactor = gamma((dof+2)/2)/np.pi/dof/gamma(dof/2)/determinant**0.5

    postfactor = (1. + Q/dof)**(-(dof + 2.)/2.)

    return postfactor * prefactor


def t1_arg(v, alpha, w_r, w_t, vr_c, Q, dof):

    return alpha * ( (v[:,1] - vr_c) / np.sqrt(w_r)) * ((dof + 2)/(Q + dof)) ** 0.5 


def t1(v, alpha, w_r, w_t, vr_c, Q, dof):
    return t.cdf(t1_arg(v, alpha, w_r, w_t, vr_c, Q, dof), dof + 2)

def skewt(v, w_r, w_t, vr_c, alpha, dof):
    Q = get_Q(v, vr_c, w_r, w_t)

    return 2 * t2(Q, dof, w_r, w_t) * t1(v, alpha, w_r, w_t, vr_c, Q, dof)

def log_skewt(v, w_r, w_t,  vr_c, alpha, dof):
    Q = get_Q(v, vr_c, w_r, w_t)

    return np.log10(2 * t2(Q, dof, w_r, w_t) * t1(v, alpha, w_r, w_t, vr_c, Q, dof))

