import numpy as np
from scipy.special import gamma
from scipy.stats import t

def skewt_pdf(v, w, v_c, alpha, nu):
    
    arg = alpha/w * (v - v_c) * ((nu + 1)/(((v - v_c)/w)**2 + nu))**0.5
    
    return 2/w * t.pdf((v-v_c)/w, scale = 1, df = nu) * t.cdf(arg, df = nu+1, scale = 1)
