import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt 
import h5py
#sns.set()
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec

def gauss(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

class components:
    def __init__(self,tangential,radial):
        self.t = tangential
        self.r = radial



class VD: 
    def __init__(self, tracer, box, boxsize, snapshot, extra = None):
        self.tracer = tracer
        self.box = box
        self.boxsize = boxsize
        self.snapshot = snapshot
        self.read_file(extra)
    def read_file(self, extra):

        if not extra:
            file_name = f'/cosma/home/dp004/dc-cues1/CentralPairwiseVels/results/{self.tracer}_b{self.box:1d}.txt'
        else:
            file_name = f'/cosma/home/dp004/dc-cues1/CentralPairwiseVels/results/{self.tracer}_b{self.box:1d}_{extra}.txt'

        self.br, self.bvt, self.bvr, self.r, self.vt, self.vr, self.wvr, self.Npairs = np.loadtxt(file_name, unpack=True)

        self.wvt =( np.unique(self.vt)[1] - np.unique(self.vt)[0]) 
        self.wvr =( np.unique(self.vr)[1] - np.unique(self.vr)[0]) 
        self.wr = self.r[0] * 2 # Mpc (bin width)
        self.r = np.unique(self.r)

        self.v = components(np.unique(self.vt), np.unique(self.vr))
        self.npairs = np.reshape(self.Npairs, (self.r.shape[0], self.v.t.shape[0], self.v.r.shape[0]))
        norm =  np.sum(self.wvt * np.sum(self.wvr * self.npairs,axis=-1),axis=-1)
        self.jointpdf = self.npairs / norm[:,np.newaxis,np.newaxis]

        tmarginal = self.wvr * np.sum(self.jointpdf,axis=-1)
        rmarginal = self.wvt * np.sum(self.jointpdf,axis=1)
        tnorm = self.wvt * np.sum(tmarginal,axis=-1)
        rnorm = self.wvr * np.sum(rmarginal,axis=-1)
        tmarginal = tmarginal/tnorm[:,np.newaxis]
        rmarginal = rmarginal/rnorm[:,np.newaxis]

        self.marginal = components(tmarginal,rmarginal)


        rmean, rstd, rskewness, rkurtosis  = self.statistics(self.v.r, self.marginal.r)
        tmean, tstd, tskewness, tkurtosis  = self.statistics(self.v.t, self.marginal.t)

        self.mean = components(tmean,rmean)
        self.std= components(tstd,rstd)
        self.skewness = components(tskewness,rskewness)
        self.kurtosis = components(tkurtosis,rkurtosis)

    def statistics(self, v, marginal):

        wv = abs(v[1] - v[0])
        mean = wv * np.sum(marginal * v, axis=-1) 
        mean_expand = mean[...,np.newaxis]
        v_expand = np.tile(v, (marginal.shape[0], marginal.shape[1],1))
        std = np.sqrt(wv * np.sum((v - mean_expand)**2 * marginal, axis=-1))
        skewness =  wv * np.sum((v - mean_expand)**3 * marginal, axis=-1)/std**3
        kurtosis =  wv * np.sum((v - mean_expand)**4 * marginal, axis=-1)/std**4

        return mean, std, skewness, kurtosis



class VD_los:
    def __init__(self, tracer, box, boxsize, snapshot, extra = None):
        self.box = box
        self.tracer = tracer
        self.boxsize = boxsize
        self.snapshot = snapshot
        self.read_file(extra)



    def read_file(self,extra):
        file_name = f'/cosma/home/dp004/dc-cues1/CentralPairwiseVels/results/{self.tracer}_b{self.box:1d}_los.txt'


        self.brperp, self.brparal, self.bv, self.rperp, self.rparal, self.v, self.wv, self.Npairs = np.loadtxt(file_name,
                                                                                                     unpack=True)
        self.wrperp = (np.unique(self.rperp)[1]  - np.unique(self.rperp)[0])
        self.wrparal = (np.unique(self.rparal)[1]  - np.unique(self.rparal)[0])
        self.wv= (np.unique(self.v)[1]  - np.unique(self.v)[0])

        self.r = components(np.unique(self.rperp), np.unique(self.rparal))

        self.v = np.unique(self.v)
        self.npairs = np.reshape(self.Npairs, (self.r.t.shape[0], self.r.r.shape[0],self.v.shape[0]))
        norm =  np.sum(self.wv * self.npairs, axis=-1)
        self.jointpdf = self.npairs /norm[:,:,np.newaxis]


