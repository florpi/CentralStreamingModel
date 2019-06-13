import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt 
import h5py
#sns.set()
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec


class components:
	def __init__(self,tangential,radial):
		self.t = tangential
		self.r = radial



class VD: 
	def __init__(self, box, boxsize, snapshot):
		self.box = box
		self.boxsize = boxsize
		self.snapshot = snapshot
		self.read_file()


	def read_file(self):
		file_name = f'/raid/c-cuesta/results/run1{self.box:02d}_b{self.boxsize}_s{self.snapshot:03d}_mass.txt'

		self.r, self.m1, self.m2, self.vt, self.vr,  self.Npairs = np.loadtxt(file_name, unpack=True)

		self.wvt =( np.unique(self.vt)[1] - np.unique(self.vt)[0]) 
		self.wvr =( np.unique(self.vr)[1] - np.unique(self.vr)[0]) 
		self.wr = self.r[0] * 2 # Mpc (bin width)
		self.r = np.unique(self.r)
		self.m1 = np.unique(self.m1)
		self.m2 = np.unique(self.m2)

		self.v = components(np.unique(self.vt), np.unique(self.vr))
		self.npairs = np.reshape(self.Npairs, (self.r.shape[0], self.m1.shape[0], self.m2.shape[0],
				self.v.t.shape[0], self.v.r.shape[0]))
		norm =	np.sum(self.wvt * np.sum(self.wvr * self.npairs,axis=-1),axis=-1)
		self.jointpdf = self.npairs / norm[:,:,:,np.newaxis,np.newaxis]

		tmarginal = self.wvr * np.sum(self.jointpdf,axis=-1)
		rmarginal = self.wvt * np.sum(self.jointpdf,axis=-2)
		tnorm = self.wvt * np.sum(tmarginal,axis=-1)
		rnorm = self.wvr * np.sum(rmarginal,axis=-1)
		tmarginal = tmarginal/tnorm[:,:,:,np.newaxis]
		rmarginal = rmarginal/rnorm[:,:,:,np.newaxis]

		self.marginal = components(tmarginal,rmarginal)

		rmean, rstd, rskewness, rkurtosis  = self.statistics(self.v.r, self.marginal.r)
		tmean, tstd, tskewness, tkurtosis  = self.statistics(self.v.t, self.marginal.t)

		self.mean = components(tmean, rmean)
		self.std= components(tstd, rstd)
		self.skewness= components(tskewness, rskewness)
		self.kurtosis= components(tkurtosis, rkurtosis)

	def statistics(self, v, marginal):

		wv = abs(v[1] - v[0])
		mean = wv * np.sum(marginal * v, axis=-1) 
		mean_expand = mean[...,np.newaxis]
		v_expand = np.tile(v, (marginal.shape[0], marginal.shape[1],1))
		std = np.sqrt(wv * np.sum((v - mean_expand)**2 * marginal, axis=-1))
		skewness =	wv * np.sum((v - mean_expand)**3 * marginal, axis=-1)/std**3
		kurtosis =	wv * np.sum((v - mean_expand)**4 * marginal, axis=-1)/std**4

		return mean, std, skewness, kurtosis


