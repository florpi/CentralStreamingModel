from CentralStreamingModel.utils.read_probabilities import VD, VD_los
from CentralStreamingModel.projection import generating_moments
import numpy as np
import os, glob, pickle

class mean_error:

	def __init__(self, mean, std):

		self.mean = mean
		self.std = std


class Read_Mean:
	
	def __init__(self, n_boxes, boxsize, snapshot = 0, tracer = 'halos', mass = False):

		boxes = range(1, n_boxes + 1)
		boxsize = int(boxsize)
		self.snapshot = snapshot
		self.tracer = tracer

		self.data_dir = '/cosma6/data/dp004/dc-cues1/simulations/RSD/tpcfs/'


		self.rt = []
		self.los = []

		for i, box in enumerate(boxes):

			if mass:
				self.rt.append(VD(tracer, box, boxsize, snapshot, extra = mass))
				self.los.append(VD_los(tracer, box, boxsize, snapshot, extra = mass))

			else:
				self.rt.append(VD(tracer, box, boxsize, snapshot))
				self.los.append(VD_los(tracer, box, boxsize, snapshot))


		self.r = self.rt[0].r

		max_v = 40
		threshold = (self.rt[0].v.r > -max_v) & (self.rt[0].v.r < max_v)
		self.v_r = self.rt[0].v.r[threshold]
		self.v_t = self.rt[0].v.t[threshold]

		self.jointpdf_rt, _ = self.compute_mean_error(self.rt, 'jointpdf')

		self.jointpdf_rt = self.jointpdf_rt[:, threshold, :]
		self.jointpdf_rt = self.jointpdf_rt[:, :, threshold]

		self.r_perp = self.los[0].r.t
		self.r_parallel = self.los[0].r.r
		self.v_los = self.los[0].v
		self.jointpdf_los, self.jointpdf_los_error = self.compute_mean_error(self.los, 'jointpdf')

		self.tpcf_dict = self.read_real_tpcf(n_boxes, mass)

		self.read_redshift_tpcf(n_boxes, mass)


	def read_real_tpcf(self, n_boxes, mass): 

		list_dictionaries = []
		for i, box in enumerate(range(1, n_boxes + 1)):
			if mass:
				real_tpcf = self.data_dir + f"real/{self.tracer}_b{box}_{mass}.pickle"
			else:
				real_tpcf = self.data_dir + f"real/{self.tracer}_b{box}.pickle"

			with open(real_tpcf, "rb") as input_file:
					list_dictionaries.append(pickle.load(input_file))
					tpcfs = [dictionary['tpcf'] for dictionary in list_dictionaries]
					mean_tpcf = np.mean(tpcfs, axis=0)
					std_tpcf = np.std(tpcfs, axis=0)

					mean_tpcf_dict =  {'r': list_dictionaries[0]['r'], 'tpcf': mean_tpcf} 

		return mean_tpcf_dict

	def read_redshift_tpcf(self, n_boxes, mass):

		list_dictionaries = []
		for i, box in enumerate(range(1, n_boxes + 1)):
			if mass:
				redshift_tpcf = self.data_dir + f"redshift/{self.tracer}_b{box}_{mass}.pickle"
			else:
				redshift_tpcf = self.data_dir + f"redshift/{self.tracer}_b{box}.pickle"


			with open(redshift_tpcf, "rb") as input_file:
				list_dictionaries.append(pickle.load(input_file))
		
		list_attributes = ['pi_sigma', 'mono', 'quad', 'hexa']

		setattr(self, 's_c', list_dictionaries[0]['r'])
		setattr(self, 'mu_c', list_dictionaries[0]['mu_bins'])

		for attribute in list_attributes:

			mean, std = self.compute_mean_error(list_dictionaries, attribute)

			setattr(self, attribute, mean_error(mean, std))

		

	def compute_mean_error(self, per_box_list, attribute):

		if isinstance(per_box_list[0], dict):
			mean = np.mean([box.get(attribute) for box in per_box_list], axis = 0 )
			std = np.std([box.get(attribute) for box in per_box_list], axis = 0 )

		else:
			mean = np.mean([getattr(box, attribute) for box in per_box_list], axis = 0 )
			std = np.std([getattr(box, attribute) for box in per_box_list], axis = 0 )

		return mean, std


