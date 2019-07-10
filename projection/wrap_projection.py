
import numpy as np
import matplotlib.pyplot as plt
import ctypes
from numbers import Number

#********* LOAD C SHARED LIBRARY ********************#

libdir = '/home/c-cuesta/CentralStreamingModel/projection/'
libname = 'projection.so'


projection_lib = ctypes.CDLL(libdir + libname)#, ctypes.RTLD_GLOBAL)


projection_lib.projection.argtypes = [ 
        ctypes.c_double, ctypes.c_double, ctypes.c_double, 
        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1),
        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1),
        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1),
        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1),
        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1)]
		  

projection_lib.projection.restype = ctypes.c_double



projection_lib.projection_vlos.argtypes = [ 
        ctypes.c_double,  
        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1),
        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1),
		ctypes.c_int, 
        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1),
        ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1),
        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1),
        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1),
        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1),
        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1),
        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=1)]
		  

projection_lib.projection_vlos.restype = None


#************ RETURN PROJECTION AT ARRAY VLOS *************#

def projection(r, popt, r_perp, r_paral, vlos):

	r = r.astype(ctypes.c_double)
	n_r = len(r)

	w_r = popt[:,0].astype(ctypes.c_double)
	w_t = popt[:,1].astype(ctypes.c_double)
	vr_c = popt[:,2].astype(ctypes.c_double)
	alpha = popt[:,3].astype(ctypes.c_double)
	k = popt[:,4].astype(ctypes.c_double)

	if isinstance(vlos, Number):
		result = projection_lib.projection(r_perp, r_paral, vlos, 
							r, n_r, w_r, w_t, vr_c, alpha, k)
		return result

	else:
		vlos_size = len(vlos)

		projected_pdf = np.zeros(vlos_size)

		projection_lib.projection_vlos(r_perp, r_paral, vlos, vlos_size,
							r, n_r, w_r, w_t, vr_c, alpha, k,
							projected_pdf)
		return projected_pdf 


def skewt_los_pdf(r, popt):
	
	def skewt_pdf(r_perp, r_paral, v_los):

		return projection(r, popt, r_perp, r_paral, v_los)

	return skewt_pdf


