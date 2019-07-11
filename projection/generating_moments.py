import numpy as np
from scipy.integrate import simps
from scipy.interpolate import interp1d

class Expectations:
    
    def __init__(self, r, v_r, v_t, jointpdf):
        self.r = r
        self.v_r = v_r
        self.v_t = v_t
        self.jointpdf = jointpdf
        
    def moment(self, r_power, t_power):
        
        m = simps(self.v_t**t_power * \
                     simps(self.v_r **r_power * self.jointpdf , self.v_r, axis=-1) ,
                     self.v_t, axis=-1)
        
        return interp1d(self.r, m, fill_value = (m[0], m[-1]), bounds_error = False)
    
    def central_moment(self, r_power, t_power):

        mean_r = simps(simps(self.v_r * self.jointpdf , self.v_r, axis=-1) , self.v_t, axis=-1)
        
        cm =  simps(self.v_t**t_power *\
                     simps((self.v_r - mean_r[:, np.newaxis, np.newaxis]) **r_power * self.jointpdf ,
                            self.v_r, axis=-1) , self.v_t, axis=-1)
        return interp1d(self.r, cm, fill_value = (cm[0], cm[-1]), bounds_error  = False)

    
    def covariance(self, r_power, t_power):
        
        expected_t = self.moment(0, t_power)(self.r)
        expected_r = self.moment(r_power, 0)(self.r)
        
        cov = simps((self.v_t**t_power - expected_t[:, np.newaxis]) *\
                     simps((self.v_r - expected_r[:, np.newaxis, np.newaxis]) **r_power * self.jointpdf ,
                            self.v_r, axis=-1) , self.v_t, axis=-1)
        
        
        return interp1d(self.r, cov, bounds_error = False, fill_value = (cov[0], cov[-1]))


def project(ex, r_perp, r_parallel):

	m_10 = ex.moment(1,0)
	c_20 = ex.central_moment(2, 0)
	c_02 = ex.central_moment(0, 2)
	c_30 = ex.central_moment(3, 0)
	c_12 = ex.central_moment(1,2)
	c_22 = ex.central_moment(2, 2)
	c_40 = ex.central_moment(4,0)
	c_04 = ex.central_moment(0,4)
	
	moments_projected = np.zeros((len(r_perp), len(r_parallel), 4))

	for i, rper in enumerate(r_perp):
		for j, rpar in enumerate(r_parallel):

			r_ = np.sqrt(rper**2 + rpar**2)
			mu = rpar/r_


			moments_projected[i,j, 0] = m_10(r_) * mu
			
			moments_projected[i,j, 1] = c_20(r_) * mu**2  \
									+ c_02(r_) * (1 - mu**2)
			
			# Not assuming independence
			moments_projected[i,j, 2] =  c_30(r_)  * mu**3  \
									+ 3 * c_12(r_) * mu * (1 - mu**2) 
			
			
			moments_projected[i,j,3] = 6*c_22(r_)  * mu**2 * (1 - mu**2) + \
									  c_40(r_) * mu**4 +  c_04(r_) * (1-mu**2)**2
		

	return moments_projected
				
def project_independent(ex, r_perp, r_parallel):

	m_10 = ex.moment(1,0)
	c_20 = ex.central_moment(2, 0)
	c_02 = ex.central_moment(0, 2)
	c_30 = ex.central_moment(3, 0)
	c_40 = ex.central_moment(4,0)
	c_04 = ex.central_moment(0,4)
	
	moments_projected = np.zeros((len(r_perp), len(r_parallel), 4))

	for i, rper in enumerate(r_perp):
		for j, rpar in enumerate(r_parallel):

			r_ = np.sqrt(rper**2 + rpar**2)
			mu = rpar/r_


			moments_projected[i,j, 0] = m_10(r_) * mu
			
			moments_projected[i,j, 1] = c_20(r_) * mu**2  \
									+ c_02(r_) * (1 - mu**2)
			
			# Not assuming independence
			moments_projected[i,j, 2] =  c_30(r_)  * mu**3  #\
									#+ 3 * m_10(r_) * c_02(r_) * mu * (1 - mu**2) 
			
			
			moments_projected[i,j,3] = c_40(r_) * mu**4 +  c_04(r_) * (1-mu**2)**2 \
			 #+ 6*c_20(r_) * c_02(r_) * mu**2 * (1 - mu**2)		

	return moments_projected
				

