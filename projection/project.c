#include <stdio.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_cdf.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_errno.h>
#include <omp.h>



const int skewt_nparams = 5;
//const double limit_integral = 30.;
const double limit_integral = 20.;


/*********** DEFINITION OF THE BISKEWT DISTRIBUTION *********************/
/************ https://arxiv.org/abs/0911.2342 **************************/

double get_Q(double v_t, double v_r, double vr_c, double w_r, double w_t)
{
    return v_t*v_t/w_t + pow(v_r - vr_c,2.)/w_r;
}

double t2(double Q, double dof, double w_r, double w_t)
{
    double determinant = w_r * w_t;


    double prefactor =  gsl_sf_gamma((dof + 2.) / 2.)/M_PI/dof/gsl_sf_gamma(dof/2.)/pow(determinant,1./2.);

    double postfactor = pow(1. + Q/dof,-(dof + 2.)/2.);

    return postfactor * prefactor;
}

double t1_arg(double v_t, double v_r, double alpha, double w_r, double w_t, double vr_c, double Q, double dof)
{
    return alpha * ( (v_r - vr_c)/sqrt(w_r)) * pow((dof + 2)/(Q + dof), 0.5);

}

double t1(double v_t, double v_r, double alpha, double w_r, double w_t, double vr_c, double Q, double dof)
{
    return gsl_cdf_tdist_P(t1_arg(v_t, v_r, alpha, w_r, w_t, vr_c, Q, dof), dof + 2); 
}

double skewt(double v_t, double v_r, double w_r, double w_t, double vr_c, double alpha, double dof)
{

    double Q = get_Q(v_t, v_r, vr_c, w_r, w_t);

    return 2 * t2(Q, dof, w_r, w_t) * t1(v_t, v_r, alpha, w_r, w_t, vr_c, Q, dof);
}


/*********** PROJECTION INTEGRALS *********************/

double integrand_rparal(double v_t, double *params){
    double v_los, theta, vr_c, w_r, w_t, alpha, dof;
    v_los = params[0];
    theta = params[1];
    w_r = params[2];
    w_t = params[3];
    vr_c = params[4];
    alpha = params[5];
    dof = params[6];

    double v_r = (v_los - v_t * sin(theta))/cos(theta);

    return skewt(v_t, v_r, w_r, w_t, vr_c, alpha, dof)/cos(theta);
}


double integrand_rperp (double v_r, double *params){

    double v_los, theta, vr_c, w_r, w_t, alpha, dof;
    v_los = params[0];
    theta = params[1];
    w_r = params[2];
    w_t = params[3];
    vr_c = params[4];
    alpha = params[5];
    dof = params[6];


    double v_t = (v_los - v_r * cos(theta))/sin(theta);

    return skewt(v_t, v_r,  w_r, w_t, vr_c, alpha, dof)/sin(theta);
}

double interpolate(double *r, int n_r, double *parameter, double r_result)
{
    double result;
    gsl_interp_accel *acc = gsl_interp_accel_alloc ();
    gsl_spline *spline = gsl_spline_alloc (gsl_interp_cspline, n_r);

    gsl_spline_init(spline, r, parameter, n_r);

    result = gsl_spline_eval(spline, r_result, acc);

    gsl_spline_free (spline);
    gsl_interp_accel_free (acc);

    return result;

}


double projection(double r_perp, double r_paral, double vlos,
        double *r, int n_r, double *w_r, double *w_t, double *vr_c, double *alpha, double *k)
{
    double  theta, r_;
    size_t nevals;
	double error, result;

    r_ = sqrt(pow(r_paral,2.) + pow(r_perp,2.));
	//printf("r : %.2f\n", r_);
    theta = atan(r_perp / r_paral);

    double params[7];

    params[0] = vlos;    //= {vlos, theta, -0.5, 1.45, 1.23, -0.3, 0.5};
    params[1] = theta;
    params[2] = interpolate(r, n_r, w_r, r_); 
    params[3] = interpolate(r, n_r, w_t, r_); 
    params[4] = interpolate(r, n_r, vr_c, r_); 
    params[5] = interpolate(r, n_r, alpha, r_); 
    params[6] = interpolate(r, n_r, k, r_); 

    //gsl_integration_cquad_workspace * wcquad = gsl_integration_cquad_workspace_alloc (1000);
    gsl_integration_workspace *wcquad= gsl_integration_workspace_alloc (1000);



	gsl_function F;
	if(r_paral > r_perp){
		F.function = &integrand_rparal;
	}
	else
	{
		F.function = &integrand_rperp;
	}
	F.params = &params;


    //gsl_integration_cquad(&F, -limit_integral, limit_integral, 0, 1e-4, wcquad, &result, &error, &nevals);
    gsl_integration_qagi(&F,  0, 1e-7, 1000, wcquad, &result, &error);


	 //gsl_integration_cquad_workspace_free(wcquad);
	 gsl_integration_workspace_free(wcquad);
	return result;

}

void projection_vlos(double r_perp, double *r_paral, double *vlos, int vlos_size,
        double *r, int n_r, double *w_r, double *w_t, double *vr_c, double *alpha, double *k,
		double *projected_pdf)
{

	int i;

	for (i = 0; i < vlos_size ; ++i){

		projected_pdf[i] = projection(r_perp, r_paral[i], vlos[i], 
        	r, n_r, w_r, w_t, vr_c, alpha, k);	
	}


}
