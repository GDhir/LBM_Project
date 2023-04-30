#ifndef cudalbm
#define cudalbm "cudalbm"

#include "utils.hpp"

#define BLOCKSIZE 256

__global__ void parlbm_AOS(double *fvals, double* fvalsprev, double *ex, double *ey, double g, double tau, int szf); 
__global__ void parlbm_SOA(double *fvals, double* fvalsprev, double *ex, double *ey, double g, double tau, int szf); 
__global__ void parlbm_SM(double *fvals, double* fvalsprev, double *ex, double *ey, double g, double tau, int szf); 

#endif