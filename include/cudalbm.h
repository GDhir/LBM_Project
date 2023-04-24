#ifndef cudalbm
#define cudalbm "cudalbm"

#include "utils.hpp"

#define BLOCKSIZE 256

__global__ void parlbm(double *fvals, double* fvalsprev, double *ex, double *ey, double g, double tau, int szf); 

#endif