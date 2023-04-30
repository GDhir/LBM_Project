#include "cudalbm.h"

__global__ void parlbm_SM(double *fvals, double* fvalsprev, double *ex, double *ey, double g, double tau, int szf) 
{

  double f1 = 3.0;
  double f2 = 9.0 / 2.0;
  double f3 = 3.0 / 2.0;

  double rt0{0};
  double rt1{0};
  double rt2{0};

  double ueqxij{0};
  double ueqyij{0};
  double uxsq{0};
  double uysq{0};
  double uxuy5{0};
  double uxuy6{0};
  double uxuy7{0};
  double uxuy8{0};
  double usq{0};

  double feq0{0};
  double feq1{0};
  double feq2{0};
  double feq3{0};
  double feq4{0};
  double feq5{0};
  double feq6{0};
  double feq7{0};
  double feq8{0};

  double rho;
  double ux;
  double uy;
  double temp;

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;
  int sz{Nx*Ny};
  int i{0}, j{0}, xval{0}, yval{0}, tempi{0}, tempj{0}, tempidx{0}, fidx{0}, locali{0}, localj{0};

  __shared__ double fvals_shared[BLOCKSIZE*Q9];

  if( idx < sz ) {

    i = idx%Nx;
    j = idx/Nx;

    for (int k = 0; k < Q9; k++) {

      xval = ex[k];
      yval = ey[k];

      tempi = i - xval;
      tempj = j - yval;

      if (tempi == Nx) {
        tempi = 0;
      } else if (tempi == -1) {
        tempi = Nx - 1;
      }

      if (tempj == Ny) {
        tempj = 0;
      } else if (tempj == -1) {
        tempj = Ny - 1;
      }

      tempidx = tempj * Nx + tempi;

      fvals_shared[ tid + k*BLOCKSIZE ] = fvalsprev[ tempidx + k*sz ];
    }

    rho = 0;
    ux = 0;
    uy = 0;

    for (int k = 0; k < Q9; k++) {

      fidx = tid + k*BLOCKSIZE;

      rho += fvals_shared[ fidx ];
      ux += fvals_shared[ fidx ] * ex[k];
      uy += fvals_shared[ fidx ] * ey[k];
    }

    ux /= rho;
    uy /= rho;

    if (j > 0 && j < Ny - 1) {

      rt0 = (4.0 / 9.0) * rho;
      rt1 = (1.0 / 9.0) * rho;
      rt2 = (1.0 / 36.0) * rho;

      ueqxij = ux + tau * g;
      ueqyij = uy;

      uxsq = ueqxij * ueqxij;
      uysq = ueqyij * ueqyij;
      uxuy5 = ueqxij + ueqyij;
      uxuy6 = -ueqxij + ueqyij;
      uxuy7 = -ueqxij - ueqyij;
      uxuy8 = ueqxij - ueqyij;
      usq = uxsq + uysq;

      feq0 = rt0 * (1 - f3 * usq);
      feq1 = rt1 * (1 + f1 * ueqxij + f2 * uxsq - f3 * usq);
      feq2 = rt1 * (1 + f1 * ueqyij + f2 * uysq - f3 * usq);
      feq3 = rt1 * (1 - f1 * ueqxij + f2 * uxsq - f3 * usq);
      feq4 = rt1 * (1 - f1 * ueqyij + f2 * uysq - f3 * usq);
      feq5 = rt2 * (1 + f1 * uxuy5 + f2 * uxuy5 * uxuy5 - f3 * usq);
      feq6 = rt2 * (1 + f1 * uxuy6 + f2 * uxuy6 * uxuy6 - f3 * usq);
      feq7 = rt2 * (1 + f1 * uxuy7 + f2 * uxuy7 * uxuy7 - f3 * usq);
      feq8 = rt2 * (1 + f1 * uxuy8 + f2 * uxuy8 * uxuy8 - f3 * usq);

      fvals[ idx + 0*sz ] = fvals_shared[ tid + 0*BLOCKSIZE ] - (fvals_shared[ tid + 0*BLOCKSIZE ] - feq0) / tau;
      fvals[ idx + 1*sz ] = fvals_shared[ tid + 1*BLOCKSIZE ] - (fvals_shared[ tid + 1*BLOCKSIZE ] - feq1) / tau;
      fvals[ idx + 2*sz ] = fvals_shared[ tid + 2*BLOCKSIZE ] - (fvals_shared[ tid + 2*BLOCKSIZE ] - feq2) / tau;
      fvals[ idx + 3*sz ] = fvals_shared[ tid + 3*BLOCKSIZE ] - (fvals_shared[ tid + 3*BLOCKSIZE ] - feq3) / tau;
      fvals[ idx + 4*sz ] = fvals_shared[ tid + 4*BLOCKSIZE ] - (fvals_shared[ tid + 4*BLOCKSIZE ] - feq4) / tau;
      fvals[ idx + 5*sz ] = fvals_shared[ tid + 5*BLOCKSIZE ] - (fvals_shared[ tid + 5*BLOCKSIZE ] - feq5) / tau;
      fvals[ idx + 6*sz ] = fvals_shared[ tid + 6*BLOCKSIZE ] - (fvals_shared[ tid + 6*BLOCKSIZE ] - feq6) / tau;
      fvals[ idx + 7*sz ] = fvals_shared[ tid + 7*BLOCKSIZE ] - (fvals_shared[ tid + 7*BLOCKSIZE ] - feq7) / tau;
      fvals[ idx + 8*sz ] = fvals_shared[ tid + 8*BLOCKSIZE ] - (fvals_shared[ tid + 8*BLOCKSIZE ] - feq8) / tau;

    } else {

      // temp = fvals[idx + 2*sz];
      // fvals[idx + 2*sz] = fvals[idx + 4*sz];
      // fvals[idx + 4*sz] = temp;

      // temp = fvals[idx + 7*sz];
      // fvals[idx + 7*sz] = fvals[idx + 5*sz];
      // fvals[idx + 5*sz] = temp;

      // temp = fvals[idx + 8*sz];
      // fvals[idx + 8*sz] = fvals[idx + 6*sz];
      // fvals[idx + 6*sz] = temp;

      fvals[idx + 2*sz] = fvals_shared[tid + 4*BLOCKSIZE];
      fvals[idx + 4*sz] = fvals_shared[tid + 2*BLOCKSIZE];
      fvals[idx + 7*sz] = fvals_shared[tid + 5*BLOCKSIZE];
      fvals[idx + 5*sz] = fvals_shared[tid + 7*BLOCKSIZE];
      fvals[idx + 8*sz] = fvals_shared[tid + 6*BLOCKSIZE];
      fvals[idx + 6*sz] = fvals_shared[tid + 8*BLOCKSIZE];

    }
  }
}