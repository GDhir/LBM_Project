#define BLOCK_SIZE 128

__global__ void parlbm(double *fvals, double *ex, double *ey, double *ftemp, double *feq,
                       double g, double tau, int szf, int Niter, int Nx,
                       int Ny, int Q9) {

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

  double rho;
  double ux;
  double uy;

  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int i = idx%Nx;
  int j = idx/Nx;

  for (int k = 0; k < Q9; k++) {

    int xval = ex[k];
    int yval = ey[k];

    int tempi = i - xval;
    int tempj = j - yval;

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

    int ftempidx = tempj * Nx * Q9 + tempi * Q9 + k;

    ftemp[k] = fvals[ftempidx];
  }

  rho = 0;
  ux = 0;
  uy = 0;

  for (int k = 0; k < Q9; k++) {

    rho += ftemp[k];
    ux += ftemp[k] * ex[k];
    uy += ftemp[k] * ey[k];
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

    feq[0] = rt0 * (1 - f3 * usq);
    feq[1] = rt1 * (1 + f1 * ueqxij + f2 * uxsq - f3 * usq);
    feq[2] = rt1 * (1 + f1 * ueqyij + f2 * uysq - f3 * usq);
    feq[3] = rt1 * (1 - f1 * ueqxij + f2 * uxsq - f3 * usq);
    feq[4] = rt1 * (1 - f1 * ueqyij + f2 * uysq - f3 * usq);
    feq[5] = rt2 * (1 + f1 * uxuy5 + f2 * uxuy5 * uxuy5 - f3 * usq);
    feq[6] = rt2 * (1 + f1 * uxuy6 + f2 * uxuy6 * uxuy6 - f3 * usq);
    feq[7] = rt2 * (1 + f1 * uxuy7 + f2 * uxuy7 * uxuy7 - f3 * usq);
    feq[8] = rt2 * (1 + f1 * uxuy8 + f2 * uxuy8 * uxuy8 - f3 * usq);

    for (int k = 0; k < Q9; k++) {

      int fidx = j * Nx * Q9 + i * Q9 + k;

      fvals[fidx] = ftemp[k] - (ftemp[k] - feq[k]) / tau;
    }
  } else {

    int fidx = j * Nx * Q9 + i * Q9;

    fvals[fidx + 4] = ftemp[2];
    fvals[fidx + 7] = ftemp[5];
    fvals[fidx + 8] = ftemp[6];

    fvals[fidx + 2] = ftemp[4];
    fvals[fidx + 5] = ftemp[7];
    fvals[fidx + 6] = ftemp[8];
  }
}