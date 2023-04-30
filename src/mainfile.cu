#include "cudalbm.h"
#include "seriallbm.hpp"

inline void chkerr(cudaError_t code) {
  if (code != cudaSuccess) {
    std::cerr << "ERROR!!!:" << cudaGetErrorString(code) << std::endl;
    exit(-1);
  }
}

void accuracyTest(double *ux, double *uy, double *uxd, double *uyd, int sz) {

  double error{0};
  double tol{1e-4};
  bool success = true;

  for (int i = 0; i < sz; i++) {

    if (abs(ux[i] - uxd[i]) > tol) {
      std::cout << "Outputs don't match at i = \t" << i << "\n";
      success = false;
      break;
    }

    if (abs(uy[i] - uyd[i]) > tol) {
      std::cout << "Outputs don't match at i = \t" << i << "\n";
      success = false;
      break;
    }
  }

  if (success)
    std::cout << "SUCCESS, Outputs match \n";
}

void analyze_AOS( int sz, int szf, double tau, double g, double U ) {

  double *fvals = new double[szf];
  std::fill(fvals, fvals + szf, 0);

  double *fvalsinit = new double[szf];
  std::fill(fvalsinit, fvalsinit + szf, 0);

  double *rho = new double[sz];
  std::fill(rho, rho + sz, 1);

  double *ux = new double[sz];
  std::fill(ux, ux + sz, 0);

  double *uy = new double[sz];
  std::fill(uy, uy + sz, 0);

  double *rhoprev = new double[sz];
  std::fill(rhoprev, rhoprev + sz, 0);

  double *uxprev = new double[sz];
  std::fill(uxprev, uxprev + sz, 0);

  double *uyprev = new double[sz];
  std::fill(uyprev, uyprev + sz, 0);

  double *fvalsprev = new double[szf];
  std::fill(fvalsprev, fvalsprev + szf, 0);

  double *feq = new double[Q9];
  std::fill(feq, feq + Q9, 0);

  setInitialVelocity(ux, uy, U);

  calcEqDis_AOS(fvalsinit, rho, ux, uy, g, tau);
  calcEqDis_AOS(fvalsprev, rho, ux, uy, g, tau);
  calcEqDis_AOS(fvals, rho, ux, uy, g, tau);

  double *ex = new double[Q9]{0, 1, 0, -1, 0, 1, -1, -1, 1};
  double *ey = new double[Q9]{0, 0, 1, 0, -1, 1, 1, -1, -1};

  double c = 1;
  int Niter = 1;
  double tol = 1e-8;
  int t = 0;

  cudaEvent_t seq_start, seq_stop;
  float seq_time;
  cudaEventCreate(&seq_start);
  cudaEventCreate(&seq_stop);
  cudaEventRecord(seq_start);

  // performLBMPushOut( fvals, rho, ux, uy, ex, ey, g, tau, szf, Niter );
  // performLBMPullIn(fvals, fvalsprev, feq, rho, ux, uy, uxprev, uyprev, ex, ey,
                  //  g, tau, szf, Niter, tol);
  // double error = 1e6;
  performLBMStepsPullIn_AOS(fvals, fvalsprev, feq, ex, ey, tau, g);

  // while (t < Niter)
  // {
  //     performLBMStepsPullIn_AOS(fvals, fvalsprev, feq, ex, ey, tau, g);
  //     calcMacroscopic_AOS( fvals, rho, ux, uy, ex, ey );
  //     error = calcVelError( ux, uy, uxprev, uyprev, tol );
  //     t += 1;
  //     std::swap( ux, uxprev );
  //     std::swap( uy, uyprev );

  //     // std::cout << error << "\t" << "Iteration = \t" << t << "\n";

  //     // if( error > tol )
  //     if( t < Niter )
  //         std::swap( fvals, fvalsprev );
  // }

  cudaEventRecord(seq_stop);
  cudaEventSynchronize(seq_stop);
  cudaEventElapsedTime(&seq_time, seq_start, seq_stop);

  calcMacroscopic_AOS(fvals, rho, ux, uy, ex, ey);

  double *dfvals, *dfvalsprev, *dex, *dey;
  chkerr(cudaMalloc((void **)&dfvals, sizeof(double) * szf));
  chkerr(cudaMalloc((void **)&dfvalsprev, sizeof(double) * szf));
  chkerr(cudaMalloc((void **)&dex, sizeof(double) * Q9));
  chkerr(cudaMalloc((void **)&dey, sizeof(double) * Q9));

  chkerr(cudaMemcpy(dfvals, fvalsinit, sizeof(double) * szf,
                    cudaMemcpyHostToDevice));
  chkerr(cudaMemcpy(dfvalsprev, fvalsinit, sizeof(double) * szf,
                    cudaMemcpyHostToDevice));
  chkerr(cudaMemcpy(dex, ex, sizeof(double) * Q9, cudaMemcpyHostToDevice));
  chkerr(cudaMemcpy(dey, ey, sizeof(double) * Q9, cudaMemcpyHostToDevice));

  double *rhod = new double[sz];
  std::fill(rhod, rhod + sz, 0);

  double *uxd = new double[sz];
  std::fill(uxd, uxd + sz, 0);

  double *uyd = new double[sz];
  std::fill(uyd, uyd + sz, 0);

  dim3 block_spec;
  block_spec.x = BLOCKSIZE;

  int gsize = ceil(sz / ((double)BLOCKSIZE));

  dim3 grid_spec(gsize, 1);

  t = 0;
  std::cout << "Entering Device Code \n";

  cudaEvent_t par_start, par_stop;
  float par_time;
  cudaEventCreate(&par_start);
  cudaEventCreate(&par_stop);
  cudaEventRecord(par_start);

  parlbm_AOS<<<grid_spec, block_spec>>>(dfvals, dfvalsprev, dex, dey, g, tau, szf);

  // while (t < Niter) {
  //   parlbm_AOS<<<grid_spec, block_spec>>>(dfvals, dfvalsprev, dex, dey, g, tau,
  //                                     szf);
  //   t++;

  //   if (t < Niter)
  //     std::swap(dfvals, dfvalsprev);
  // }

  cudaEventRecord(par_stop);
  cudaEventSynchronize(par_stop);
  cudaEventElapsedTime(&par_time, par_start, par_stop);

  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  double *fvalsd = new double[szf];
  std::fill(fvalsd, fvalsd + szf, 0);

  chkerr(
      cudaMemcpy(fvalsd, dfvals, sizeof(double) * szf, cudaMemcpyDeviceToHost));

  chkerr(cudaFree(dfvals));
  chkerr(cudaFree(dfvalsprev));
  chkerr(cudaFree(dex));
  chkerr(cudaFree(dey));

  calcMacroscopic_AOS(fvalsd, rhod, uxd, uyd, ex, ey);

  accuracyTest(ux, uy, uxd, uyd, sz);

  std::string fullfiledir = root_dir + "TextFiles/AOS/";

  // printu(ux, uy,fullfiledir + "velocity_AOS.txt");
  // printval(rho, fullfiledir + "rho_AOS.txt");
  // printf(fvals, fullfiledir + "fvals_AOS.txt");

  // printu(uxd, uyd, fullfiledir + "velocitydevice_AOS.txt");
  // printval(rhod, fullfiledir + "rhodevice_AOS.txt");
  // printf(fvalsd, fullfiledir + "fvalsdevice_AOS.txt");

  std::string filenameval = fullfiledir + "timecalcNx=" + std::to_string(Nx) + "Ny=" + std::to_string(Ny) + ".txt";

  std::ofstream fileval( filenameval );

  fileval << seq_time << "\n";
  fileval << par_time << "\n";
  fileval << seq_time/par_time << "\n";

  double seqmlups = sz*Niter*1.0/std::pow( 10, 6 )/seq_time;
  double parmlups = sz*Niter*1.0/std::pow( 10, 6 )/par_time;

  fileval << seqmlups << "\n";
  fileval << parmlups << "\n";
  fileval << sz << "\n";

}

void analyze_SOA( int sz, int szf, double tau, double g, double U ) {

  double *fvals = new double[szf];
  std::fill(fvals, fvals + szf, 0);

  double *fvalsinit = new double[szf];
  std::fill(fvalsinit, fvalsinit + szf, 0);

  double *rho = new double[sz];
  std::fill(rho, rho + sz, 1);

  double *ux = new double[sz];
  std::fill(ux, ux + sz, 0);

  double *uy = new double[sz];
  std::fill(uy, uy + sz, 0);

  double *rhoprev = new double[sz];
  std::fill(rhoprev, rhoprev + sz, 0);

  double *uxprev = new double[sz];
  std::fill(uxprev, uxprev + sz, 0);

  double *uyprev = new double[sz];
  std::fill(uyprev, uyprev + sz, 0);

  double *fvalsprev = new double[szf];
  std::fill(fvalsprev, fvalsprev + szf, 0);

  double *feq = new double[Q9];
  std::fill(feq, feq + Q9, 0);

  setInitialVelocity(ux, uy, U);

  calcEqDis_AOS(fvalsinit, rho, ux, uy, g, tau);
  calcEqDis_AOS(fvalsprev, rho, ux, uy, g, tau);
  calcEqDis_AOS(fvals, rho, ux, uy, g, tau);

  double *ex = new double[Q9]{0, 1, 0, -1, 0, 1, -1, -1, 1};
  double *ey = new double[Q9]{0, 0, 1, 0, -1, 1, 1, -1, -1};

  double c = 1;
  int Niter = 1;
  double tol = 1e-8;
  int t = 0;

  cudaEvent_t seq_start, seq_stop;
  float seq_time;
  cudaEventCreate(&seq_start);
  cudaEventCreate(&seq_stop);
  cudaEventRecord(seq_start);

  // performLBMPushOut( fvals, rho, ux, uy, ex, ey, g, tau, szf, Niter );
  // performLBMPullIn(fvals, fvalsprev, feq, rho, ux, uy, uxprev, uyprev, ex, ey,
                  //  g, tau, szf, Niter, tol);
  // double error = 1e6;
  performLBMStepsPullIn_AOS(fvals, fvalsprev, feq, ex, ey, tau, g);

  // while (t < Niter)
  // {
  //     performLBMStepsPullIn_SOA(fvals, fvalsprev, feq, ex, ey, tau, g);
  //     calcMacroscopic_SOA( fvals, rho, ux, uy, ex, ey );
  //     error = calcVelError( ux, uy, uxprev, uyprev, tol );
  //     // std::swap( fvalsprev, fvals );
  //     t += 1;
  //     std::swap( ux, uxprev );
  //     std::swap( uy, uyprev );

  //     std::cout << error << "\t" << "Iteration = \t" << t << "\n";

  //     // if( error > tol )
  //     if( t < Niter )
  //         std::swap( fvals, fvalsprev );
  // }

  cudaEventRecord(seq_stop);
  cudaEventSynchronize(seq_stop);
  cudaEventElapsedTime(&seq_time, seq_start, seq_stop);

  calcMacroscopic_AOS(fvals, rho, ux, uy, ex, ey);

  double *dfvals, *dfvalsprev, *dex, *dey;

  chkerr(cudaMalloc((void **)&dfvals, sizeof(double) * szf));
  chkerr(cudaMalloc((void **)&dfvalsprev, sizeof(double) * szf));
  chkerr(cudaMalloc((void **)&dex, sizeof(double) * Q9));
  chkerr(cudaMalloc((void **)&dey, sizeof(double) * Q9));

  double *rhod = new double[sz];
  std::fill(rhod, rhod + sz, 0);

  double *uxd = new double[sz];
  std::fill(uxd, uxd + sz, 0);

  double *uyd = new double[sz];
  std::fill(uyd, uyd + sz, 0);

  setInitialVelocity(uxd, uyd, U);
  std::fill(fvalsinit, fvalsinit + szf, 0);
  calcEqDis_SOA(fvalsinit, rhod, uxd, uyd, g, tau);

  chkerr(cudaMemcpy(dfvals, fvalsinit, sizeof(double) * szf,
                    cudaMemcpyHostToDevice));
  chkerr(cudaMemcpy(dfvalsprev, fvalsinit, sizeof(double) * szf,
                    cudaMemcpyHostToDevice));
  chkerr(cudaMemcpy(dex, ex, sizeof(double) * Q9, cudaMemcpyHostToDevice));
  chkerr(cudaMemcpy(dey, ey, sizeof(double) * Q9, cudaMemcpyHostToDevice));

  dim3 block_spec;
  block_spec.x = BLOCKSIZE;

  int gsize = ceil(sz / ((double)BLOCKSIZE));

  dim3 grid_spec(gsize, 1);

  t = 0;
  std::cout << "Entering Device Code \n";

  cudaEvent_t par_start, par_stop;
  float par_time;
  cudaEventCreate(&par_start);
  cudaEventCreate(&par_stop);
  cudaEventRecord(par_start);

  parlbm_SOA<<<grid_spec, block_spec>>>(dfvals, dfvalsprev, dex, dey, g, tau, szf);

  // while (t < Niter) {
  //   parlbm_SOA<<<grid_spec, block_spec>>>(dfvals, dfvalsprev, dex, dey, g, tau,
  //                                     szf);
  //   t++;

  //   if (t < Niter)
  //     std::swap(dfvals, dfvalsprev);
  // }

  cudaEventRecord(par_stop);
  cudaEventSynchronize(par_stop);
  cudaEventElapsedTime(&par_time, par_start, par_stop);

  cudaError_t err = cudaGetLastError();

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  double *fvalsd = new double[szf];
  std::fill(fvalsd, fvalsd + szf, 0);

  chkerr(
      cudaMemcpy(fvalsd, dfvals, sizeof(double) * szf, cudaMemcpyDeviceToHost));

  chkerr(cudaFree(dfvals));
  chkerr(cudaFree(dfvalsprev));
  chkerr(cudaFree(dex));
  chkerr(cudaFree(dey));

  calcMacroscopic_SOA(fvalsd, rhod, uxd, uyd, ex, ey);

  accuracyTest(ux, uy, uxd, uyd, sz);

  std::string fullfiledir = root_dir + "TextFiles/SOA/";

  // printu(ux, uy,fullfiledir + "velocity_SOA.txt");
  // printval(rho, fullfiledir + "rho_SOA.txt");
  // printf(fvals, fullfiledir + "fvals_SOA.txt");

  // printu(uxd, uyd, fullfiledir + "velocitydevice_SOA.txt");
  // printval(rhod, fullfiledir + "rhodevice_SOA.txt");
  // printf(fvalsd, fullfiledir + "fvalsdevice_SOA.txt");

  std::string filenameval = fullfiledir + "timecalcNx=" + std::to_string(Nx) + "Ny=" + std::to_string(Ny) + ".txt";

  std::ofstream fileval( filenameval );

  fileval << seq_time << "\n";
  fileval << par_time << "\n";
  fileval << seq_time/par_time << "\n";

  double seqmlups = sz*Niter*1.0/std::pow( 10, 6 )/seq_time;
  double parmlups = sz*Niter*1.0/std::pow( 10, 6 )/par_time;

  fileval << seqmlups << "\n";
  fileval << parmlups << "\n";
  // fileval << sz << "\n";

}

int main( int argc, char* argv[] ) {

  std::string optimization = argv[1];

  int szf = Ny * Nx * Q9;
  int sz = Nx * Ny;

  constexpr double tau = 1;
  // double g = 0.0001373;
  // double U = 0.0333*1.5;

  constexpr double g = 0.001102;
  constexpr double U = 0.1;

  if( optimization == "AOS" ) 
    analyze_AOS( sz, szf, tau, g, U );
  else if( optimization == "SOA" )
    analyze_SOA( sz, szf, tau, g, U );
  else if( optimization == "SM" ) {

  }
  else {
    throw std::invalid_argument( "Wrong argument" );
  }

  return 0;
}