#include "seriallbm.hpp"
#include "cudalbm.h"


inline void chkerr(cudaError_t code)
{
    if (code != cudaSuccess)
    {
        std::cerr << "ERROR!!!:" << cudaGetErrorString(code) <<std::endl;
        exit(-1);
    }
}

void accuracyTest( double *ux, double *uy, double *uxd, double *uyd, int sz ) {

    double error{0};
    double tol{1e-4};

    for( int i = 0; i < sz; i++ ) {

        if( abs( ux[i] - uxd[i] ) > tol ) {
            std::cout << "Outputs don't match at i = \t" << i << "\n";
            break; 
        }

        if( abs( uy[i] - uyd[i] ) > tol ) {
            std::cout << "Outputs don't match at i = \t" << i << "\n";
            break; 
        }

    }

    std::cout << "SUCCESS, Outputs match \n";

}

int main()
{

    int szf = Ny * Nx * Q9;
    int sz = Nx * Ny;

    constexpr double tau = 1;
    // double g = 0.0001373;
    // double U = 0.0333*1.5;

    constexpr double g = 0.001102;
    constexpr double U = 0.1;

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

    double *fvalsprev = new double[szf];
    std::fill(fvalsprev, fvalsprev + szf, 0);

    double *feq = new double[Q9];
    std::fill(feq, feq + Q9, 0);

    setInitialVelocity(ux, uy, U);

    calcEqDis(fvalsinit, rho, ux, uy, g, tau);
    calcEqDis(fvalsprev, rho, ux, uy, g, tau);
    calcEqDis(fvals, rho, ux, uy, g, tau);

    double *ex = new double[Q9]{0, 1, 0, -1, 0, 1, -1, -1, 1};
    double *ey = new double[Q9]{0, 0, 1, 0, -1, 1, 1, -1, -1};

    double c = 1;
    int Niter = 1;

    // performLBMPushOut( fvals, rho, ux, uy, ex, ey, g, tau, szf, Niter );
    performLBMPullIn(fvals, fvalsprev, feq, rho, ux, uy, ex, ey, g, tau, szf, Niter);

    calcMacroscopic(fvals, rho, ux, uy, ex, ey);

    double* dfvals, *dfvalsprev, *dex, *dey, *dfeq; 
    chkerr(cudaMalloc((void **) &dfvals,  sizeof(double) * szf));
    chkerr(cudaMalloc((void **) &dfvalsprev,  sizeof(double) * szf));
    chkerr(cudaMalloc((void **) &dex,  sizeof(double) * Q9));
    chkerr(cudaMalloc((void **) &dey,  sizeof(double) * Q9));

    chkerr(cudaMemcpy(dfvals, fvalsinit, sizeof(double) * szf, cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dfvalsprev, fvalsinit, sizeof(double) * szf, cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dex, ex, sizeof(double) * Q9, cudaMemcpyHostToDevice));
    chkerr(cudaMemcpy(dey, ey, sizeof(double) * Q9, cudaMemcpyHostToDevice));

    dim3 block_spec;
    block_spec.x = BLOCKSIZE;

    int gsize = ceil( szf/( (double) BLOCKSIZE ) );

    dim3 grid_spec( gsize, 1 );

    parlbm<<<grid_spec, block_spec>>>(dfvals, dfvalsprev, dex, dey, g, tau, szf);

    double *rhod = new double[sz];
    std::fill(rhod, rhod + sz, 0);

    double *uxd = new double[sz];
    std::fill(uxd, uxd + sz, 0);

    double *uyd = new double[sz];
    std::fill(uyd, uyd + sz, 0);

    double *fvalsd = new double[szf];
    std::fill(fvalsd, fvalsd + szf, 0);

    chkerr( cudaMemcpy( fvalsd, dfvals, sizeof(double) * szf, cudaMemcpyDeviceToHost) );
    
    calcMacroscopic(fvalsd, rhod, uxd, uyd, ex, ey);

    accuracyTest( ux, uy, uxd, uyd, sz );

    printu(ux, uy, "velocity.txt");
    printval(rho, "rho.txt");
    printf(fvals, "fvals.txt");

    printu(uxd, uyd, "velocitydevice.txt");
    printval(rhod, "rhodevice.txt");
    printf(fvalsd, "fvalsdevice.txt");

    return 0;
}