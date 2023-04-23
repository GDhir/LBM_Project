#include "seriallbm.hpp"

int main()
{

    int szf = Ny * Nx * Q9;
    int sz = Nx * Ny;

    double tau = 1;
    // double g = 0.0001373;
    // double U = 0.0333*1.5;

    double g = 0.001102;
    double U = 0.1;

    double *fvals = new double[szf];
    std::fill(fvals, fvals + szf, 0.001);

    double *rho = new double[sz];
    std::fill(rho, rho + sz, 1);

    double *ux = new double[sz];
    std::fill(ux, ux + sz, 0);

    double *uy = new double[sz];
    std::fill(uy, uy + sz, 0);

    setInitialVelocity(ux, uy, U);

    calcEqDis(fvals, rho, ux, uy, g, tau);

    double *ex = new double[Q9]{0, 1, 0, -1, 0, 1, -1, -1, 1};
    double *ey = new double[Q9]{0, 0, 1, 0, -1, 1, 1, -1, -1};

    double c = 1;
    int Niter = 5000;

    // performLBMPushOut( fvals, rho, ux, uy, ex, ey, g, tau, szf, Niter );
    performLBMPullIn(fvals, rho, ux, uy, ex, ey, g, tau, szf, Niter);

    printu(ux, uy, "velocity.txt");
    printval(rho, "rho.txt");
    printf(fvals, "fvals.txt");

    return 0;
}