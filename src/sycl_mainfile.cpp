#include <sycl/sycl.hpp>
#include "utils.hpp"
#include "seriallbm.hpp"
#include <iostream>
using namespace sycl;

class Kernel;

double performLBMStepsSYCL( sycl::queue& Q, sycl::nd_range<1>& grid, sycl::buffer<double, 1>& fvalsbuf, sycl::buffer<double, 1>& fvalsprevbuf, double tau, double g ) {

    auto wall_begin = std::chrono::steady_clock::now();

    event e = Q.submit([&](handler& h) {

        sycl::accessor fvalsAcc{fvalsbuf, h};
        sycl::accessor fvalsprevAcc{fvalsprevbuf, h};
        // sycl::accessor indexAcc{indexbuf, h};

        h.parallel_for(grid, [=]( auto& it ) {
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

            auto idx = it.get_global_id();
            // indexAcc[ idx ] = idx;

            int i{0}, j{0}, xval{0}, yval{0}, tempi{0}, tempj{0}, tempidx{0}, fidx{0};

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

                    fvalsAcc[ idx + k*sz ] = fvalsprevAcc[ tempidx + k*sz ];

                }

                rho = 0;
                ux = 0;
                uy = 0;

                for (int k = 0; k < Q9; k++) {

                    fidx = idx + k*sz;

                    rho += fvalsAcc[ fidx ];
                    ux += fvalsAcc[ fidx ] * ex[k];
                    uy += fvalsAcc[ fidx ] * ey[k];
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

                    fvalsAcc[ idx + 0*sz ] = fvalsAcc[ idx + 0*sz ] - (fvalsAcc[ idx + 0*sz ] - feq0) / tau;
                    fvalsAcc[ idx + 1*sz ] = fvalsAcc[ idx + 1*sz ] - (fvalsAcc[ idx + 1*sz ] - feq1) / tau;
                    fvalsAcc[ idx + 2*sz ] = fvalsAcc[ idx + 2*sz ] - (fvalsAcc[ idx + 2*sz ] - feq2) / tau;
                    fvalsAcc[ idx + 3*sz ] = fvalsAcc[ idx + 3*sz ] - (fvalsAcc[ idx + 3*sz ] - feq3) / tau;
                    fvalsAcc[ idx + 4*sz ] = fvalsAcc[ idx + 4*sz ] - (fvalsAcc[ idx + 4*sz ] - feq4) / tau;
                    fvalsAcc[ idx + 5*sz ] = fvalsAcc[ idx + 5*sz ] - (fvalsAcc[ idx + 5*sz ] - feq5) / tau;
                    fvalsAcc[ idx + 6*sz ] = fvalsAcc[ idx + 6*sz ] - (fvalsAcc[ idx + 6*sz ] - feq6) / tau;
                    fvalsAcc[ idx + 7*sz ] = fvalsAcc[ idx + 7*sz ] - (fvalsAcc[ idx + 7*sz ] - feq7) / tau;
                    fvalsAcc[ idx + 8*sz ] = fvalsAcc[ idx + 8*sz ] - (fvalsAcc[ idx + 8*sz ] - feq8) / tau;

                }
                else {

                    temp = fvalsAcc[idx + 2*sz];
                    fvalsAcc[idx + 2*sz] = fvalsAcc[idx + 4*sz];
                    fvalsAcc[idx + 4*sz] = temp;

                    temp = fvalsAcc[idx + 7*sz];
                    fvalsAcc[idx + 7*sz] = fvalsAcc[idx + 5*sz];
                    fvalsAcc[idx + 5*sz] = temp;

                    temp = fvalsAcc[idx + 8*sz];
                    fvalsAcc[idx + 8*sz] = fvalsAcc[idx + 6*sz];
                    fvalsAcc[idx + 6*sz] = temp;

                }
            }
        });
    });

    Q.wait_and_throw();

    auto wall_end = std::chrono::steady_clock::now();

    auto wall_diff = std::chrono::duration_cast<std::chrono::nanoseconds>(wall_end - wall_begin).count();
    std::cout << "event time elapsed " << wall_diff << "nanos" << std::endl;

    return(e.template get_profiling_info<info::event_profiling::command_end>() -
       e.template get_profiling_info<info::event_profiling::command_start>());
}

void calcMacroscopicSYCL_SOA( sycl::buffer<double, 1>& fvalsbuf, double* uxd, double* uyd, double* rhod ) {

    sycl::host_accessor hostfvalsAcc{fvalsbuf};
    int fidx{0};

    for (int j = 0; j < Ny; j++)
    {
        for (int i = 0; i < Nx; i++)
        {

            int idx = j * Nx + i;

            uxd[idx] = 0;
            uyd[idx] = 0;
            rhod[idx] = 0;

            // if( !isSolid[ idx ] ) {

            for (int k = 0; k < Q9; k++)
            {

                fidx = sz*k + idx;

                rhod[idx] += hostfvalsAcc[fidx];
                uxd[idx] += hostfvalsAcc[fidx] * ex[k];
                uyd[idx] += hostfvalsAcc[fidx] * ey[k];

            }
            
            // std::cout << hostindexAcc[ idx ] << "\n";

            uxd[idx] /= rhod[idx];
            uyd[idx] /= rhod[idx];
        }
    }

}

int main() {
    std::cout << "Listing Available Platforms\n";

    for (auto platform : sycl::platform::get_platforms())
    {
        std::cout << "Platform: "
                  << platform.get_info<sycl::info::platform::name>()
                  << std::endl;

        for (auto device : platform.get_devices())
        {
            std::cout << "\tDevice: "
                      << device.get_info<sycl::info::device::name>()
                      << std::endl;
        }
    }

    std::cout << "Using the GPU Selctor\n";
    sycl::queue Q(sycl::gpu_selector_v, property::queue::enable_profiling{});

    std::cout << "Selector chose: " << Q.get_device().get_info<sycl::info::device::name>() << std::endl;

    constexpr double tau = 1;
    // double g = 0.0001373;
    // double U = 0.0333*1.5;

    constexpr double g = 0.001102;
    constexpr double U = 0.1;

    double *fvals = new double[szf];
    std::fill(fvals, fvals + szf, 0);

    double *fvalsd = new double[szf];
    std::fill(fvalsd, fvalsd + szf, 0);

    double *fvalsinit = new double[szf];
    std::fill(fvalsinit, fvalsinit + szf, 0);

    double *rho = new double[sz];
    std::fill(rho, rho + sz, 1);

    double *ux = new double[sz];
    std::fill(ux, ux + sz, 0);

    double *uy = new double[sz];
    std::fill(uy, uy + sz, 0);

    double *rhod = new double[sz];
    std::fill(rhod, rhod + sz, 1);

    double *uxd = new double[sz];
    std::fill(uxd, uxd + sz, 0);

    double *uyd = new double[sz];
    std::fill(uyd, uyd + sz, 0);

    double *fvalsprev = new double[szf];
    std::fill(fvalsprev, fvalsprev + szf, 0);

    double *fvalsprevd = new double[szf];
    std::fill(fvalsprevd, fvalsprevd + szf, 0);

    double *feq = new double[Q9];
    std::fill(feq, feq + Q9, 0);

    setInitialVelocity(ux, uy, U);
    setInitialVelocity(uxd, uyd, U);

    calcEqDis_SOA(fvalsinit, rho, ux, uy, g, tau);
    // calcEqDis_SOA(fvalsprev, rho, ux, uy, g, tau);
    // calcEqDis_SOA(fvals, rho, ux, uy, g, tau);

    std::copy( fvalsinit, fvalsinit + szf, fvalsprev );
    std::copy( fvalsinit, fvalsinit + szf, fvalsprevd );    
    std::copy( fvalsinit, fvalsinit + szf, fvals );
    std::copy( fvalsinit, fvalsinit + szf, fvalsd );

    constexpr int Niter = 1;
    // double tol = 1e-8;
    int t = 0;

    auto wall_begin = std::chrono::steady_clock::now();

    performLBMStepsPullIn_SOA(fvals, fvalsprev, feq, tau, g);

    auto wall_end = std::chrono::steady_clock::now();

    auto seq_time = std::chrono::duration_cast<std::chrono::milliseconds>(wall_end - wall_begin).count();

    calcMacroscopic_SOA( fvals, rho, ux, uy );

    /************************************************************************************************/
    // Entering SYCL Code
    /************************************************************************************************/

    // double* indexes = new double[sz];
    // std::fill(indexes, indexes + sz, 0);

    // size_t gridsz = ceil( sz/(double) BLOCKSIZE );

    sycl::range global{sz};
    sycl::range local{BLOCKSIZE};

    sycl::nd_range grid{ global, local };

    auto fvalsbuf = sycl::buffer( fvalsd, sycl::range{szf} );
    auto fvalsprevbuf = sycl::buffer( fvalsprevd, sycl::range{szf} );

    double sycl_time = performLBMStepsSYCL( Q, grid, fvalsbuf, fvalsprevbuf, tau, g )*( pow(10, -6) );
    calcMacroscopicSYCL_SOA( fvalsbuf, uxd, uyd, rhod );

    accuracyTest( ux, uy, uxd, uyd, sz );

    std::string fullfiledir = root_dir + "TextFiles/SYCL/SOA/";

    // printu(ux, uy,fullfiledir + "velocityhost.txt");
    // printval(rho, fullfiledir + "rhohost.txt");
    // printf(fvals, fullfiledir + "fvals.txt");

    printu(uxd, uyd, fullfiledir + "velocitydevice.txt");
    printval(rhod, fullfiledir + "rhodevice.txt");
    // printf(fvals, fullfiledir + "fvalsdevice.txt");

    std::string filenameval = fullfiledir + "timecalcNx=" + std::to_string(Nx) + "Ny=" + std::to_string(Ny) + ".txt";

    std::ofstream fileval( filenameval );
    fileval << seq_time << "\n";
    fileval << sycl_time << "\n";
    fileval << seq_time/sycl_time << "\n";

//   std::cout << seq_time << "\n";
//   std::cout << par_time << "\n";
//   std::cout << seq_time/par_time << "\n";

    double seqmlups = sz*Niter*1.0/std::pow( 10, 3 )/seq_time;
    double parmlups = sz*Niter*1.0/std::pow( 10, 3 )/sycl_time;

    fileval << seqmlups << "\n";
    fileval << parmlups << "\n";

    // fileval << sycl_time << "\n";

    // double syclmlups = sz*Niter*1.0/std::pow( 10, 3 )/sycl_time;

    // fileval << syclmlups << "\n";

    return 0;
}