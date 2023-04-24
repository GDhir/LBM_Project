#include "utils.hpp"
#include<cmath>

double calcVelError( double* ux, double* uy, double* uxprev, double* uyprev, double tol ) {

    double error{0};
    double u;
    double uprev;

    for( int j = 1; j < Ny - 1; j++ ) {
        for( int i = 0; i < Nx; i++ ) {

            int idx = j*Nx + i;

            u = std::sqrt( std::pow( ux[idx], 2 ) + std::pow( uy[idx], 2 ) );
            uprev = std::sqrt( std::pow( uxprev[idx], 2 ) + std::pow( uyprev[idx], 2 ) );

            error += std::pow( u - uprev, 2 );

        }
    }

    return std::sqrt( error );   
}