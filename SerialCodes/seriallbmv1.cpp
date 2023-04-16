#include<iostream>
#include<vector>
#include<fstream>
#include<cstring>
#include<algorithm>
#include<math.h>

#define Q9 9
#define dim 2

constexpr int Ny = 12;
constexpr int Nx = 40;

void calcMacroscopic( double* fvals, double* rho, double* ux, double* uy, double* ex, double* ey ) {

    for( int j = 0; j < Ny; j++ ) {
        for( int i = 0; i < Nx; i++ ) {

            int idx = j*Nx + i;

            ux[ idx ] = 0;
            uy[ idx ] = 0;
            rho[ idx ] = 0;

            // if( !isSolid[ idx ] ) {

                for( int k = 0; k < Q9; k++ ) {
                    
                    int fidx = idx*Q9 + k;

                    rho[ idx ] += fvals[ fidx ];
                    ux[ idx ] += fvals[ fidx ] * ex[ k ];
                    uy[ idx ] += fvals[ fidx ] * ey[ k ];

                }

                if( rho[ idx ] != 0 ) {

                    // std::cout << rho[ idx ] << "\n";

                    ux[ idx ] /= rho[ idx ];
                    uy[ idx ] /= rho[ idx ];

                    // std::cout << ux[ idx ] << "\n";
                }

            // }
        }
    }
}

void performStream( double* fvals, double* ftemp, double* ex, double* ey ) {

    for( int j = 0; j < Ny; j++ ) {
        for( int i = 0; i < Nx; i++ ) {

            int idx = j*Nx + i;

            for( int k = 0; k < Q9; k++ ) {

                int xval = ex[k];
                int yval = ey[k];

                int tempi = xval + i;
                int tempj = yval + j;

                if( tempi == Nx ) {
                    tempi = 0;
                }
                else if( tempi == -1 ) {
                    tempi = Nx - 1;
                }

                if( tempj == Ny ) {
                    tempj = 0;
                }
                else if( tempj == -1 ) {
                    tempj = Ny - 1;
                }

                // std::cout << tempi << "\t" << tempj << "\t" << i << "\t" << j << "\n";

                int fidx = idx*Q9 + k;

                int ftempidx = tempj*Nx*Q9 + tempi*Q9 + k;

                ftemp[ ftempidx ] = fvals[ fidx ];

            }            
        }
    }
}

void calcEqDis( double* feq, double* rho, double* ux, double* uy, double g, double tau ) {

    double f1 = 3.0;
    double f2 = 9.0/2.0;
    double f3 = 3.0/2.0;

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

    for( int j = 1; j < Ny - 1; j++ ) {
        for( int i = 0; i < Nx; i++ ) {

            int idx = j*Nx + i;

            rt0 = ( 4.0/9.0 )*rho[ idx ];
            rt1 = ( 1.0/9.0 )*rho[idx];
            rt2 = ( 1.0/36.0 )*rho[idx];

            ueqxij = ux[ idx ] + tau*g;
            ueqyij = uy[ idx ];

            uxsq = ueqxij * ueqxij;
            uysq = ueqyij * ueqyij;
            uxuy5 = ueqxij + ueqyij;
            uxuy6 = -ueqxij + ueqyij;
            uxuy7 = -ueqxij -ueqyij;
            uxuy8 = ueqxij -ueqyij;
            usq = uxsq + uysq; 

            int fidx = idx*Q9;

            feq[ fidx + 0 ] = rt0*( 1 - f3*usq);
            feq[ fidx + 1 ] = rt1*( 1 + f1*ueqxij + f2*uxsq - f3*usq);
            feq[ fidx + 2 ] = rt1*( 1 + f1*ueqyij + f2*uysq - f3*usq);
            feq[ fidx + 3 ] = rt1*( 1 - f1*ueqxij + f2*uxsq - f3*usq);
            feq[ fidx + 4 ] = rt1*( 1 - f1*ueqyij + f2*uysq - f3*usq);
            feq[ fidx + 5 ] = rt2*( 1 + f1*uxuy5 + f2*uxuy5*uxuy5 - f3*usq);
            feq[ fidx + 6 ] = rt2*( 1 + f1*uxuy6 + f2*uxuy6*uxuy6 - f3*usq);
            feq[ fidx + 7 ] = rt2*( 1 + f1*uxuy7 + f2*uxuy7*uxuy7 - f3*usq);
            feq[ fidx + 8 ] = rt2*( 1 + f1*uxuy8 + f2*uxuy8*uxuy8 - f3*usq);

            // std::cout << feq[ fidx ] << "\t" << feq[ fidx + 1 ] << "\t" << feq[ fidx + 2 ] << "\n";

        }
    }
}

void collide( double* f, double*ftemp, double* feq, double tau  ) {

    for( int j = 1; j < Ny - 1; j++ ) {
        for( int i = 0; i < Nx; i++ ) {

            for( int k = 0; k < Q9; k++ ) {

                int fidx = j*Nx*Q9 + i*Q9 + k;

                f[ fidx ] = ftemp[ fidx ] - ( ftemp[ fidx ] - feq[ fidx ] )/tau;

            }

        }
    }
}

void applyBC( double* f, double* ftemp ) {

    for( int i = 0; i < Nx; i++ ) {

        int fupidx = ( Ny - 1 )*Nx*Q9 + i*Q9;

        int flowidx = i*Q9;

        std::swap( f[ fupidx + 1 ], f[ fupidx + 3 ] );
        std::swap( f[ fupidx + 2 ], f[ fupidx + 4 ] );
        std::swap( f[ fupidx + 5 ], f[ fupidx + 7 ] );
        std::swap( f[ fupidx + 6 ], f[ fupidx + 8 ] );
        std::swap( f[ fupidx + 3 ], f[ fupidx + 1 ] );
        std::swap( f[ fupidx + 4 ], f[ fupidx + 2 ] );
        std::swap( f[ fupidx + 7 ], f[ fupidx + 5 ] );
        std::swap( f[ fupidx + 8 ], f[ fupidx + 6 ] );

        std::swap( f[ flowidx + 1 ], f[ flowidx + 3 ] );
        std::swap( f[ flowidx + 2 ], f[ flowidx + 4 ] );
        std::swap( f[ flowidx + 5 ], f[ flowidx + 7 ] );
        std::swap( f[ flowidx + 6 ], f[ flowidx + 8 ] );
        std::swap( f[ flowidx + 3 ], f[ flowidx + 1 ] );
        std::swap( f[ flowidx + 4 ], f[ flowidx + 2 ] );
        std::swap( f[ flowidx + 7 ], f[ flowidx + 5 ] );
        std::swap( f[ flowidx + 8 ], f[ flowidx + 6 ] );

        // f[ fupidx + 1 ] = ftemp[ fupidx + 3 ];
        // f[ fupidx + 2 ] = ftemp[ fupidx + 4 ];
        // f[ fupidx + 5 ] = ftemp[ fupidx + 7 ];
        // f[ fupidx + 6 ] = ftemp[ fupidx + 8 ];
        // f[ fupidx + 3 ] = ftemp[ fupidx + 1 ];
        // f[ fupidx + 4 ] = ftemp[ fupidx + 2 ];
        // f[ fupidx + 7 ] = ftemp[ fupidx + 5 ];
        // f[ fupidx + 8 ] = ftemp[ fupidx + 6 ];

        // f[ flowidx + 1 ] = ftemp[ flowidx + 3 ];
        // f[ flowidx + 2 ] = ftemp[ flowidx + 4 ];
        // f[ flowidx + 5 ] = ftemp[ flowidx + 7 ];
        // f[ flowidx + 6 ] = ftemp[ flowidx + 8 ];
        // f[ flowidx + 3 ] = ftemp[ flowidx + 1 ];
        // f[ flowidx + 4 ] = ftemp[ flowidx + 2 ];
        // f[ flowidx + 7 ] = ftemp[ flowidx + 5 ];
        // f[ flowidx + 8 ] = ftemp[ flowidx + 6 ];

    }

}

double calcError( double* val, double* valprev ) {

    double error{0};
    int idx{0};

    for( int j = 1; j < Ny - 1; j++ ) {
        for( int i = 1; i < Nx - 1; i++ ) {

            idx = j*Nx + i;

            error += std::pow( ( val[ idx ] - valprev[ idx ] ), 2 );

        }
    }

    error = std::sqrt( error );

}

void performLBM( double* fvals, double* rho, double* ux, double* uy, double*ex, double* ey, double g, double tau, int szf, int Niter ) {

    double* ftemp = new double[ szf ];
    std::fill( ftemp, ftemp + szf, 0 );

    double* feq = new double[ szf ];
    std::fill( feq, feq + szf, 0 );

    int sz = Nx*Ny;

    double* rhoprev = new double[ sz ];
    for( int i = 0; i < sz; i++ ) {
        rhoprev[i] = rho[i];
    }

    double error{0.2};
    double tol{ 1e-20 };
    int t = 0;

    while( t < Niter ) {

        calcMacroscopic( fvals, rho, ux, uy, ex, ey );
        performStream( fvals, ftemp, ex, ey );
        calcEqDis( feq, rho, ux, uy, g, tau );
        collide( fvals, ftemp, feq, tau );
        applyBC( fvals, ftemp );

        error = calcError( rho, rhoprev );
        std::swap( rhoprev, rho );
        std::cout << error << "\n";

        t += 1;
    }

}

void printu( double* ux, double* uy, std::string fname ){

    std::ofstream fhandle{ fname, std::ofstream::out };

    fhandle << Nx << "\t" << Ny << "\n";    

    for( int j = 0; j < Ny; j++ ) {
        for( int i = 0; i < Nx; i++ ) {

            int idx = j*Nx + i;

            fhandle << ux[ idx ] << "\t" << uy[ idx ] << "\t" << i << "\t" << j << "\n";

        }
    }

    fhandle.close();

}

void printval( double* val, std::string fname ){

    std::ofstream fhandle{ fname, std::ofstream::out };

    fhandle << Nx << "\t" << Ny << "\n";    

    for( int j = 0; j < Ny; j++ ) {
        for( int i = 0; i < Nx; i++ ) {

            int idx = j*Nx + i;

            fhandle << val[ idx ] << "\n";

        }
    }

    fhandle.close();

}

void printf( double* val, std::string fname ){

    std::ofstream fhandle{ fname, std::ofstream::out };

    fhandle << Nx << "\t" << Ny << "\n";    

    for( int j = 0; j < Ny; j++ ) {
        for( int i = 0; i < Nx; i++ ) {

            int idx = j*Nx + i;

            for( int k = 0; k < Q9; k++ ) {

                int fidx = idx*Q9 + k;

                fhandle << val[fidx] << "\t";

            }

            fhandle << "\n";

        }
    }

    fhandle.close();

}

void setInitialVelocity( double* ux, double* uy, double U ) {

    for( int j = 0; j < Ny; j++ ) {

        for( int i = 0; i < Nx; i++ ) {

            double y = j/ (double) (Ny - 1);

            int idx = j*Nx + i;

            ux[ idx ] = 4*U*( y - y*y );

        }

    }

}

int main() {

    int szf = Ny*Nx*Q9; 
    int sz = Nx*Ny;

    double tau = 1;
    double g = 0.0001373;
    double U = 0.0333;

    double* fvals = new double[ szf ];
    std::fill( fvals, fvals + szf, 0.001 );

    double* rho = new double[ sz ];
    std::fill( rho, rho + sz, 1 );

    double* ux = new double[ sz ];
    std::fill( ux, ux + sz, 0 );

    double* uy = new double[ sz ];
    std::fill( uy, uy + sz, 0 );

    setInitialVelocity( ux, uy, U );

    calcEqDis( fvals, rho, ux, uy, g, tau );
    
    double restw = 4.0/9.0;
    double stw = 1.0/9.0;
    double diagw = 1.0/36.0;

    double* ex = new double[ Q9 ]{ 0, 1, 0, -1, 0, 1, -1, -1, 1 };
    double* ey = new double[ Q9 ]{ 0, 0, 1, 0, -1, 1, 1, -1, -1 };
    double* w = new double[ Q9 ]{ restw, stw, stw, stw, stw, diagw, diagw, diagw, diagw };

    std::cout << restw << "\t" << stw << "\t" << diagw << "\n";

    double c = 1;
    int Niter = 100;

    performLBM( fvals, rho, ux, uy, ex, ey, g, tau, szf, Niter );

    printu( ux, uy, "velocity.txt" );
    printval( rho, "rho.txt" );
    printf( fvals, "fvals.txt" );

    return 0;
}