import numpy as np
import os
import matplotlib.pyplot as plt

def plotu( ux, uy ):

    a = 5

def plotvelocity( filename, filenameprefix ):

    with open( filename ) as filev1:

        linevals = filev1.readlines()

        Nx, Ny = linevals[0].split()

        Nx, Ny = int(Nx), int(Ny)

        uxvals = np.zeros( ( Ny, Nx ) )
        uyvals = np.zeros( ( Ny, Nx ) )

        uvals = np.zeros( ( Ny, Nx ) )

        X = np.zeros( ( Ny, Nx ) )
        Y = np.zeros( ( Ny, Nx ) )


        print( Nx, Ny )

        for line in linevals[ 1: ]:

            ux, uy, i, j = line.split()

            ux, uy, i, j = float( ux ), float( uy ), int( i ), int( j )

            uvals[ j, i ] = np.sqrt( ux**2 + uy**2 )
            uxvals[ j, i ] = ux
            uyvals[ j, i ] = uy
            X[ j, i ] = i
            Y[ j, i ] = j

        
        print( uyvals[ 1:Ny - 1, : ] )

        plt.figure()
        plt.quiver( X[ 1:Ny - 1, : ], Y[ 1:Ny - 1, : ], uxvals[ 1:Ny - 1, : ], uyvals[ 1:Ny - 1, : ] )
        plt.savefig( filenameprefix + "plotval_pullin.png" )

        plt.figure()
        plt.contour( X[ 1:Ny - 1, : ], Y[ 1:Ny - 1, : ], uvals[ 1:Ny - 1, : ], levels = 40 )
        plt.colorbar()
        plt.savefig( filenameprefix + "plotcontour_pullin.png" )

        plt.figure()
        plt.contourf( X[ 1:Ny - 1, : ], Y[ 1:Ny - 1, : ], uvals[ 1:Ny - 1, : ], levels = 40 )
        plt.colorbar()
        plt.savefig( filenameprefix + "plotcontourf_pullin.png" )

if __name__ == "__main__":

    print (os.getcwd())

    plotvelocity( "../build/velocity.txt", "host" )
    plotvelocity( "../build/velocitydevice.txt", "device" )

    a = 4

    

