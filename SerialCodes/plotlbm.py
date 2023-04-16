import numpy as np
import os
import matplotlib.pyplot as plt

def plotu( ux, uy ):

    a = 5

if __name__ == "__main__":

    print (os.getcwd())

    with open( "/home/gaurav/LBM_Project/SerialCodes/velocity.txt" ) as filev1:

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

        plt.quiver( X, Y, uxvals, uyvals )
        plt.savefig( "plotval.png" )

        plt.figure()
        plt.contour( X, Y, uvals )
        plt.savefig( "plotcontour.png" )


    a = 4

    

