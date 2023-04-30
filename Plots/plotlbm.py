import numpy as np
import os
import matplotlib.pyplot as plt
import os
import subprocess
import re
from mpl_toolkits import mplot3d

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

        
        # print( uyvals[ 1:Ny - 1, : ] )

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

def runSim( root_dir, outfile, inargs ):

    build_dir = root_dir + "/build"

    compile_cmdlist = ["make"]
    cmdlist = ["./lbm-gpu", inargs]

    open(outfile, 'w').close()

    subprocess.run(compile_cmdlist, cwd=build_dir  )

    subprocess.run(cmdlist, stdout=open(outfile, 'a'), stderr=open(outfile, 'a'), cwd=build_dir  )

def getPerfData( fileval ):

    # build_dir = root_dir + "/TextFiles"

    # fileval = build_dir + "/timecalc_SOA.txt"

    seqdata = 0
    pardata = 0
    speedup = 0

    with open(fileval) as fval:

        alldata = fval.readlines()

        seqdata = float( alldata[ 0 ] ) 
        pardata = float( alldata[ 1 ] )
        speedup = float( alldata[ 2 ] ) 

    return ( seqdata, pardata, speedup )

def plotPerfDataContourScatter( root_dir, folderName, Nxscatter, Nyscatter, speedup, Nxvals, Nyvals ):

    fullfileprefix = root_dir + "PlotFiles/" + folderName + "/"

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(Nxscatter, Nyscatter, speedup)
    ax.set_xlabel('Nx')
    ax.set_ylabel('Ny')
    ax.set_zlabel('SpeedUp')
    plt.savefig( fullfileprefix + "speedupplot.png" )
    plt.show()

    X, Y = np.meshgrid(Nxvals, Nyvals)

    speedupnp = np.array( speedup )
    speedupnp = np.reshape( speedupnp, (len(Nxvals), len(Nyvals)) )

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, speedupnp, 50)
    ax.set_xlabel('Nx')
    ax.set_ylabel('Ny')
    ax.set_zlabel('Speedup')
    plt.savefig( fullfileprefix + "speedupcontour.png" )
    plt.show()

def problemSizeBench( root_dir, folderName, Nxvals, Nyvals ):

    Nxscatter = []
    Nyscatter = []

    # Nxvals = [4096]
    # Nyvals = [128]

    utilsfile = root_dir + "include/utils.hpp"
    fullfile = []

    seqdata = []
    pardata = []
    speedup = []
    i = 0

    for Nx in Nxvals:
        for Ny in Nyvals:
            print(i)
            i += 1

            Nxscatter.append( Nx )
            Nyscatter.append( Ny )

            with open( utilsfile ) as fval:

                fullfile = list( fval.readlines() )

                for idx, line in enumerate(fullfile):

                    if re.match("constexpr int Ny", line):

                        fullfile[idx] = "constexpr int Ny = " + str( Ny ) + ";\n"
    

                    elif re.match( "constexpr int Nx", line ):

                        fullfile[idx] = "constexpr int Nx = " + str( Nx ) + ";\n"

            
            with open( utilsfile, "w" ) as fval:
                fval.writelines(fullfile)

            outfile = root_dir + "build/txt/" + "out" + "Nx=" + str(Nx) + "Ny=" + str(Ny) + ".txt"
            runSim( root_dir, outfile, folderName )

            build_dir = root_dir + "/TextFiles/" + folderName + "/"

            fileval = build_dir + "timecalc" + "Nx=" + str(Nx) + "Ny=" + str(Ny) + ".txt"

            seqtime, partime, speedupval = getPerfData( fileval )

            seqdata.append( seqtime )
            pardata.append( partime )
            speedup.append( speedupval )

    # plotPerfDataContourScatter( root_dir, folderName, Nxscatter, Nyscatter, speedup, Nxvals, Nyvals )

    print( seqdata )
    print( pardata )
    print( speedup )  

def AOSvsSOA( root_dir ):

    # Nxvals = [ 512, 1024, 2048, 4096, 8192 ]
    # Nyvals = [ 16, 32, 64, 128, 256 ]

    # Nxvals = [2048, 4096, 8192 ]
    # Nyvals = [64, 128, 256 ]

    Nxvals = [ 512, 1024, 2048, 4096, 8192 ]
    Nyvals = [ 256 ]

    problemSizeBench( root_dir, "AOS", Nxvals, Nyvals )
    problemSizeBench( root_dir, "SOA", Nxvals, Nyvals )

    options = ["SOA", "AOS"]

    alldata = []
    speedupdata = []

    dirval = root_dir + "PlotFiles/"

    plt.figure()

    for idx, opt in enumerate( options ):

        perfdata = []

        for Nx in Nxvals:
            for Ny in Nyvals:

                build_dir = root_dir + "/TextFiles/" + opt + "/"

                fileval = build_dir + "timecalc" + "Nx=" + str(Nx) + "Ny=" + str(Ny) + ".txt"

                _, partime,_ = getPerfData( fileval )

                perfdata.append( partime )

        alldata.append( perfdata )

        if idx > 0:
            for j, timeval in enumerate( alldata[idx] ):
                speedupdata.append( timeval/alldata[0][j] )

        plt.plot( Nxvals, alldata[idx], "-o", label = opt )
        plt.xlabel( "Number of Nodes in X Dimension $(N_x)$" )
        plt.ylabel( "Time" )
    
    plt.legend()
    plt.savefig( dirval + "AOSvsSOA_Time.png" )

    plt.figure()
    plt.plot( Nxvals, speedupdata, "-o" )
    plt.xlabel( "Number of Nodes in X Dimension $(N_x)$" )
    plt.ylabel( "SpeedUp = AOS Time/SOA Time" )
    plt.savefig( dirval + "AOSvsSOA_Speedup.png" )

   

if __name__ == "__main__":

    print (os.getcwd())

    # plotvelocity( "../build/velocity.txt", "host" )
    # plotvelocity( "../build/velocitydevice.txt", "device" )

    root_dir = "/uufs/chpc.utah.edu/common/home/u1444601/CS6235/LBM_Project/"

    # plotvelocity( root_dir + "TextFiles/velocity_SOA.txt", root_dir + "PlotFiles/host_SOA" )
    # plotvelocity( root_dir + "TextFiles/velocitydevice_SOA.txt", root_dir + "PlotFiles/device_SOA" )

    # problemSizeBench( root_dir, "AOS" )

    AOSvsSOA( root_dir )

    # import matplotlib

    # print( matplotlib.__version__ )
    

