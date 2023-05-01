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

def runSim( root_dir, outfile, executableName, inargs ):

    build_dir = root_dir + "/build"

    compile_cmdlist = ["make"]
    cmdlist = ["./" + executableName, inargs]

    # open(outfile, 'w').close()

    subprocess.run(compile_cmdlist, cwd=build_dir  )

    subprocess.run(cmdlist, stdout=open(outfile, 'a'), stderr=open(outfile, 'a'), cwd=build_dir  )

def getPerfData( fileval ):

    # build_dir = root_dir + "/TextFiles"

    # fileval = build_dir + "/timecalc_SOA.txt"

    seqdata = 0
    pardata = 0
    speedup = 0
    seqmlups = 0
    parmlups = 0

    with open(fileval) as fval:

        alldata = fval.readlines()

        seqdata = float( alldata[ 0 ] ) 
        pardata = float( alldata[ 1 ] )
        speedup = float( alldata[ 2 ] )
        seqmlups = float( alldata[ 3 ] ) 
        parmlups = float( alldata[ 4 ] )

    return ( seqdata, pardata, speedup, seqmlups, parmlups )

def plotPerfDataContourScatter( root_dir, folderName, libraryName, Nxscatter, Nyscatter, speedup, Nxvals, Nyvals ):

    fullfileprefix = root_dir + "PlotFiles/" + libraryName + "/" + folderName + "/"

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

def problemSizeBench( root_dir, folderName, libraryName, Nxvals, Nyvals ):

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

            if libraryName == "CUDA":  
                runSim( root_dir, outfile, "lbm-gpu", folderName )
            elif libraryName == "SYCL":
                runSim( root_dir, outfile, "lbm-sycl", folderName )      

            build_dir = root_dir + "/TextFiles/" + libraryName + "/" + folderName + "/"

            fileval = build_dir + "timecalc" + "Nx=" + str(Nx) + "Ny=" + str(Ny) + ".txt"

            seqtime, partime, speedupval, _, _ = getPerfData( fileval )

            seqdata.append( seqtime )
            pardata.append( partime )
            speedup.append( speedupval )

    # plotPerfDataContourScatter( root_dir, folderName, libraryName, Nxscatter, Nyscatter, speedup, Nxvals, Nyvals )

    print( seqdata )
    print( pardata )
    print( speedup )  

def SYCLvsCUDABench( root_dir, option ):

    Nxvals = [ 512, 1024, 2048, 4096, 8192 ]
    Nyvals = [ 256 ]

    libraryNames = ["SYCL", "CUDA"]

    problemSizeBench( root_dir, "SOA", "SYCL", Nxvals, Nyvals )

    alldata = []
    speedupdata = []
    allmlupsdata = []

    dirval = root_dir + "PlotFiles/"

    for idx, library in enumerate( libraryNames ):

        perfdata = []
        mlupsdata = []

        for Nx in Nxvals:
            for Ny in Nyvals:

                build_dir = root_dir + "/TextFiles/" + library + "/" + option + "/"

                fileval = build_dir + "timecalc" + "Nx=" + str(Nx) + "Ny=" + str(Ny) + ".txt"

                _, partime, speedup, seqmlups, parmlups = getPerfData( fileval )

                perfdata.append( partime )
                mlupsdata.append( parmlups )

        alldata.append( perfdata )
        allmlupsdata.append( mlupsdata )

        if idx > 0:
            for j, timeval in enumerate( perfdata ):
                speedupdata.append( timeval/alldata[0][j] )

        plt.plot( Nxvals, mlupsdata, "-o", label = library )
        plt.xlabel( "Number of Nodes in X Dimension $(N_x)$" )
        plt.ylabel( "MLUPS" )

    fignameprefix = libraryNames[0] + "vs" + libraryNames[1]

    plt.legend()
    # plt.savefig( dirval + fignameprefix + "_Time.png" )
    plt.savefig( dirval + fignameprefix + "_MLUPS.png" )

    plt.figure()
    plt.plot( Nxvals, speedupdata, "-o" )
    plt.xlabel( "Number of Nodes in X Dimension $(N_x)$" )
    plt.ylabel( "SpeedUp" )
    plt.savefig( dirval + fignameprefix + "_Speedup.png" )


def compareOptions( root_dir, options ):

    # Nxvals = [ 512, 1024, 2048, 4096, 8192 ]
    # Nyvals = [ 16, 32, 64, 128, 256 ]

    # Nxvals = [2048, 4096, 8192 ]
    # Nyvals = [64, 128, 256 ]

    Nxvals = [ 512, 1024, 2048, 4096, 8192 ]
    Nyvals = [ 256 ]

    # problemSizeBench( root_dir, "AOS", Nxvals, Nyvals )
    # problemSizeBench( root_dir, "SOA", Nxvals, Nyvals )

    # options = ["SOA", "AOS"]

    alldata = []
    speedupdata = []
    allmlupsdata = []

    dirval = root_dir + "PlotFiles/CUDA/"

    plt.figure()

    for idx, opt in enumerate( options ):

        problemSizeBench( root_dir, opt, Nxvals, Nyvals )

        perfdata = []
        mlupsdata = []

        for Nx in Nxvals:
            for Ny in Nyvals:

                build_dir = root_dir + "/TextFiles/CUDA/" + opt + "/"

                fileval = build_dir + "timecalc" + "Nx=" + str(Nx) + "Ny=" + str(Ny) + ".txt"

                _, partime, speedup, seqmlups, parmlups = getPerfData( fileval )

                perfdata.append( partime )
                mlupsdata.append( parmlups )

        alldata.append( perfdata )
        allmlupsdata.append( mlupsdata )

        if idx > 0:
            for j, timeval in enumerate( perfdata ):
                speedupdata.append( timeval/alldata[0][j] )

        # plt.plot( Nxvals, alldata[idx], "-o", label = opt )
        # plt.xlabel( "Number of Nodes in X Dimension $(N_x)$" )
        # plt.ylabel( "Time" )

        plt.plot( Nxvals, mlupsdata, "-o", label = opt )
        plt.xlabel( "Number of Nodes in X Dimension $(N_x)$" )
        plt.ylabel( "MLUPS" )
    
    fignameprefix = options[0] + "vs" + options[1]

    plt.legend()
    # plt.savefig( dirval + fignameprefix + "_Time.png" )
    plt.savefig( dirval + fignameprefix + "_MLUPS.png" )

    plt.figure()
    plt.plot( Nxvals, speedupdata, "-o" )
    plt.xlabel( "Number of Nodes in X Dimension $(N_x)$" )
    plt.ylabel( "SpeedUp" )
    plt.savefig( dirval + fignameprefix + "_Speedup.png" )


if __name__ == "__main__":

    print (os.getcwd())

    # plotvelocity( "../build/velocity.txt", "host" )
    # plotvelocity( "../build/velocitydevice.txt", "device" )

    root_dir = "/uufs/chpc.utah.edu/common/home/u1444601/CS6235/LBM_Project/"

    # plotvelocity( root_dir + "TextFiles/velocity_SOA.txt", root_dir + "PlotFiles/host_SOA" )
    # plotvelocity( root_dir + "TextFiles/velocitydevice_SOA.txt", root_dir + "PlotFiles/device_SOA" )

    # problemSizeBench( root_dir, "AOS" )

    options = ["SOA", "AOS"]
    # compareOptions( root_dir, options )

    SYCLvsCUDABench( root_dir, "SOA" )

    # import matplotlib

    # print( matplotlib.__version__ )
    

