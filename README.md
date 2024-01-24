

## Installing CombBLAS  ##

You need to build and install CombBLAS in order to run the driver SpGEMM program.

Instructions for doing this can be found here https://github.com/PASSIONLab/CombBLAS

The makefile for the driver program has some hard-coded filepaths in it, so to ensure that it works, please make sure to do the following:
 - Create a directory called `install` in the main CombBLAS directory
 - When building CombBLAS, include the following cmake argument: `-DCMAKE_INSTALL_PREFIX=/path/to/CombBLAS/install`


## Building the driver program ##

Please ensure you have `mpicxx` in your `PATH` before trying to build the driver program

Please also ensure the CombBLAS directory lives in the same parent directory as the `spgemm` directory (this should be the case by default, you shouldn't have to change anything)

cd into the `spgemm` directory and type `make`. If CombBLAS is installed in the correct location, this step should proceed without issue. 


## Running the driver program ##

To have the driver program execute CombBLAS's 3D SpGEMM routine, run `mpirun <MPI-ARGS> ./combblas-spgemm 3D /path/to/input/matrix 1 <LAYERS>`

The path to the input matrix should be a path to a matrix market file. The `LAYERS` argument controls how many layers are in the processor grid. The driver program will print the runtime averaged over 4 iterations. Use whatever profiling program you'd like in order to get more detailed measurements. 


