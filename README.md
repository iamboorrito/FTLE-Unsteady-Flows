# FTLE Unsteady Flows
An implementation of the FTLE algorithm. To compile the C files, you need OpenMP and gcc. I compile them on MacOS using

> gcc-8 -fopenmp -o flow flow_map.c

and use Matlab to run the .m script. There is also a standalone C file called ftle.c which computes the flow map and finite time Lyapunov exponent field and outputs them both as .ppm and .csv files.

[Double Gyre made by standalone]("Generated Images"/"Double Gyre"/ftle.png)
