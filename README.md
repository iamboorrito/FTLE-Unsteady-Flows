# FTLE Unsteady Flows
A (mostly) poorly written implementation of the FTLE algorithm for my dynamical systems final project. To compile the C files, you need OpenMP and gcc. I compile them on MacOS using

> gcc-8 -fopenmp -o flow flow_map.c

and use Matlab to run the .m script. There is also a standalone C file called ftle.c which computes the flow map and finite time Lyapunov exponent field and outputs them both as .ppm and .csv files.

![Double Gyre made by standalone](Generated%20Images/Double%20Gyre/ftle3.png)

![Unsteady Pendulum](Generated%20Images/Unsteady%20Pendulum%20Flow/augpend.png)

![Sinusoidal flow by standalone](Generated%20Images/sinusoidal_ftle.png)
