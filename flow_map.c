//  
//  I compile with >> gcc-8 -fopenmp ftle.c -o ftle
//  since I'm running on Mac with gcc-8 being my gcc which is
//  installed by brew. Need OpenMP to get parallel code to work.
//
//  run with >> ./ftle x0 xend y0 yend t0 tend sizex sizey
// 
//  where the spacial variables define a bounding box [x0, x1]x[y0, y1]
//  and [t0, t1] define the time interval to compute the flow map over.
//
//  If using over an interval outside of [-200, 200]x[-200, 200], you will
//  need to change the #define variables to extend the trajectory cutoff
//  window as necessary.
//
//  Meant to be used with Matlab as
//
//  % compute flow map with flow executable
//  [status, result] = system(sprintf('./flow %f %f %f %f %f %f %d %d',...
//   x0, x1, y0, y1, t0, t1, numx, numy));
//
//  % evaluate expression for stacked flow maps 
//  flows = eval(result);
//  flow_mapx = flows(1:numx, :)'; 
//  flow_mapy = flows(1+numx:end, :)';
//
//  ftle.c
//  flow_map
//
//  Created by Evan Burton on 12/4/18.
//  Copyright © 2018 Evan Burton. All rights reserved.
//
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>

#define xubound 200
#define xlbound -200
#define yubound 200
#define ylbound -200

double A = 0.1;
double ep = 0.1;
double w = M_PI/5;

// The x velocity component x' = f1(t, x, y)
double f1(double t, double x, double y){
    
    double g = x*(ep*sin(w*t)*x + (1 - 2*ep*sin(w*t)));
    return -A*M_PI*sin(M_PI*g)*cos(M_PI*y);
    
    //return -A*M_PI*sin(M_PI*x)*cos(M_PI*y);

    //return y;
}

// The y velocity component y' = f2(t, x, y)
double f2(double t, double x, double y){

    
    double g = x*(ep*sin(w*t)*x + (1 - 2*ep*sin(w*t)));
    double dg = 1 + (2*x - ep)*ep*sin(w*t);
    return A*M_PI*cos(M_PI*g)*sin(M_PI*y)*dg;
    
    //return A*M_PI*cos(M_PI*x)*sin(M_PI*y);
    
    //return sin(x*t);
}

int offset(int, int, int, int, int);
void rk4(double a, double b, double h, double x0, double y0, double* fval);


int main(int argc, const char * argv[]) {
    
    double x0 = atof(argv[1]);
    double xend = atof(argv[2]);
    
    double y0 = atof(argv[3]);
    double yend = atof(argv[4]);
    
    double t0 = atof(argv[5]);
    double tend = atof(argv[6]);

    unsigned int numx = atof(argv[7]);
    unsigned int numy = atof(argv[8]);

    //double flow_map[numx][numy][2];
    //double exps[numx][numy];

    double* flow_map;
    flow_map = malloc(numx*numy*2 * sizeof(double)); 

    double xs[numx];
    double ys[numy];

    // time step for rk4
    double dt = 0.01;

    if (tend < t0){
        dt = -dt;
    }

    double dx = (xend-x0)/((double)numx-1);
    double dy = (yend-y0)/((double)numy-1);

    #pragma omp parallel
    {
        #pragma omp for
        for(int i = 0; i < numx; i++)
            xs[i] = x0 + i*dx;

        #pragma omp for
        for(int j = 0; j < numy; j++)
            ys[j] = y0 + j*dy;
    }

    #pragma omp parallel for schedule(guided) shared(xs, ys, t0, tend, dt, flow_map)
    for(int i = 0; i < numx; i++){
        for(int j = 0; j < numy; j++){

            double fval[2];
            
            rk4(t0, tend, dt, xs[i], ys[j], fval);

            flow_map[offset(i, j, 0, numx, numy)] = fval[0];
            flow_map[offset(i, j, 1, numx, numy)] = fval[1];
        }
        //printf("%d / %d\n", i, numx);
    }

    printf("[");
    for (int k = 0; k < 2; k++){
        for (int i=0; i < numx; i++) {
            for (int j=0; j < numy-1; j++) {
                printf("%f,", flow_map[offset(i, j, k, numx, numy)]);
            }
            printf("%f\n", flow_map[offset(i, numy-1, k, numx, numy)]);
        }
    }
    printf("]");

    free(flow_map);
    return 0;
}

int offset(int x, int y, int z, int numx, int numy) { 
    return (z * numx * numy) + (y * numx) + x; 
}

void rk4(double a, double b, double h, double x0, double y0, double* fval){

    // Get number of points
    int n = fabs((b-a)/h) + 1;
    double xi = x0;
    double yi = y0;
    double t = a;

    double k1, k2, k3, k4, l1, l2, l3, l4;
    
    for(int i = 0; i < n; i++){

        // RK4 Scheme
        k1 = h*f1(t, xi, yi);
        l1 = h*f2(t, xi, yi);
        
        k2 = h*f1(t + h/2.0, xi + k1/2.0, yi + l1/2);
        l2 = h*f2(t + h/2.0, xi + k1/2.0, yi + l1/2);
        
        k3 = h*f1(t + h/2.0, xi + k2/2.0, yi + l2/2);
        l3 = h*f2(t + h/2.0, xi + k2/2.0, yi + l2/2);
        
        k4 = h*f1(t+h, xi+k3, yi + l3);
        l4 = h*f2(t+h, xi+k3, yi + l3);

        xi = xi + (k1+2*(k2+k3)+k4)/6.0;
        yi = yi + (l1+2*(l2+l3)+l4)/6.0;

        // Ensure spacial variables do not leave bounding boxs
        if (xi > xubound || xi < xlbound || yi > yubound || yi < xlbound){
                fval[0] = xi;
                fval[1] = yi;
                return;
        }

        t += h;
    }

    fval[0] = xi;
    fval[1] = yi;
    
}
