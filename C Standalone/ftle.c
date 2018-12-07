//  compile with >> gcc-8 -fopenmp ftle.c -o ftle
//  run with >> ./ftle x0 xend y0 yend t0 tend sizex sizey
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

#define xubound 100
#define xlbound -100
#define yubound 100
#define ylbound -100

double A = 0.1;
double ep = 0.1;
double w = M_PI/5;

// Vector field x-component x' = f1(t, x, y)
double f1(double t, double x, double y){
    double g = x*(ep*sin(w*t)*x + (1 - 2*ep*sin(w*t)));
    return -A*M_PI*sin(M_PI*g)*cos(M_PI*y);
}

// Vector field y-component y' = f2(t, x, y)
double f2(double t, double x, double y){
    double g = x*(ep*sin(w*t)*x + (1 - 2*ep*sin(w*t)));
    double dg = 1 + (2*x - ep)*ep*sin(w*t);
    return A*M_PI*cos(M_PI*g)*sin(M_PI*y)*dg;
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

    double* exps;
    exps = malloc(numx*numy * sizeof(double)); 

    double xs[numx];
    double ys[numy];

    printf("Computing ftle for t in [%.2f, %.2f]\n", t0, tend);
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

    double max_exp = 0;
    double max_flowx = 0;
    double max_flowy = 0;
    //double fx1, fx2, fy1, fy2, a11, a12, a22;

    printf("Computing flow map\n");
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

    for(int i = 0; i < numx; i++){
        for(int j = 0; j < numy; j++){

            unsigned int indx1 = offset(i, j, 0, numx, numy);
            unsigned int indx2 = offset(i, j, 1, numx, numy);

            if (flow_map[indx1] > max_flowx)
                max_flowx = flow_map[indx1];

            if (flow_map[indx2] > max_flowy)
                max_flowy = flow_map[indx2];
        }
    }

    #pragma omp barrier
    double deltaT = fabs(tend - t0);

    printf("Computing FTLE\n");
    #pragma omp parallel for schedule(guided) shared(exps)
    for(int i = 0; i < numx; i++){
        for(int j = 0; j < numy; j++){
            if (i > 0 && i <= numx-2 && j > 0 && j <= numy-2){
                // Derivative matrix Dphi
                double fx1 = (flow_map[offset(i+1, j, 0, numx, numy)] - flow_map[offset(i-1, j, 0, numx, numy)])/(2*dx);
                double fx2 = (flow_map[offset(i+1, j, 1, numx, numy)] - flow_map[offset(i-1, j, 1, numx, numy)])/(2*dx);
                double fy1 = (flow_map[offset(i, j+1, 0, numx, numy)] - flow_map[offset(i, j-1, 0, numx, numy)])/(2*dy);
                double fy2 = (flow_map[offset(i, j+1, 1, numx, numy)] - flow_map[offset(i, j+1, 1, numx, numy)])/(2*dy);
                
                // Find max eigenvalue of Dphi' Dphi
                // [fx1 fx2] [fx1 fy1] = [fx1^2+fx2^2, fx1*fy1 + fx2*fy2]
                // [fy1 fy2] [fx2 fy2] = [ a12       , fy1^2 + fy2^2    ]

                // Compute entries of Cauchy-Green Strain Matrix
                double a11 = fx1*fx1 + fx2*fx2;
                double a12 = fx1*fy1 + fx2*fy2;
                double a22 = fy1*fy1 + fy2*fy2;

                // Reuse fy1 as trace, fy2 as det
                double tr = a11 + a22;
                double det = a11*a22 - a12*a12;
                // lambda1 = 1/2 * (tr + sqrt(tr^2 - 4*det))
                // norm(A) = sqrt(lmax(A'A))
                double lambda = sqrt((tr + sqrt(tr*tr - 4*det))/2);
                double ftle = log(lambda)/deltaT;
                unsigned int indx = offset(i, j, 0, numx, numy);

                exps[indx] = ftle;
                
                // Save for coloring picture
                if (ftle > max_exp)
                    max_exp = ftle;
            
            }else{
                exps[offset(i, j, 0, numx, numy)] = 0;
            }

        }
    }
    
    printf("Writing files\n");

    mkdir("ftle_gen", 0777);
    mkdir("ftle_gen/data", 0777);
    FILE *ftle = fopen("ftle_gen/ftle.ppm", "wb");
    FILE* flowx = fopen("ftle_gen/flow_x.ppm", "wb");
    FILE *flowy = fopen("ftle_gen/flow_y.ppm", "wb");

    FILE *ftledata = fopen("ftle_gen/data/ftle.csv", "wb");
    FILE *flowxdata = fopen("ftle_gen/data/flowx.csv", "wb");
    FILE *flowydata = fopen("ftle_gen/data/flowy.csv", "wb");

    fprintf(ftle, "P6\n%i %i 255\n", numx, numy);
    fprintf(flowy, "P6\n%i %i 255\n", numx, numy);
    fprintf(flowx, "P6\n%i %i 255\n", numx, numy);

    int color;
    for (int j=numy-1; j >=0; j--) {
        for (int i=0; i < numx; i++) {
            
            color = (int)(255*exps[offset(i, j, 0, numx, numy)]/max_exp);
            if (color < 0){
                fputc(0, ftle);   // 0 .. 255
                fputc(0, ftle); // 0 .. 255
                fputc(-color, ftle);  // 0 .. 255
            }else{
                fputc(color, ftle);   // 0 .. 255
                fputc(0, ftle); // 0 .. 255
                fputc(0, ftle);  // 0 .. 255
            }

            color = (int)(255*flow_map[offset(i, j, 0, numx, numy)]/max_flowx);
            if (color < 0){
                fputc(0, flowx);   // 0 .. 255
                fputc(0, flowx); // 0 .. 255
                fputc(-color, flowx);  // 0 .. 255
            }else{
                fputc(color, flowx);   // 0 .. 255
                fputc(0, flowx); // 0 .. 255
                fputc(0, flowx);  // 0 .. 255
            }

            color = (int)(255*flow_map[offset(i, j, 1, numx, numy)]/max_flowy);
            if (color < 0){
                fputc(0, flowy);   // 0 .. 255
                fputc(0, flowy); // 0 .. 255
                fputc(-color, flowy);  // 0 .. 255
            }else{
                fputc(color, flowy);   // 0 .. 255
                fputc(0, flowy); // 0 .. 255
                fputc(0, flowy);  // 0 .. 255
            }

            fprintf(ftledata, "%f,", exps[offset(i, j, 0, numx, numy)]);
            fprintf(flowxdata, "%f,", flow_map[offset(i, j, 0, numx, numy)]);
            fprintf(flowydata, "%f,", flow_map[offset(i, j, 1, numx, numy)]);
        }

            //fprintf(ftledata, "\n");
            //fprintf(flowxdata, "\n");
            //fprintf(flowydata, "\n");
 
    }

    fclose(ftle);
    fclose(flowx);
    fclose(flowy);

    fclose(ftledata);
    fclose(flowxdata);
    fclose(flowydata);

    free(flow_map);
    free(exps);

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
