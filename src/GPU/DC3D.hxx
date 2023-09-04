#ifndef DC3D_H
#define DC3D_H

#include <sycl/sycl.hpp>

struct Tipo_1
{
    double dummy[5], sd, cd, sdsd, cdcd, sdcd, s2d, c2d;
};

struct Tipo_2
{
    double alp1, alp2, alp3, alp4, alp5;
    double sd, cd, sdsd, cdcd, sdcd, s2d, c2d;
};

struct Tipo_c2_1
{
    double xi2, et2, q2, r__, r2, r3, r5, y, d__, tt, alx, ale, x11, y11;
	double x32, y32, ey, ez, fy, fz, gy, gz, hy, hz;
};

SYCL_EXTERNAL
int dccon0_(double *alpha, double *dip, Tipo_2 *c0_2);

SYCL_EXTERNAL
int dccon2_(double *xi, double *et, double *q, 
            double *sd, double *cd, int *kxi, int *ket, Tipo_c2_1 *c2_1);

SYCL_EXTERNAL
int ua_(double *xi, double *et, double *q, double *disl1,
        double *disl2, double *disl3, double *u, Tipo_2 *c0_2, Tipo_c2_1 *c2_1);

SYCL_EXTERNAL
int ub_(double *xi, double *et, double *q, double *disl1,
        double *disl2, double *disl3, double *u, Tipo_2 *c0_2, Tipo_c2_1 *c2_1);

SYCL_EXTERNAL
int uc_(double *xi, double *et, double *q, double *z__,
        double *disl1, double *disl2, double *disl3, double *u, Tipo_2 *c0_2, Tipo_c2_1 *c2_1);

SYCL_EXTERNAL
int dc3d_(double *alpha, double *x, double *y, 
          double *z__, double *depth, double *dip, double *al1, 
          double *al2, double *aw1, double *aw2, double *disl1, 
          double *disl2, double *disl3, double *ux, double *uy, 
          double *uz, double *uxx, double *uyx, double *uzx, 
          double *uxy, double *uyy, double *uzy, double *uxz, 
          double *uyz, double *uzz, int *iret );
#endif

