#ifndef _DEFORMACION_H_
#define _DEFORMACION_H_

#include <sycl/sycl.hpp>

SYCL_EXTERNAL
void aplicarOkadaStandardGPU( sycl::double2 *d_datosVolumenes_1, int num_volx, int num_voly, double lon_ini, double incx,
    double lat_ini, double incy, double LON_C_ent, double LAT_C_ent, double DEPTH_C_ent, double FAULT_L,
    double FAULT_W, double STRIKE, double DIP_ent, double RAKE, double SLIP, double H,
    sycl::nd_item<2> item );

#endif

