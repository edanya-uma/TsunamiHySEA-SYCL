#ifndef DEFORMACION_H
#define DEFORMACION_H

#include <sycl/sycl.hpp>

SYCL_EXTERNAL
void convertirAMenosValorAbsolutoGPU(double *d_in, float *d_out, int n, sycl::nd_item<1> item);

SYCL_EXTERNAL
void truncarDeformacionGPU(double *d_def, int num_volx, int num_voly, double crop_value, sycl::nd_item<2> item);

SYCL_EXTERNAL
void sumarDeformacionADatosGPU(sycl::double2 *d_datosVolumenes_1, double *d_def, double *d_eta1Inicial,
        int num_volx, int num_voly, sycl::nd_item<2> item);

SYCL_EXTERNAL
void aplicarOkadaStandardGPU(sycl::double2 *d_datosVolumenes_1, double *d_def, int num_volx, int num_voly, double lon_ini,
        double incx, double lat_ini, double incy, double LON_C_ent, double LAT_C_ent, double DEPTH_C_ent, double FAULT_L,
        double FAULT_W, double STRIKE, double DIP_ent, double RAKE, double SLIP, double H, sycl::nd_item<2> item);

void aplicarOkada( sycl::queue &q, sycl::event &ev, sycl::double2 *d_datosVolumenes_1, double *d_eta1Inicial, int crop_flag,
		double crop_value, double *d_deltaTVolumenes, float *d_vec, int num_volx, int num_voly, double lon_ini, double incx,
		double lat_ini, double incy, double LON_C, double LAT_C, double DEPTH_C, double FAULT_L, double FAULT_W, double STRIKE,
		double DIP, double RAKE, double SLIP, sycl::range<2> blockGridEstNivel, sycl::range<2> threadBlockEst, double H);

#endif

