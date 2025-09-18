#ifndef NETCDF_KERNELS_H
#define NETCDF_KERNELS_H

#include "Constantes.hxx"

SYCL_EXTERNAL
void obtenerEtaNetCDF(sycl::double2 *d_datosVolumenes_1, float *d_vec, int num_volx, int num_voly,
                                   double epsilon_h, double Hmin, double H, sycl::nd_item<2> item);

SYCL_EXTERNAL
void obtenerUxNetCDF(sycl::double2 *d_datosVolumenes_1, sycl::double2 *d_datosVolumenes_2, float *d_vec,
                                   int num_volx, int num_voly, double epsilon_h, double Hmin, double H,
                                   double Q, sycl::nd_item<2> item);

SYCL_EXTERNAL
void obtenerUyNetCDF(sycl::double2 *d_datosVolumenes_1, sycl::double2 *d_datosVolumenes_2, float *d_vec,
                                   int num_volx, int num_voly, double epsilon_h, double Hmin, double H,
                                   double Q, sycl::nd_item<2> item);

SYCL_EXTERNAL
void obtenerBatimetriaModificadaNetCDF(sycl::double2 *d_datosVolumenes_1, float *d_vec, int num_volx,
                                   int num_voly, double Hmin, double H, sycl::nd_item<2> item);

SYCL_EXTERNAL
void obtenerEta1MaximaNetCDF(double *d_eta1_maxima, float *d_vec, int num_volx, int num_voly,
                                   double Hmin, double H, sycl::nd_item<2> item);

SYCL_EXTERNAL
void obtenerTiemposLlegadaNetCDF(double *d_tiempos_llegada, float *d_vec, int num_volx, int num_voly,
                                   double T, sycl::nd_item<2> item);

#endif

