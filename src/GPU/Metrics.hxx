#ifndef _METRICS_H_
#define _METRICS_H_

#include "Constantes.hxx"

SYCL_EXTERNAL
void inicializarEta1MaximaNivel0GPU(sycl::double2 *d_datosVolumenes_1, double *d_eta1_maxima,
                                    int num_volx, int num_voly, double epsilon_h,
                                    sycl::nd_item<2> item);

SYCL_EXTERNAL
void actualizarEta1MaximaNivel0GPU(sycl::double2 *d_datosVolumenes_1, double *d_eta1_maxima,
                                   int num_volx, int num_voly, double tiempo_act,
                                   double epsilon_h, sycl::nd_item<2> item);

SYCL_EXTERNAL
void inicializarTiemposLlegadaNivel0GPU(double *d_tiempos_llegada, int num_volx,
                                   int num_voly, sycl::nd_item<2> item);

SYCL_EXTERNAL
void actualizarTiemposLlegadaNivel0GPU(sycl::double2 *d_datosVolumenes_1, double *d_tiempos_llegada,
                                   double *d_eta1_inicial, int num_volx, int num_voly,
                                   double tiempo_act, double dif_at, sycl::nd_item<2> item);

#endif

