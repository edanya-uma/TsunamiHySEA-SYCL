#ifndef _VOLUMEN_KERNEL_H_
#define _VOLUMEN_KERNEL_H_

#include "Constantes.hxx"

SYCL_EXTERNAL
void spongeLayerPaso1(double *h, double dist, double ls, double h0);

SYCL_EXTERNAL
void spongeLayerPaso2(double *q, double dist, double ls, double h0, double v0);

SYCL_EXTERNAL
void disImplicita( sycl::double2  Want1, sycl::double2 Want2,
                   sycl::double2 *acum1, sycl::double2 *acum2, double delta_T,
                   double mf0, double epsilon_h );

SYCL_EXTERNAL
void filtroEstado( sycl::double2 *acum1, sycl::double2 *acum2, double vmax,
                   double delta_T, double epsilon_h );

SYCL_EXTERNAL
void inicializarEta1MaximaNivel0GPU(sycl::double2 *d_datosVolumenes_1,
                                    double *d_eta1_maxima, int num_volx,
                                    int num_voly, double epsilon_h,
                                    sycl::nd_item<2> item );

SYCL_EXTERNAL
void actualizarEta1MaximaNivel0GPU(sycl::double2 *d_datosVolumenes_1,
                                   double *d_eta1_maxima, int num_volx,
                                   int num_voly, double tiempo_act,
                                   double epsilon_h,
                                   sycl::nd_item<2> item );

SYCL_EXTERNAL
void obtenerDeltaTVolumenesGPU(sycl::double2 *d_datosVolumenesNivel0_3,
                               sycl::double2 *d_acumulador_1,
                               double *d_deltaTVolumenes, int numVolxNivel0,
                               int numVolyNivel0, float CFL,
                               sycl::nd_item<2> item );

SYCL_EXTERNAL
void obtenerEstadosPaso1Nivel0GPU(
    sycl::double2 *d_datosVolumenes_1, sycl::double2 *d_datosVolumenes_2,
    sycl::double2 *d_datosVolumenes_3, double *d_anchoVolumenes,
    double *d_altoVolumenes, sycl::double2 *d_acumulador_1, int num_volx,
    int num_voly, double delta_T, double Hmin, int tam_spongeSup,
    int tam_spongeInf, int tam_spongeIzq, int tam_spongeDer, double sea_level,
    sycl::nd_item<2> item );

SYCL_EXTERNAL
void obtenerEstadoYDeltaTVolumenesNivel0GPU(
    sycl::double2 *d_datosVolumenes_1, sycl::double2 *d_datosVolumenes_2,
    sycl::double2 *d_datosVolumenes_3, double *d_anchoVolumenes,
    double *d_altoVolumenes, sycl::double2 *d_acumulador_1,
    sycl::double2 *d_acumulador_2, double *d_deltaTVolumenes, int num_volx,
    int num_voly, double CFL, double delta_T, double mf0, double vmax,
    double hpos, double epsilon_h, double Hmin, int tam_spongeSup,
    int tam_spongeInf, int tam_spongeIzq, int tam_spongeDer, double sea_level,
    sycl::nd_item<2> item_ct1);

SYCL_EXTERNAL
void escribirVolumenesGuardadoGPU(sycl::double2 *d_datosVolumenes_1,
                                  sycl::double2 *d_datosVolumenes_2,
                                  sycl::double2 *d_datosVolumenesGuardado_1,
                                  sycl::double2 *d_datosVolumenesGuardado_2,
                                  int *d_posicionesVolumenesGuardado,
                                  int num_puntos_guardar, double epsilon_h,
                                  sycl::nd_item<1> item);
#endif

