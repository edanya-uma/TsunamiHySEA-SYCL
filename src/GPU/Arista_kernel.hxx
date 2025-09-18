#ifndef ARISTA_KERNEL_H
#define ARISTA_KERNEL_H

#include "Constantes.hxx"
#include <stdio.h>

SYCL_EXTERNAL int obtenerPosXVer(int ppv, bool es_periodica, int num_volx);

SYCL_EXTERNAL double limitador(double p0m, double p0, double p1, double p1p, double S);

SYCL_EXTERNAL double limitador_monotono(double p0m, double p0, double p1, double p1p, double S);

SYCL_EXTERNAL void tratamientoSecoMojado(double h0, double h1, double eta0, double eta1, double H0, double H1,
              double epsilon_h, double *q0n, double *q1n);

SYCL_EXTERNAL void positividad(double h0, double h1, double delta_T, double area0, double area1, double *Fmenos);

SYCL_EXTERNAL void positividad_H(double h0, double h1, double cosPhi0, double cosPhi1, double delta_T,
              double area0, double area1, double *Fmenos);

// ARISTAS VERTICALES
SYCL_EXTERNAL void procesarAristaVer_2s_1step(double h0, double h1, double q0x, double longitud, double area0,
              double area1, double delta_T, sycl::double2 *d_acumulador_1, int pos_vol0, int pos_vol1, double cosPhi0,
              double *Fmas2s, double *Fmenos2s);

SYCL_EXTERNAL void procesarAristaVer_2s_waf_1step(double h0m, double h0, double h1, double h1p, double q0x, double q1x,
              double H0m, double H0, double H1, double H1p, double normal, double longitud, double area0, double area1,
              double delta_T, sycl::double2 *d_acumulador_1, int pos_vol0, int pos_vol1, double epsilon_h, double cosPhi0,
              double hpos, double *Fmas2s, double *Fmenos2s, bool alfa_cero);

SYCL_EXTERNAL void procesarAristaVer_2s_2step(sycl::double3 *W0, sycl::double3 *W1, double H0, double H1,
              sycl::double2 *d_acumulador_1, sycl::double2 *d_acumulador_2, int pos_vol0, int pos_vol1,
              double CFL, double cosPhi0, double longitud, double delta_T, double area0, double area1,
              double cvis, sycl::double2 *Fmas2s, sycl::double2 *Fmenos2s);

SYCL_EXTERNAL void procesarAristaVer_2s_waf_2step(sycl::double3 *W0m, sycl::double3 *W0, sycl::double3 *W1,
              sycl::double3 *W1p, double H0m, double H0, double H1, double H1p, double normal, double longitud,
              double area0, double area1, double delta_T, sycl::double2 *d_acumulador_1, sycl::double2 *d_acumulador_2,
              int pos_vol0, int pos_vol1, double CFL, double epsilon_h, double cosPhi0, double cvis, double hpos,
              sycl::double2 *Fmas2, sycl::double2 *Fmenos2, bool alfa_cero);

SYCL_EXTERNAL void procesarAristasVerNivel0Paso1GPU( sycl::nd_item<2> idx,
              sycl::double2 *d_datosVolumenes_1, sycl::double2 *d_datosVolumenes_2,
              sycl::double2 *d_datosVolumenes_3, double *d_altoVolumenes, int num_volx, int num_voly,
              double borde1, double borde2, bool es_periodica, double delta_T, sycl::double2 *d_acumulador_1,
              double CFL, double epsilon_h, double hpos, double cvis, int tipo);

SYCL_EXTERNAL void procesarAristasVerNivel0Paso2GPU( sycl::nd_item<2> idx,
              sycl::double2 *d_datosVolumenes_1, sycl::double2 *d_datosVolumenes_2,
              sycl::double2 *d_datosVolumenes_3, double *d_altoVolumenes, int num_volx, int num_voly,
              double borde1, double borde2, bool es_periodica, double delta_T, sycl::double2 *d_acumulador_1,
              sycl::double2 *d_acumulador_2, double CFL, double epsilon_h, double hpos, double cvis, int tipo);

// ARISTAS HORIZONTALES
SYCL_EXTERNAL void procesarAristaHor_2s_1step(double h0, double h1, double q0y, double longitud, double area0,
              double area1, double delta_T, sycl::double2 *d_acumulador_1, int pos_vol0, int pos_vol1, double cosPhi0,
              double cosPhi1, double *Fmas2s, double *Fmenos2s);

SYCL_EXTERNAL void procesarAristaHor_2s_waf_1step(double h0m, double h0, double h1, double h1p, double q0y, double q1y,
              double H0m, double H0, double H1, double H1p, double normal, double longitud, double area0, double area1,
              double delta_T, sycl::double2 *d_acumulador_1, int pos_vol0, int pos_vol1, double epsilon_h, double cosPhi0,
              double cosPhi1, double hpos, double *Fmas2s, double *Fmenos2s, bool alfa_cero);

SYCL_EXTERNAL void procesarAristaHor_2s_2step(sycl::double3 *W0, sycl::double3 *W1, double H0, double H1, double longitud,
              double area0, double area1, double delta_T, sycl::double2 *d_acumulador_1, sycl::double2 *d_acumulador_2,
              int pos_vol0, int pos_vol1, double CFL, double epsilon_h, double cosPhi0, double cosPhi1,
              double cvis, double hpos, sycl::double2 *Fmas2s, sycl::double2 *Fmenos2s);

SYCL_EXTERNAL void procesarAristaHor_2s_waf_2step(sycl::double3 *W0m, sycl::double3 *W0, sycl::double3 *W1, sycl::double3 *W1p,
              double H0m, double H0, double H1, double H1p, double normal, double longitud, double area0, double area1,
              double delta_T, sycl::double2 *d_acumulador_1, sycl::double2 *d_acumulador_2, int pos_vol0, int pos_vol1,
              double CFL, double epsilon_h, double cosPhi0, double cosPhi1, double cvis, double hpos,
              sycl::double2 *Fmas2, sycl::double2 *Fmenos2, bool alfa_cero);

SYCL_EXTERNAL void procesarAristasHorNivel0Paso1GPU( sycl::nd_item<2> idx,
              sycl::double2 *d_datosVolumenes_1, sycl::double2 *d_datosVolumenes_2,
              sycl::double2 *d_datosVolumenes_3, double *d_anchoVolumenes, int num_volx, int num_voly,
              double borde1, double borde2, double delta_T, sycl::double2 *d_acumulador_1,
              double CFL, double epsilon_h, double hpos, double cvis, int tipo);

SYCL_EXTERNAL void procesarAristasHorNivel0Paso2GPU( sycl::nd_item<2> idx,
              sycl::double2 *d_datosVolumenes_1, sycl::double2 *d_datosVolumenes_2,
              sycl::double2 *d_datosVolumenes_3, double *d_anchoVolumenes, int num_volx, int num_voly,
              double borde1, double borde2, double delta_T, sycl::double2 *d_acumulador_1, sycl::double2 *d_acumulador_2,
              double CFL, double epsilon_h, double hpos, double cvis, int tipo);

// DELTA_T INICIAL
SYCL_EXTERNAL void procesarAristaDeltaTInicial(sycl::double3 *W0, sycl::double3 *W1, double H0, double H1, double normal1_x,
              double normal1_y, double longitud, sycl::double2 *d_acumulador_2, int pos_vol0, int pos_vol1, double epsilon_h,
              double cosPhi0);

SYCL_EXTERNAL void procesarAristasVerDeltaTInicialNivel0GPU( sycl::nd_item<2> idx,
              sycl::double2 *d_datosVolumenesNivel0_1, sycl::double2 *d_datosVolumenesNivel0_2,
              sycl::double2 *d_datosVolumenes_3, int numVolxNivel0, int numVolyNivel0, double borde1, double borde2,
              bool es_periodica, double *d_altoVolumenesNivel0, sycl::double2 *d_acumulador_1, double epsilon_h, int tipo);

SYCL_EXTERNAL void procesarAristasHorDeltaTInicialNivel0GPU( sycl::nd_item<2> idx,
              sycl::double2 *d_datosVolumenesNivel0_1, sycl::double2 *d_datosVolumenesNivel0_2,
              int numVolxNivel0, int numVolyNivel0, double borde1, double borde2, double *d_anchoVolumenesNivel0,
              sycl::double2 *d_acumulador_1, double epsilon_h, int tipo);

#endif

