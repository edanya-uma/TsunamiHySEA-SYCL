#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cstring>
#include <fstream>
#include <string>
#include <sycl/sycl.hpp>
#include "stopwatch.hxx"
#include "Volumen_kernel.hxx"
#include "Arista_kernel.hxx"
#include "Deformacion.hxx"
#include "netcdf.hxx"
#include "netcdfSeries.hxx"

using namespace sycl;

/*****************/
/* NetCDF saving */
/*****************/

void guardarNetCDFNivel0(double2 *datosVolumenes_1, double2 *datosVolumenes_2, float *vec, int num, int num_volx,
                int num_voly, double Hmin, double tiempo_act, double epsilon_h, double H, double Q, double T)
{
    double tiempo = tiempo_act*T;
    double2 datos;
    double h;
    double maxheps, factor;
    int i;
    int num_volumenes = num_volx*num_voly;

    escribirTiempoNC(num, tiempo);
    for (i=0; i<num_volumenes; i++) {
        datos = datosVolumenes_1[i];
        vec[i] = (float) ( (datos.x() < epsilon_h) ? -9999.0 : (datos.x() - datos.y() - Hmin)*H );
    }
    escribirEtaNC(num_volx, num_voly, num, vec);
    for (i=0; i<num_volumenes; i++) {
        h = datosVolumenes_1[i].x();
        if (h < epsilon_h) {
            maxheps = epsilon_h;
            factor = sqrt2*h / sqrt(h*h*h*h + maxheps*maxheps*maxheps*maxheps)*(Q/H);
        }
        else {
            factor = Q/(h*H);
        }
        vec[i] = (float) (datosVolumenes_2[i].x()*factor);
    }
    escribirUxNC(num_volx, num_voly, num, vec);
    for (i=0; i<num_volumenes; i++) {
        h = datosVolumenes_1[i].x();
        if (h < epsilon_h) {
            maxheps = epsilon_h;
            factor = sqrt2*h / sqrt(h*h*h*h + maxheps*maxheps*maxheps*maxheps)*(Q/H);
        }
        else {
            factor = Q/(h*H);
        }
        vec[i] = (float) (datosVolumenes_2[i].y()*factor);
    }
    escribirUyNC(num_volx, num_voly, num, vec);
}

/***************/
/* Time series */
/***************/

// Format of the NetCDF time series file:
// For each point:
//   <longitude>
//   <latitude>
//   <bathymetry (original if okada_flag is INITIAL_FROM_FILE; deformed if okada_flag is OKADA_STANDARD)
//   <minimum eta>
//   <maximum eta>
// For each time:
//   For each point:
//     <eta point 1> <u point 1> <v point 1> ... <eta point n> <u point n> <v point n>
void obtenerBatimetriaParaSerieTiempos(double2 *datosVolumenesNivel_1, int numPuntosGuardar,
        int *posicionesVolumenesGuardado, float *profPuntosGuardado, double Hmin, double H)
{
    int i;

    for (i=0; i<numPuntosGuardar; i++) {
        if (posicionesVolumenesGuardado[i] != -1)
            profPuntosGuardado[i] = (float) ((datosVolumenesNivel_1[i].y() + Hmin)*H);
    }
}

void guardarSerieTiemposNivel0(double2 *datosVolumenesNivel_1, double2 *datosVolumenesNivel_2,
            int numPuntosGuardar, int *posicionesVolumenesGuardado, float *etaPuntosGuardado,
            float *uPuntosGuardado, float *vPuntosGuardado, float *etaMinPuntosGuardado,
            float *etaMaxPuntosGuardado, double Hmin, int num_ts, double tiempo_act,
            double H, double Q, double T)
{
    double tiempo = tiempo_act*T;
    double h;
    int i;

    for (i=0; i<numPuntosGuardar; i++) {
        if (posicionesVolumenesGuardado[i] != -1) {
            h = datosVolumenesNivel_1[i].x();
            etaPuntosGuardado[i] = (float) ((h - datosVolumenesNivel_1[i].y() - Hmin)*H);
            uPuntosGuardado[i] = (float) (datosVolumenesNivel_2[i].x()*Q/H);
            vPuntosGuardado[i] = (float) (datosVolumenesNivel_2[i].y()*Q/H);
            etaMinPuntosGuardado[i] = min(etaPuntosGuardado[i], etaMinPuntosGuardado[i]);
            etaMaxPuntosGuardado[i] = max(etaPuntosGuardado[i], etaMaxPuntosGuardado[i]);
        }
    }
    writeStateTimeSeriesNC(num_ts, tiempo, numPuntosGuardar, etaPuntosGuardado, uPuntosGuardado, vPuntosGuardado);
}

/***************/
/* Blocks size */
/***************/

void obtenerTamBloquesKernel(int num_volx, int num_voly, range<2> *blockGridVer1, range<2> *blockGridVer2,
                             range<2> *blockGridHor1, range<2> *blockGridHor2, range<2> *blockGridEst)
{
    int num_aristas_ver1, num_aristas_ver2;
    int num_aristas_hor1, num_aristas_hor2;

    num_aristas_ver1 = (num_volx/2 + 1)*num_voly;
    num_aristas_ver2 = ((num_volx&1) == 0) ? num_volx*num_voly/2 : num_aristas_ver1;
    num_aristas_hor1 = (num_voly/2 + 1)*num_volx;
    num_aristas_hor2 = ((num_voly&1) == 0) ? num_volx*num_voly/2 : num_aristas_hor1;

    (*blockGridVer1)[1] = iDivUp(num_aristas_ver1/num_voly, NUM_HEBRASX_ARI);
    (*blockGridVer1)[0] = iDivUp(num_voly, NUM_HEBRASY_ARI);
    (*blockGridVer2)[1] = iDivUp(num_aristas_ver2/num_voly, NUM_HEBRASX_ARI);
    (*blockGridVer2)[0] = iDivUp(num_voly, NUM_HEBRASY_ARI);

    (*blockGridHor1)[1] = iDivUp(num_volx, NUM_HEBRASX_ARI);
    (*blockGridHor1)[0] = iDivUp(num_aristas_hor1/num_volx, NUM_HEBRASY_ARI);
    (*blockGridHor2)[1] = iDivUp(num_volx, NUM_HEBRASX_ARI);
    (*blockGridHor2)[0] = iDivUp(num_aristas_hor2/num_volx, NUM_HEBRASY_ARI);

    (*blockGridEst)[1] = iDivUp(num_volx, NUM_HEBRASX_EST);
    (*blockGridEst)[0] = iDivUp(num_voly, NUM_HEBRASY_EST);
}

/*******************/
/* Free GPU memory */
/*******************/

void liberarMemoria( sycl::queue &q, int numNiveles, double2 *d_datosVolumenesNivel_1, double2 *d_datosVolumenesNivel_2,
            tipoDatosSubmalla d_datosNivel, double *d_eta1MaximaNivel, double *d_deltaTVolumenesNivel,
            double2 *d_acumuladorNivel1_1, double2 *d_acumuladorNivel1_2, int leer_fichero_puntos,
            int *d_posicionesVolumenesGuardado, double2 *d_datosVolumenesGuardado_1, double2 *d_datosVolumenesGuardado_2)
{
    sycl::free(d_datosNivel.areaYCosPhi,q);
    sycl::free(d_datosNivel.anchoVolumenes,q);
    sycl::free(d_datosNivel.altoVolumenes,q);
    sycl::free(d_datosVolumenesNivel_1,q);
    sycl::free(d_datosVolumenesNivel_2,q);
    sycl::free(d_eta1MaximaNivel,q);
    sycl::free(d_deltaTVolumenesNivel,q);
    sycl::free(d_acumuladorNivel1_1,q);
    sycl::free(d_acumuladorNivel1_2,q);
    if (leer_fichero_puntos == 1) {
        sycl::free(d_posicionesVolumenesGuardado,q);
        sycl::free(d_datosVolumenesGuardado_1,q);
        sycl::free(d_datosVolumenesGuardado_2,q);
    }
}

/******************/
/* Main functions */
/******************/

double obtenerDeltaTInicialNivel0( queue &q,
        double2 *d_datosVolumenesNivel0_1, double2 *d_datosVolumenesNivel0_2, tipoDatosSubmalla d_datosNivel0,
        double *d_deltaTVolumenesNivel0, double2 *d_acumulador_1, int numVolxNivel0, int numVolyNivel0, double borde_izq,
        double borde_der, double borde_sup, double borde_inf, double CFL, double epsilon_h,
        range<2> blockGridVer1, range<2> blockGridVer2,
        range<2> blockGridHor1, range<2> blockGridHor2, range<2> threadBlockAri, range<2> blockGridEst, range<2> threadBlockEst)
{
    double delta_T;
    event  ev;
    nd_range<2> Krange { blockGridVer1, threadBlockAri }; // Kernel range

    Krange = nd_range<2> { blockGridVer1*threadBlockAri, threadBlockAri };
    ev = q.parallel_for( Krange, [=]( nd_item<2> idx )
    {
        procesarAristasVerDeltaTInicialNivel0GPU( idx, d_datosVolumenesNivel0_1, d_datosVolumenesNivel0_2,
        d_datosNivel0.areaYCosPhi, numVolxNivel0, numVolyNivel0, borde_izq, borde_der, d_datosNivel0.altoVolumenes,
        d_acumulador_1, epsilon_h, 1);
    });

    Krange = nd_range<2> { blockGridVer2*threadBlockAri, threadBlockAri };
    ev = q.parallel_for( Krange, ev, [=]( nd_item<2> idx )
    {
        procesarAristasVerDeltaTInicialNivel0GPU( idx, d_datosVolumenesNivel0_1, d_datosVolumenesNivel0_2,
        d_datosNivel0.areaYCosPhi, numVolxNivel0, numVolyNivel0, borde_izq, borde_der, d_datosNivel0.altoVolumenes,
        d_acumulador_1, epsilon_h, 2);
    });

    Krange = nd_range<2> { blockGridHor1*threadBlockAri, threadBlockAri };
    ev = q.parallel_for( Krange, ev, [=]( nd_item<2> idx )
    {
        procesarAristasHorDeltaTInicialNivel0GPU(idx,d_datosVolumenesNivel0_1, d_datosVolumenesNivel0_2,
        numVolxNivel0, numVolyNivel0, borde_sup, borde_inf, d_datosNivel0.anchoVolumenes, d_acumulador_1, epsilon_h, 1);
    });


    Krange = nd_range<2> { blockGridHor2*threadBlockAri, threadBlockAri };
    ev = q.parallel_for( Krange, ev, [=]( nd_item<2> idx )
    {
        procesarAristasHorDeltaTInicialNivel0GPU(idx,d_datosVolumenesNivel0_1, d_datosVolumenesNivel0_2,
        numVolxNivel0, numVolyNivel0, borde_sup, borde_inf, d_datosNivel0.anchoVolumenes, d_acumulador_1, epsilon_h, 2);
    });

    Krange = nd_range<2> { blockGridEst*threadBlockEst, threadBlockEst };
    ev = q.parallel_for( Krange, ev, [=]( nd_item<2> idx )
    {
        obtenerDeltaTVolumenesGPU(d_datosNivel0.areaYCosPhi, d_acumulador_1,
        d_deltaTVolumenesNivel0, numVolxNivel0, numVolyNivel0, CFL, idx);
    });

    // Abbreviations
    size_t size = numVolxNivel0*numVolyNivel0;
    double *d_data = d_deltaTVolumenesNivel0;

    double minResult = std::numeric_limits<double>::max();
    buffer<double> minBuf { &minResult, 1 };
    q.submit( [d_data,size,&minBuf,&ev]( handler &h )
    {
        h.depends_on(ev);
        auto minReduction = reduction(minBuf,h,minimum<>());
        h.parallel_for( range<1>(size), minReduction, [=]( id<1> idx, auto &min )
        {
            min.combine(d_data[idx]);
        });
    });

    return minBuf.get_host_access()[0];
}

double siguientePasoNivel0( sycl::queue &q, event ev,
        double2 *d_datosVolumenesNivel0_1, double2 *d_datosVolumenesNivel0_2, tipoDatosSubmalla d_datosNivel0,
        double2 *d_acumuladorNivel_1, double2 *d_acumuladorNivel_2, int numVolxNivel0, int numVolyNivel0, double *d_deltaTVolumenesNivel0,
        double borde_sup, double borde_inf, double borde_izq, double borde_der, double Hmin, int tam_spongeSup, int tam_spongeInf,
        int tam_spongeIzq, int tam_spongeDer, double sea_level, double *tiempo_act, double CFL, double delta_T, double mf0, double vmax,
        double epsilon_h, double hpos, double cvis, double L, double H, int64_t tam_datosVolDouble2Nivel0, range<2> blockGridVer1,
        range<2> blockGridVer2, range<2> blockGridHor1, range<2> blockGridHor2, range<2> threadBlockAri, range<2> blockGridEst, range<2> threadBlockEst)
{
    // PASO 1
    nd_range<2> Krange { blockGridVer1*threadBlockAri, threadBlockAri }; // Kernel range

    Krange = nd_range { blockGridVer1*threadBlockAri, threadBlockAri };
    ev = q.parallel_for<class siguiente1>( Krange, ev, [=]( nd_item<2> idx )
    {
        procesarAristasVerNivel0Paso1GPU(idx,d_datosVolumenesNivel0_1, d_datosVolumenesNivel0_2,
        d_datosNivel0.areaYCosPhi, d_datosNivel0.altoVolumenes, numVolxNivel0, numVolyNivel0, borde_izq, borde_der, delta_T,
        d_acumuladorNivel_1, CFL, epsilon_h, hpos, cvis, 1);
    });

    Krange = nd_range { blockGridVer2*threadBlockAri, threadBlockAri };
    ev = q.parallel_for<class siguiente2>( Krange, ev, [=]( nd_item<2> idx )
    {
        procesarAristasVerNivel0Paso1GPU(idx,d_datosVolumenesNivel0_1, d_datosVolumenesNivel0_2,
        d_datosNivel0.areaYCosPhi, d_datosNivel0.altoVolumenes, numVolxNivel0, numVolyNivel0, borde_izq, borde_der, delta_T,
        d_acumuladorNivel_1, CFL, epsilon_h, hpos, cvis, 2);
    });

    Krange = nd_range { blockGridHor1*threadBlockAri, threadBlockAri };
    ev = q.parallel_for<class siguiente3>( Krange, ev, [=]( nd_item<2> idx )
    {
        procesarAristasHorNivel0Paso1GPU( idx, d_datosVolumenesNivel0_1, d_datosVolumenesNivel0_2,
        d_datosNivel0.areaYCosPhi, d_datosNivel0.anchoVolumenes, numVolxNivel0, numVolyNivel0, borde_sup, borde_inf, delta_T,
        d_acumuladorNivel_1, CFL, epsilon_h, hpos, cvis, 1);
    });

    Krange = nd_range { blockGridHor2*threadBlockAri, threadBlockAri };
    ev = q.parallel_for<class siguiente4>( Krange, ev, [=]( nd_item<2> idx )
    {
        procesarAristasHorNivel0Paso1GPU( idx, d_datosVolumenesNivel0_1, d_datosVolumenesNivel0_2,
        d_datosNivel0.areaYCosPhi, d_datosNivel0.anchoVolumenes, numVolxNivel0, numVolyNivel0, borde_sup, borde_inf, delta_T,
        d_acumuladorNivel_1, CFL, epsilon_h, hpos, cvis, 2);
    });

    Krange = nd_range { blockGridEst*threadBlockAri, threadBlockEst };
    ev = q.parallel_for<class siguiente5>( Krange, ev, [=]( nd_item<2> idx )
    {
        obtenerEstadosPaso1Nivel0GPU(d_datosVolumenesNivel0_1, d_datosVolumenesNivel0_2,
        d_datosNivel0.areaYCosPhi, d_datosNivel0.anchoVolumenes, d_datosNivel0.altoVolumenes, d_acumuladorNivel_1,
        numVolxNivel0, numVolyNivel0, delta_T, Hmin, tam_spongeSup, tam_spongeInf, tam_spongeIzq, tam_spongeDer, sea_level, idx);
    });

    ev = q.memset(d_acumuladorNivel_1, 0, tam_datosVolDouble2Nivel0, ev);

    // PASO 2
    Krange = nd_range { blockGridVer1*threadBlockAri, threadBlockAri };
    ev = q.parallel_for<class siguiente6>( Krange, ev, [=]( nd_item<2> idx )
    {
        procesarAristasVerNivel0Paso2GPU(idx,d_datosVolumenesNivel0_1, d_datosVolumenesNivel0_2,
        d_datosNivel0.areaYCosPhi, d_datosNivel0.altoVolumenes, numVolxNivel0, numVolyNivel0, borde_izq, borde_der,
        delta_T, d_acumuladorNivel_1, d_acumuladorNivel_2, CFL, epsilon_h, hpos, cvis, 1);
    });

    Krange = nd_range { blockGridVer2*threadBlockAri, threadBlockAri };
    ev = q.parallel_for<class siguiente7>( Krange, ev, [=]( nd_item<2> idx )
    {
        procesarAristasVerNivel0Paso2GPU(idx, d_datosVolumenesNivel0_1, d_datosVolumenesNivel0_2,
        d_datosNivel0.areaYCosPhi, d_datosNivel0.altoVolumenes, numVolxNivel0, numVolyNivel0, borde_izq, borde_der,
        delta_T, d_acumuladorNivel_1, d_acumuladorNivel_2, CFL, epsilon_h, hpos, cvis, 2);
    });

    Krange = nd_range { blockGridHor1*threadBlockAri, threadBlockAri };
    ev = q.parallel_for<class siguiente8>( Krange, ev, [=]( nd_item<2> idx )
    {
        procesarAristasHorNivel0Paso2GPU( idx, d_datosVolumenesNivel0_1, d_datosVolumenesNivel0_2,
        d_datosNivel0.areaYCosPhi, d_datosNivel0.anchoVolumenes, numVolxNivel0, numVolyNivel0, borde_sup, borde_inf,
        delta_T, d_acumuladorNivel_1, d_acumuladorNivel_2, CFL, epsilon_h, hpos, cvis, 1);
    });

    Krange = nd_range { blockGridHor2*threadBlockAri, threadBlockAri };
    ev = q.parallel_for<class siguiente9>( Krange, ev, [=]( nd_item<2> idx )
    {
        procesarAristasHorNivel0Paso2GPU( idx, d_datosVolumenesNivel0_1, d_datosVolumenesNivel0_2,
        d_datosNivel0.areaYCosPhi, d_datosNivel0.anchoVolumenes, numVolxNivel0, numVolyNivel0, borde_sup, borde_inf,
        delta_T, d_acumuladorNivel_1, d_acumuladorNivel_2, CFL, epsilon_h, hpos, cvis, 2);
    });

    Krange = nd_range { blockGridEst*threadBlockEst, threadBlockEst };
    ev = q.parallel_for<class siguiente10>( Krange, ev, [=]( nd_item<2> idx )
    {
        obtenerEstadoYDeltaTVolumenesNivel0GPU(d_datosVolumenesNivel0_1, d_datosVolumenesNivel0_2,
        d_datosNivel0.areaYCosPhi, d_datosNivel0.anchoVolumenes, d_datosNivel0.altoVolumenes, d_acumuladorNivel_1,
        d_acumuladorNivel_2, d_deltaTVolumenesNivel0, numVolxNivel0, numVolyNivel0, CFL, delta_T, mf0, vmax, hpos,
        epsilon_h, Hmin, tam_spongeSup, tam_spongeInf, tam_spongeIzq, tam_spongeDer, sea_level, idx );
    });

    *tiempo_act += delta_T;

    // Abbreviations
    size_t size = numVolxNivel0*numVolyNivel0;
    double *d_data = d_deltaTVolumenesNivel0;

    // Reduction
    double minResult = std::numeric_limits<double>::max();
    buffer<double> minBuf { &minResult, 1 };
    q.submit( [d_data,size,&minBuf,&ev]( handler &h )
    {
        h.depends_on(ev);
        auto minReduction = reduction(minBuf,h,minimum<>());
        h.parallel_for<class siguiente11>( range<1>(size), minReduction, [=]( id<1> idx, auto &min )
        {
            min.combine(d_data[idx]);
        });
    });
    return minBuf.get_host_access()[0];
}

int shallowWater(int numNiveles, int okada_flag, double LON_C, double LAT_C, double DEPTH_C, double FAULT_L,
            double FAULT_W, double STRIKE, double DIP, double RAKE, double SLIP, double2 *datosVolumenesNivel_1,
            double2 *datosVolumenesNivel_2, tipoDatosSubmalla datosNivel, int leer_fichero_puntos, int numPuntosGuardar,
            int *posicionesVolumenesGuardado, double *lonPuntos, double *latPuntos, int numVolxNivel0, int numVolyNivel0,
            int64_t numVolumenesNivel, double Hmin, char *nombre_bati, std::string prefijo, double borde_sup, double borde_inf,
            double borde_izq, double borde_der, int tam_spongeSup, int tam_spongeInf, int tam_spongeIzq, int tam_spongeDer,
            double tiempo_tot, double tiempoGuardarNetCDF, double tiempoGuardarSeries, double CFL, double mf0, double vmax,
            double epsilon_h, double hpos, double cvis, double L, double H, double Q, double T, double *tiempo)
{
    queue q; event ev;
    std::cout << "Running on: " << q.get_device().get_info<info::device::name>() << std::endl;
    std::cout << "Using platform: " << q.get_device().get_platform().get_info<info::platform::name>() << std::endl;

    double2 *d_datosVolumenesNivel_1;
    double2 *d_datosVolumenesNivel_2;
    double2 *d_acumuladorNivel_1, *d_acumuladorNivel_2;
    tipoDatosSubmalla d_datosNivel;
    double *d_deltaTVolumenesNivel;
    double *d_eta1MaximaNivel;
    float *vec;
    double *p_eta;
    int num = 0;
    double sea_level = SEA_LEVEL/H;
    range<2> blockGridVer1Nivel(0,0);
    range<2> blockGridVer2Nivel(0,0);
    range<2> blockGridHor1Nivel(0,0);
    range<2> blockGridHor2Nivel(0,0);
    range<2> blockGridEstNivel(0,0);
    range<2> threadBlockAri(NUM_HEBRASX_ARI, NUM_HEBRASY_ARI);
    range<2> threadBlockEst(NUM_HEBRASX_EST, NUM_HEBRASY_EST);
    range<1> blockGridPuntos(iDivUp(numPuntosGuardar, NUM_HEBRAS_PUNTOS));
    range<1> threadBlockPuntos(NUM_HEBRAS_PUNTOS);
    double lon_ini, lat_ini;
    double incx, incy;

    int *d_posicionesVolumenesGuardado;
    double2 *d_datosVolumenesGuardado_1, *d_datosVolumenesGuardado_2;
    float *etaPuntosGuardado;
    float *uPuntosGuardado;
    float *vPuntosGuardado;
    float *etaMinPuntosGuardado, *etaMaxPuntosGuardado;
    int num_ts = 0;

    int64_t tam_datosVolDoubleNivel;
    int64_t tam_datosVolDouble2Nivel;
    int64_t tam_datosDeltaTDouble;
    int64_t tam_datosAcumDouble2Nivel;
    int64_t tam_datosVolGuardadoDouble2 = ((int64_t) numPuntosGuardar)*sizeof(double2);
    double deltaTNivel;
    double tiempoActSubmalla;
    double sigTiempoGuardarNetCDF = 0.0;
    double sigTiempoGuardarSeries = 0.0;
    int i, iter;

    tiempoActSubmalla = 0.0;
    obtenerTamBloquesKernel(numVolxNivel0, numVolyNivel0, &blockGridVer1Nivel, &blockGridVer2Nivel,
        &blockGridHor1Nivel, &blockGridHor2Nivel, &blockGridEstNivel);

    tam_datosDeltaTDouble = 0;
    tam_datosAcumDouble2Nivel = 0;
    d_datosNivel.areaYCosPhi    = malloc_device<double2>(numVolyNivel0  ,q);
    d_datosNivel.anchoVolumenes = malloc_device<double >(numVolyNivel0+1,q);
    d_datosNivel.altoVolumenes  = malloc_device<double >(numVolyNivel0  ,q);
    tam_datosVolDoubleNivel = numVolumenesNivel*sizeof(double);
    tam_datosVolDouble2Nivel = numVolumenesNivel*sizeof(double2);
    tam_datosDeltaTDouble = max(tam_datosDeltaTDouble, tam_datosVolDoubleNivel);
    tam_datosAcumDouble2Nivel = max(tam_datosAcumDouble2Nivel, tam_datosVolDouble2Nivel);


    d_datosVolumenesNivel_1 = malloc_device<double2>(tam_datosVolDouble2Nivel /sizeof(double2),q);
    d_datosVolumenesNivel_2 = malloc_device<double2>(tam_datosVolDouble2Nivel /sizeof(double2),q);
    d_eta1MaximaNivel       = malloc_device<double >(tam_datosVolDoubleNivel  /sizeof(double ),q);
    d_deltaTVolumenesNivel  = malloc_device<double >(tam_datosDeltaTDouble    /sizeof(double ),q);
    d_acumuladorNivel_1     = malloc_device<double2>(tam_datosAcumDouble2Nivel/sizeof(double2),q);
    d_acumuladorNivel_2     = malloc_device<double2>(tam_datosAcumDouble2Nivel/sizeof(double2),q);

    if ( d_datosNivel.areaYCosPhi    == nullptr ||
         d_datosNivel.anchoVolumenes == nullptr ||
         d_datosNivel.altoVolumenes  == nullptr ||
         d_datosVolumenesNivel_1     == nullptr ||
         d_datosVolumenesNivel_2     == nullptr ||
         d_eta1MaximaNivel           == nullptr ||
         d_deltaTVolumenesNivel      == nullptr ||
         d_acumuladorNivel_1         == nullptr ||
         d_acumuladorNivel_2         == nullptr )
    {
        free(d_datosNivel.areaYCosPhi,q);
        free(d_datosNivel.anchoVolumenes,q);
        free(d_datosNivel.altoVolumenes,q);
        free(d_datosVolumenesNivel_1,q);
        free(d_datosVolumenesNivel_2,q);
        free(d_eta1MaximaNivel,q);
        free(d_deltaTVolumenesNivel,q);
        free(d_acumuladorNivel_1,q);
        free(d_acumuladorNivel_2,q);
        return 1;
    }


    if (leer_fichero_puntos == 1)
    {
        d_posicionesVolumenesGuardado = malloc_device<int>(numPuntosGuardar,q);
        d_datosVolumenesGuardado_1    = malloc_device<double2>(tam_datosVolGuardadoDouble2/sizeof(double2),q);
        d_datosVolumenesGuardado_2    = malloc_device<double2>(tam_datosVolGuardadoDouble2/sizeof(double2),q);

        if ( d_posicionesVolumenesGuardado == nullptr ||
             d_datosVolumenesGuardado_1    == nullptr ||
             d_datosVolumenesGuardado_2    == nullptr )
        {
            liberarMemoria(q,numNiveles, d_datosVolumenesNivel_1, d_datosVolumenesNivel_2, d_datosNivel,
                d_eta1MaximaNivel, d_deltaTVolumenesNivel, d_acumuladorNivel_1, d_acumuladorNivel_2,
                1, d_posicionesVolumenesGuardado, d_datosVolumenesGuardado_1, d_datosVolumenesGuardado_2);
            return 1;
        }
    }


    q.memcpy(d_datosNivel.areaYCosPhi, datosNivel.areaYCosPhi, numVolyNivel0*sizeof(double2));
    q.memcpy(d_datosNivel.anchoVolumenes, datosNivel.anchoVolumenes, (numVolyNivel0+1)*sizeof(double));
    q.memcpy(d_datosNivel.altoVolumenes, datosNivel.altoVolumenes, numVolyNivel0*sizeof(double));
    q.memcpy(d_datosVolumenesNivel_1, datosVolumenesNivel_1, tam_datosVolDouble2Nivel);
    q.memcpy(d_datosVolumenesNivel_2, datosVolumenesNivel_2, tam_datosVolDouble2Nivel);
    if (leer_fichero_puntos == 1)
    {
        q.memcpy(d_posicionesVolumenesGuardado, posicionesVolumenesGuardado, ((size_t) numPuntosGuardar)*sizeof(int) );
    }
    q.wait();

    if (okada_flag == OKADA_STANDARD)
    {
        lon_ini = datosNivel.longitud[0];
        lat_ini = datosNivel.latitud[0];
        incx = (datosNivel.longitud[numVolxNivel0-1] - datosNivel.longitud[0])/(numVolxNivel0-1);
        incy = (datosNivel.latitud[numVolyNivel0-1] - datosNivel.latitud[0])/(numVolyNivel0-1);
        fprintf(stdout, "Applying Okada\n");

        nd_range<2> okada_range { blockGridEstNivel*threadBlockEst, threadBlockEst };
        q.parallel_for( okada_range, [=]( nd_item<2> idx )
        {
            aplicarOkadaStandardGPU(d_datosVolumenesNivel_1, numVolxNivel0, numVolyNivel0,
            lon_ini, incx, lat_ini, incy, LON_C, LAT_C, DEPTH_C, FAULT_L, FAULT_W, STRIKE, DIP, RAKE, SLIP, H,
            idx);
        });
    }
    q.wait();

    nd_range<2> Krange { blockGridEstNivel*threadBlockEst, threadBlockEst };
    q.parallel_for( Krange, [=]( nd_item<2> idx )
    {
        inicializarEta1MaximaNivel0GPU(d_datosVolumenesNivel_1,
            d_eta1MaximaNivel, numVolxNivel0, numVolyNivel0, epsilon_h,idx);
    }).wait();


    // INICIO NETCDF
    double mf0_ini = sqrt(mf0*pow(H,4.0/3.0)/(9.81*L));
    if (tiempoGuardarNetCDF >= 0.0) {
        vec = (float *) malloc(tam_datosDeltaTDouble);
        if (vec == NULL) {
            liberarMemoria(q,numNiveles, d_datosVolumenesNivel_1, d_datosVolumenesNivel_2, d_datosNivel,
                d_eta1MaximaNivel, d_deltaTVolumenesNivel, d_acumuladorNivel_1, d_acumuladorNivel_2,
                leer_fichero_puntos, d_posicionesVolumenesGuardado, d_datosVolumenesGuardado_1,
                d_datosVolumenesGuardado_2);
            return 2;
        }
        for (i=0; i<numVolumenesNivel; i++)
            vec[i] = (float) ((datosVolumenesNivel_1[i].y() + Hmin)*H);
        iter = crearFicherosNC(nombre_bati, okada_flag, (char *) prefijo.c_str(), numVolxNivel0, numVolyNivel0,
                    datosNivel.longitud, datosNivel.latitud, tiempo_tot*T, CFL, epsilon_h*H, mf0_ini, vmax*Q/H,
                    hpos*H, 1.0-cvis, borde_sup, borde_inf, borde_izq, borde_der, LON_C, LAT_C, DEPTH_C,
                    FAULT_L, FAULT_W, STRIKE, DIP, RAKE, SLIP, vec);
        if (iter == 1) {
            liberarMemoria(q,numNiveles, d_datosVolumenesNivel_1, d_datosVolumenesNivel_2, d_datosNivel,
                d_eta1MaximaNivel, d_deltaTVolumenesNivel, d_acumuladorNivel_1, d_acumuladorNivel_2,
                leer_fichero_puntos, d_posicionesVolumenesGuardado, d_datosVolumenesGuardado_1,
                d_datosVolumenesGuardado_2);
            free(vec);
            return 2;
        }
    }
    if (leer_fichero_puntos == 1) {
        etaPuntosGuardado = (float *) malloc(numPuntosGuardar*sizeof(float));
        uPuntosGuardado = (float *) malloc(numPuntosGuardar*sizeof(float));
        vPuntosGuardado = (float *) malloc(numPuntosGuardar*sizeof(float));
        etaMinPuntosGuardado = (float *) malloc(numPuntosGuardar*sizeof(float));
        etaMaxPuntosGuardado = (float *) malloc(numPuntosGuardar*sizeof(float));
        if (etaMaxPuntosGuardado == NULL) {
            if (etaPuntosGuardado != NULL)          free(etaPuntosGuardado);
            if (uPuntosGuardado != NULL)            free(uPuntosGuardado);
            if (vPuntosGuardado != NULL)            free(vPuntosGuardado);
            if (etaMinPuntosGuardado != NULL)       free(etaMinPuntosGuardado);
            if (tiempoGuardarNetCDF >= 0.0)
                free(vec);
            liberarMemoria(q,numNiveles, d_datosVolumenesNivel_1, d_datosVolumenesNivel_2, d_datosNivel,
                d_eta1MaximaNivel, d_deltaTVolumenesNivel, d_acumuladorNivel_1, d_acumuladorNivel_2,
                leer_fichero_puntos, d_posicionesVolumenesGuardado, d_datosVolumenesGuardado_1,
                d_datosVolumenesGuardado_2);
            return 2;
        }
        initTimeSeriesNC(nombre_bati, (char *) prefijo.c_str(), numPuntosGuardar, lonPuntos, latPuntos,
            tiempo_tot*T, CFL, epsilon_h*H, mf0_ini, vmax*Q/H, hpos*H, 1.0-cvis, borde_sup, borde_inf,
            borde_izq, borde_der, LON_C, LAT_C, DEPTH_C, FAULT_L, FAULT_W, STRIKE, DIP, RAKE, SLIP, okada_flag);
        for (i=0; i<numPuntosGuardar; i++) {
            if (posicionesVolumenesGuardado[i] == -1) {
                etaPuntosGuardado[i] = -9999.0f;
                uPuntosGuardado[i] = -9999.0f;
                vPuntosGuardado[i] = -9999.0f;
                etaMinPuntosGuardado[i] = -9999.0f;
                etaMaxPuntosGuardado[i] = -9999.0f;
            }
			else {
				etaMinPuntosGuardado[i] = 1e30f;
				etaMaxPuntosGuardado[i] = -1e30f;
			}
        }
    }
    // FIN NETCDF

    q.memset(d_acumuladorNivel_1, 0, tam_datosVolDouble2Nivel);
    q.memset(d_acumuladorNivel_2, 0, tam_datosVolDouble2Nivel);
    q.wait();

    stopwatch<double> timer;
    // Note: d_datosNivel needs to be passed by value.
    deltaTNivel =   obtenerDeltaTInicialNivel0(q,d_datosVolumenesNivel_1, d_datosVolumenesNivel_2, d_datosNivel,
                    d_deltaTVolumenesNivel, d_acumuladorNivel_1, numVolxNivel0, numVolyNivel0, borde_izq,
                    borde_der, borde_sup, borde_inf, CFL, epsilon_h, blockGridVer1Nivel, blockGridVer2Nivel,
                    blockGridHor1Nivel, blockGridHor2Nivel, threadBlockAri, blockGridEstNivel, threadBlockEst);
    fprintf(stdout, "Initial deltaT = %e sec\n", deltaTNivel*T);

    q.memset(d_acumuladorNivel_1, 0, tam_datosVolDouble2Nivel).wait();

    iter = 1;
    while (tiempoActSubmalla < tiempo_tot) {
        // INICIO NETCDF
        if ((tiempoGuardarNetCDF >= 0.0) && (tiempoActSubmalla >= sigTiempoGuardarNetCDF))
        {
            q.memcpy(datosVolumenesNivel_1, d_datosVolumenesNivel_1, tam_datosVolDouble2Nivel, ev);
            q.memcpy(datosVolumenesNivel_2, d_datosVolumenesNivel_2, tam_datosVolDouble2Nivel, ev);
            q.wait();

            sigTiempoGuardarNetCDF += tiempoGuardarNetCDF;
            guardarNetCDFNivel0(datosVolumenesNivel_1, datosVolumenesNivel_2, vec, num, numVolxNivel0, numVolyNivel0,
                Hmin, tiempoActSubmalla, epsilon_h, H, Q, T);
            num++;
        }
        if ((tiempoGuardarSeries >= 0.0) && (tiempoActSubmalla >= sigTiempoGuardarSeries))
        {
            sigTiempoGuardarSeries += tiempoGuardarSeries;

            nd_range<1> my_range { blockGridPuntos*threadBlockPuntos, threadBlockPuntos };
            q.parallel_for( my_range, ev, [=]( nd_item<1> idx )
            {
                escribirVolumenesGuardadoGPU(d_datosVolumenesNivel_1,
                d_datosVolumenesNivel_2, d_datosVolumenesGuardado_1, d_datosVolumenesGuardado_2,
                d_posicionesVolumenesGuardado, numPuntosGuardar, epsilon_h,idx);
            });
            q.wait();

            q.memcpy(datosVolumenesNivel_1, d_datosVolumenesGuardado_1, tam_datosVolGuardadoDouble2 );
            q.memcpy(datosVolumenesNivel_2, d_datosVolumenesGuardado_2, tam_datosVolGuardadoDouble2 );
            guardarSerieTiemposNivel0(datosVolumenesNivel_1, datosVolumenesNivel_2, numPuntosGuardar, posicionesVolumenesGuardado,
                etaPuntosGuardado, uPuntosGuardado, vPuntosGuardado, etaMinPuntosGuardado, etaMaxPuntosGuardado,
                Hmin, num_ts, tiempoActSubmalla, H, Q, T);
            num_ts++;
        }
        // FIN NETCDF

        Krange = nd_range<2> { blockGridEstNivel*threadBlockEst, threadBlockEst };
        ev = q.parallel_for<class actualizar>( Krange, {ev}, [=]( nd_item<2> idx )
        {
            actualizarEta1MaximaNivel0GPU(d_datosVolumenesNivel_1,
            d_eta1MaximaNivel, numVolxNivel0, numVolyNivel0, tiempoActSubmalla, epsilon_h,idx);
        });

        // d_datosNivel needs to be passed by value.
        deltaTNivel = siguientePasoNivel0(q,ev,d_datosVolumenesNivel_1, d_datosVolumenesNivel_2, d_datosNivel, d_acumuladorNivel_1,
                        d_acumuladorNivel_2, numVolxNivel0, numVolyNivel0, d_deltaTVolumenesNivel, borde_sup, borde_inf,
                        borde_izq, borde_der, Hmin, tam_spongeSup, tam_spongeInf, tam_spongeIzq, tam_spongeDer, sea_level,
                        &tiempoActSubmalla, CFL, deltaTNivel, mf0, vmax, epsilon_h, hpos, cvis, L, H, tam_datosVolDouble2Nivel,
                        blockGridVer1Nivel, blockGridVer2Nivel, blockGridHor1Nivel, blockGridHor2Nivel, threadBlockAri,
                        blockGridEstNivel, threadBlockEst);

        ev = q.memset(d_acumuladorNivel_1, 0, tam_datosVolDouble2Nivel);
        ev = q.memset(d_acumuladorNivel_2, 0, tam_datosVolDouble2Nivel, ev);

        fprintf(stdout, "Iteration %3d, deltaT = %e sec, ", iter, deltaTNivel*T);
        fprintf(stdout, "Time = %g sec\n", tiempoActSubmalla*T);
        iter++;
    }
    q.wait();
    *tiempo = timer.elapsed();

    // INICIO NETCDF
    if (tiempoGuardarNetCDF >= 0.0) {
        if (okada_flag == OKADA_STANDARD) 
        {
            q.memcpy(datosVolumenesNivel_1,d_datosVolumenesNivel_1, tam_datosVolDouble2Nivel).wait();
            for (i=0; i<numVolumenesNivel; i++)
                vec[i] = (float) ((datosVolumenesNivel_1[i].y() + Hmin)*H);
            guardarBatimetriaModificadaNC(vec);
        }

        q.memcpy(datosVolumenesNivel_1, d_eta1MaximaNivel, tam_datosVolDoubleNivel).wait();
        p_eta = (double *) datosVolumenesNivel_1;
        for (i=0; i<numVolumenesNivel; i++) {
            if (p_eta[i] < -1e20)
                vec[i] = -9999.0f;
            else
                vec[i] = (float) ((p_eta[i] - Hmin)*H);
        }
        cerrarFicheroNC(vec);
        free(vec);
    }
    if (leer_fichero_puntos == 1) 
    {
        nd_range<1> my_range = nd_range<1> { blockGridPuntos*threadBlockPuntos, threadBlockPuntos };
        q.parallel_for( my_range, [=]( nd_item<1> idx )
        {
            escribirVolumenesGuardadoGPU(d_datosVolumenesNivel_1,
            d_datosVolumenesNivel_2, d_datosVolumenesGuardado_1, d_datosVolumenesGuardado_2,
            d_posicionesVolumenesGuardado, numPuntosGuardar, epsilon_h,idx);
        }).wait();

        q.memcpy(datosVolumenesNivel_1, d_datosVolumenesGuardado_1, tam_datosVolGuardadoDouble2).wait();
        obtenerBatimetriaParaSerieTiempos(datosVolumenesNivel_1, numPuntosGuardar, posicionesVolumenesGuardado,
            etaPuntosGuardado, Hmin, H);
        guardarBatimetriaModificadaTimeSeriesNC(etaPuntosGuardado);
        guardarAmplitudesTimeSeriesNC(etaMinPuntosGuardado, etaMaxPuntosGuardado);
        closeTimeSeriesNC();
        free(etaPuntosGuardado);
        free(uPuntosGuardado);
        free(vPuntosGuardado);
        free(etaMinPuntosGuardado);
        free(etaMaxPuntosGuardado);
    }
    // FIN NETCDF

    liberarMemoria(q,numNiveles, d_datosVolumenesNivel_1, d_datosVolumenesNivel_2, d_datosNivel,
        d_eta1MaximaNivel, d_deltaTVolumenesNivel, d_acumuladorNivel_1, d_acumuladorNivel_2,
        leer_fichero_puntos, d_posicionesVolumenesGuardado, d_datosVolumenesGuardado_1,
        d_datosVolumenesGuardado_2);

    return 0;
}

