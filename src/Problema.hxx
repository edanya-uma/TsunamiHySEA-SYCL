#ifndef _PROBLEMA_HXX_
#define _PROBLEMA_HXX_

#include "Constantes.hxx"
#include <sycl/sycl.hpp>
#include <string>
#include <fstream>

template <class T>
void obtenerSiguienteDato( std::ifstream &fich, T &dato);
extern template void obtenerSiguienteDato<int>( std::ifstream &fich, int &dato);
extern template void obtenerSiguienteDato<double>(std::ifstream &fich, double &dato);
extern template void obtenerSiguienteDato<std::string>(std::ifstream &fich, std::string &dato);

bool existeFichero(std::string fichero);
void liberarMemoria(sycl::double2 *datosVolumenesNivel_1, sycl::double2 *datosVolumenesNivel_2,
		tipoDatosSubmalla datosNivel, int *posicionesVolumenesGuardado, double *lonPuntos, double *latPuntos);

int obtenerPosEnPuntosGuardado(double *lonPuntos, double *latPuntos, int num_puntos, double lon, double lat);
int obtenerIndicePunto(int numVolxNivel0, int numVolyNivel0, tipoDatosSubmalla *datosNivel, double lon, double lat);

void ordenarFallasOkadaStandardPorTiempo(int numFaults, double *LON_C, double *LAT_C, double *DEPTH_C, double *FAULT_L,
		double *FAULT_W, double *STRIKE, double *DIP, double *RAKE, double *SLIP, double *defTime);
void obtenerDatosFallaOkadaStandard( std::ifstream &fich, double &time, double &LON_C, double &LAT_C, double &DEPTH_C,
		double &FAULT_L, double &FAULT_W, double &STRIKE, double &DIP, double &RAKE, double &SLIP);

int cargarDatosProblema(std::string fich_ent, std::string &nombre_bati, std::string &prefijo, int *numNiveles, int *okada_flag,
		int *numFaults, int *crop_flag, double *crop_value, double *LON_C, double *LAT_C, double *DEPTH_C, double *FAULT_L,
		double *FAULT_W, double *STRIKE, double *DIP, double *RAKE, double *SLIP, double *defTime, sycl::double2 **datosVolumenesNivel_1,
		sycl::double2 **datosVolumenesNivel_2, tipoDatosSubmalla *datosNivel, int *numVolxNivel0, int *numVolyNivel0,
		int64_t *numVolumenesNivel, int *leer_fichero_puntos, int *numPuntosGuardar, int **posicionesVolumenesGuardado,
		double **lonPuntos, double **latPuntos, double *Hmin, double *borde_sup, double *borde_inf, double *borde_izq,
		double *borde_der, int *tam_spongeSup, int *tam_spongeInf, int *tam_spongeIzq, int *tam_spongeDer, double *tiempo_tot,
		double *tiempoGuardarNetCDF, double *tiempoGuardarSeries, double *CFL, double *mf0, double *vmax, double *epsilon_h,
		double *hpos, double *cvis, double *dif_at, double *L, double *H, double *Q, double *T);

void mostrarDatosProblema(std::string version, int numNiveles, int okada_flag, int numFaults, int crop_flag, double crop_value,
		tipoDatosSubmalla datosNivel, int numVolxNivel0, int numVolyNivel0, double tiempo_tot, double tiempoGuardarNetCDF,
		int leer_fichero_puntos, double tiempoGuardarSeries, double CFL, double mf0, double vmax, double epsilon_h,
		double hpos, double cvis, double dif_at, double L, double H, double Q, double T);

#endif

