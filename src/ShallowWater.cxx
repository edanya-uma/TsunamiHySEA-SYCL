/**********************************************************************
 Tsunami-HySEA numerical model open source v1.3.0
 developed by the EDANYA Research Group, University of Malaga (Spain).
 Tsunami-HySEA SYCL version implemented by Intel Corporation engineers
 Matthias Kirchhart and Kazuki Minemura
 https://www.uma.es/edanya
 https://edanya.uma.es/hysea/
 Contact and support: hysea@uma.es
 Distributed under GNU General Public License, Version 2
**********************************************************************/

#include <string>
#include "Constantes.hxx"
#include "Problema.hxx"

using namespace std;
using namespace sycl;

/*****************/
/* Funciones GPU */
/*****************/

int shallowWater(int numNiveles, int okada_flag, int numFaults, int crop_flag, double crop_value, double *LON_C,
			double *LAT_C, double *DEPTH_C, double *FAULT_L, double *FAULT_W, double *STRIKE, double *DIP, double *RAKE,
			double *SLIP, double *defTime, double2 *datosVolumenesNivel_1, double2 *datosVolumenesNivel_2, tipoDatosSubmalla datosNivel,
			int leer_fichero_puntos, int numPuntosGuardar, int *posicionesVolumenesGuardado, double *lonPuntos, double *latPuntos,
			int numVolxNivel0, int numVolyNivel0, int64_t numVolumenesNivel, double Hmin, char *nombre_bati, string prefijo,
			double borde_sup, double borde_inf, double borde_izq, double borde_der, int tam_spongeSup, int tam_spongeInf,
			int tam_spongeIzq, int tam_spongeDer, double tiempo_tot, double tiempoGuardarNetCDF, double tiempoGuardarSeries,
			double CFL, double mf0, double vmax, double epsilon_h, double hpos, double cvis, double dif_at, double L, double H,
			double Q, double T, char *version, double *tiempo);

/*********************/
/* Fin funciones GPU */
/*********************/

int main(int argc, char *argv[])
{
	string version = "1.3.0";
    double2 *datosVolumenesNivel_1;
    double2 *datosVolumenesNivel_2;
    tipoDatosSubmalla datosNivel;
    int *posicionesVolumenesGuardado = NULL;
    double *lonPuntos = NULL;
    double *latPuntos = NULL;
    int leer_fichero_puntos;
    int numPuntosGuardar;
    int err;
    int numVolxNivel0, numVolyNivel0;
    int64_t numVolumenesNivel;
    double borde_sup, borde_inf, borde_izq, borde_der;
    int tam_spongeSup, tam_spongeInf, tam_spongeIzq, tam_spongeDer;
    double tiempo_tot;
    double tiempoGuardarNetCDF, tiempoGuardarSeries;
    double Hmin, epsilon_h, dif_at;
    double mf0, vmax;
    double CFL, hpos, cvis;
    double L, H, Q, T;
    string fich_ent;
    string nombre_bati;
    string prefijo;
    int numNiveles;
    double tiempo_gpu;
    int okada_flag, numFaults;
	int crop_flag;
	double crop_value;
	double LON_C[MAX_FAULTS], LAT_C[MAX_FAULTS], DEPTH_C[MAX_FAULTS], FAULT_L[MAX_FAULTS], FAULT_W[MAX_FAULTS];
	double STRIKE[MAX_FAULTS], DIP[MAX_FAULTS], RAKE[MAX_FAULTS], SLIP[MAX_FAULTS], defTime[MAX_FAULTS];

    if (argc < 2) {
        cerr << "/**********************************************************************" << std::endl;
        cerr << " Tsunami-HySEA numerical model open source v" << version << std::endl;
        cerr << " developed by the EDANYA Research Group, University of Malaga (Spain). " << std::endl;
        cerr << " Tsunami-HySEA SYCL version implemented by Intel Corporation engineers " << std::endl;
        cerr << " Matthias Kirchhart and Kazuki Minemura                                " << std::endl;
        cerr << " https://www.uma.es/edanya                                             " << std::endl;
        cerr << " https://edanya.uma.es/hysea/                                          " << std::endl;
        cerr << " Contact and support: hysea@uma.es                                     " << std::endl;
        cerr << " Distributed under GNU General Public License, Version 2               " << std::endl;
        cerr << "**********************************************************************/" << std::endl;
        cerr << std::endl;
        cerr << "Use: " << std::endl;
        cerr << argv[0] << " dataFile" << std::endl << std::endl; 
        cerr << "dataFile format:" << std::endl;
        cerr << "  Bathymetry name" << std::endl;
        cerr << "  Bathymetry file" << std::endl;
        cerr << "  Initialization of states (" << SEA_SURFACE_FROM_FILE << ": Sea surface displacement from file," << std::endl;
        cerr << "                            " << OKADA_STANDARD << ": Standard Okada)" << std::endl;
        cerr << "  If " << SEA_SURFACE_FROM_FILE << ":" << std::endl;
        cerr << "    Initial state file" << std::endl;
        cerr << "  Else if " << OKADA_STANDARD << ":" << std::endl;
		cerr << "    Number of faults (>= 1)" << std::endl;
		cerr << "    For every fault, a line containing:" << std::endl;
		cerr << "      Time(sec) Lon_epicenter Lat_epicenter Depth_hypocenter(km) Fault_length(km) Fault_width(km) Strike Dip Rake Slip(m)" << std::endl;
		cerr << "    Crop deformations (" << NO_CROP << ": No, " << CROP_RELATIVE << ": Relative cropping, " << CROP_ABSOLUTE << ": Absolute cropping)" << std::endl;
		cerr << "    If " << CROP_RELATIVE << ":" << std::endl;
		cerr << "      Percentage [0,1]" << std::endl;
		cerr << "    Else if " << CROP_ABSOLUTE << ":" << std::endl;
		cerr << "      Threshold value (m)" << std::endl;
        cerr << "  NetCDF file prefix" << std::endl;
        cerr << "  Number of levels (should be 1)" << std::endl;
		cerr << "  Upper border condition (1: open, -1: wall)" << std::endl;
		cerr << "  Lower border condition (1: open, -1: wall)" << std::endl;
		cerr << "  Left border condition (1: open, -1: wall, 2: periodic)" << std::endl;
		cerr << "  Right border condition (1: open, -1: wall, 2: periodic)" << std::endl;
        cerr << "  Simulation time (sec)" << std::endl;
        cerr << "  Saving time of NetCDF files (sec) (-1: do not save)" << std::endl;
        cerr << "  Read points from file (0: no, 1: yes)" << std::endl;
        cerr << "  If 1:" << std::endl;
        cerr << "    File with points" << std::endl;
        cerr << "    Saving time of time series (sec)" << std::endl;
        cerr << "  CFL" << std::endl;
        cerr << "  Epsilon h (m)" << std::endl;
        cerr << "  Threshold for the 2s+WAF scheme (m)" << std::endl;
        cerr << "  Stability coefficient" << std::endl;
        cerr << "  Water-bottom friction (Manning coefficient)" << std::endl;
        cerr << "  Maximum allowed velocity of water" << std::endl;
        cerr << "  L (typical length)" << std::endl;
        cerr << "  H (typical depth)" << std::endl;
        cerr << "  Threshold for arrival times (m) (epsilon h if not specified)" << std::endl;
        return EXIT_FAILURE;
    }

    fich_ent = argv[1];
    if (! existeFichero(fich_ent)) {
        cerr << "Error: File '" << fich_ent << "' not found" << std::endl;
        return EXIT_FAILURE;
    }

    err = cargarDatosProblema(fich_ent, nombre_bati, prefijo, &numNiveles, &okada_flag, &numFaults, &crop_flag,
			&crop_value, LON_C, LAT_C, DEPTH_C, FAULT_L, FAULT_W, STRIKE, DIP, RAKE, SLIP, defTime, &datosVolumenesNivel_1,
			&datosVolumenesNivel_2, &datosNivel, &numVolxNivel0, &numVolyNivel0, &numVolumenesNivel, &leer_fichero_puntos,
			&numPuntosGuardar, &posicionesVolumenesGuardado, &lonPuntos, &latPuntos, &Hmin, &borde_sup, &borde_inf,
			&borde_izq, &borde_der, &tam_spongeSup, &tam_spongeInf, &tam_spongeIzq, &tam_spongeDer, &tiempo_tot,
			&tiempoGuardarNetCDF, &tiempoGuardarSeries, &CFL, &mf0, &vmax, &epsilon_h, &hpos, &cvis, &dif_at,
			&L, &H, &Q, &T);
    if (err > 0)
        return EXIT_FAILURE;

    mostrarDatosProblema(version, numNiveles, okada_flag, numFaults, crop_flag, crop_value, datosNivel, numVolxNivel0,
        numVolyNivel0, tiempo_tot, tiempoGuardarNetCDF, leer_fichero_puntos, tiempoGuardarSeries, CFL, mf0, vmax,
        epsilon_h, hpos, cvis, dif_at, L, H, Q, T);

    cout << std::scientific;
    err = shallowWater(numNiveles, okada_flag, numFaults, crop_flag, crop_value, LON_C, LAT_C, DEPTH_C, FAULT_L, FAULT_W,
            STRIKE, DIP, RAKE, SLIP, defTime, datosVolumenesNivel_1, datosVolumenesNivel_2, datosNivel, leer_fichero_puntos,
            numPuntosGuardar, posicionesVolumenesGuardado, lonPuntos, latPuntos, numVolxNivel0, numVolyNivel0, numVolumenesNivel,
            Hmin, const_cast<char *>(nombre_bati.c_str()), prefijo, borde_sup, borde_inf, borde_izq, borde_der, tam_spongeSup,
            tam_spongeInf, tam_spongeIzq, tam_spongeDer, tiempo_tot, tiempoGuardarNetCDF, tiempoGuardarSeries, CFL, mf0,
            vmax, epsilon_h, hpos, cvis, dif_at, L, H, Q, T, const_cast<char *>(version.c_str()), &tiempo_gpu);
    if (err > 0) {
        if (err == 1)
            cerr << "Error: Not enough GPU memory" << std::endl;
        else if (err == 2)
            cerr << "Error: Not enough CPU memory" << std::endl;
        liberarMemoria(datosVolumenesNivel_1, datosVolumenesNivel_2, datosNivel,
            posicionesVolumenesGuardado, lonPuntos, latPuntos);
        return EXIT_FAILURE;
    }
    cout << std::endl << "Runtime: " << tiempo_gpu << " sec" << std::endl;
    liberarMemoria(datosVolumenesNivel_1, datosVolumenesNivel_2, datosNivel,
        posicionesVolumenesGuardado, lonPuntos, latPuntos);

    return EXIT_SUCCESS;
}

