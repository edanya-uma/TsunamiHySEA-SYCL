#ifndef _NETCDF_HXX_
#define _NETCDF_HXX_

extern int ncid;
extern int time_id;
extern int eta_id, eta_max_id;
extern int ux_id, uy_id;
extern int grid_id_okada;

void abrirGRD(const char *nombre_fich, int *nvx, int *nvy);
void leerLongitudGRD(double *lon);
void leerLatitudGRD(double *lat);
void leerBatimetriaGRD(float *bati);
void leerEtaGRD(float *eta);
void cerrarGRD();

void crearFicheroNC(double *lon_grid, double *lat_grid, double *lon, double *lat, char *nombre_bati,
			char *prefijo, int *p_ncid, int *p_time_id, int *p_eta_id, int *p_qx_id, int *p_qy_id,
			int num_volx, int num_voly, double tiempo_tot, double CFL, double epsilon_h, double mf0,
			double vmax, double hpos, double cvis, double *bati);

int crearFicherosNC(char *nombre_bati, int okada_flag, char *prefijo, int num_volx, int num_voly, double *lon_grid,
			double *lat_grid, double tiempo_tot, double CFL, double epsilon_h, double mf0, double vmax,
			double hpos, double cvis, double borde_sup, double borde_inf, double borde_izq, double borde_der,
			double LON_C, double LAT_C, double DEPTH_C, double FAULT_L, double FAULT_W, double STRIKE,
			double DIP, double RAKE, double SLIP, float *bati);

void escribirTiempoNC(int num, float tiempo_act);
void escribirEtaNC(int num_volx, int num_voly, int num, float *eta);
void escribirUxNC(int num_volx, int num_voly, int num, float *ux);
void escribirUyNC(int num_volx, int num_voly, int num, float *uy);
void guardarBatimetriaModificadaNC(float *vec);
void cerrarFicheroNC(float *eta_max);


#endif

