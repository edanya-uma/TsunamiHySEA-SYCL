#include "netcdf_kernel.hxx"
#include "Constantes.hxx"
#include <sycl/sycl.hpp>

using namespace sycl;

void obtenerEtaNetCDF(double2 *d_datosVolumenes_1, float *d_vec, int num_volx,
			int num_voly, double epsilon_h, double Hmin, double H, nd_item<2> item)
{
	double2 h;
	int pos, pos_x_hebra, pos_y_hebra;

	pos_x_hebra = item.get_group(1) * NUM_HEBRASX_EST + item.get_local_id(1);
	pos_y_hebra = item.get_group(0) * NUM_HEBRASY_EST + item.get_local_id(0);

	if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
		pos = pos_y_hebra*num_volx + pos_x_hebra;

		h = d_datosVolumenes_1[pos];
		d_vec[pos] = (float) ( (h.x() < epsilon_h) ? -9999.0f : (h.x() - h.y() - Hmin)*H );
	}
}

void obtenerUxNetCDF(double2 *d_datosVolumenesNivel_1, double2 *d_datosVolumenesNivel_2, float *d_vec,
			int num_volx, int num_voly, double epsilon_h, double Hmin, double H, double Q, nd_item<2> item)
{
	int pos, pos_x_hebra, pos_y_hebra;
	double maxheps, factor;
	double h;

	pos_x_hebra = item.get_group(1) * NUM_HEBRASX_EST + item.get_local_id(1);
	pos_y_hebra = item.get_group(0) * NUM_HEBRASY_EST + item.get_local_id(0);

	if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
		pos = pos_y_hebra*num_volx + pos_x_hebra;

		h = d_datosVolumenesNivel_1[pos].x();
		if (h < epsilon_h) {
			maxheps = epsilon_h;
			factor = M_SQRT2*h / sqrt(h*h*h*h + maxheps*maxheps*maxheps*maxheps)*(Q/H);
		}
		else {
			factor = Q/(h*H);
		}
		d_vec[pos] = (float) (d_datosVolumenesNivel_2[pos].x()*factor);
	}
}

void obtenerUyNetCDF(double2 *d_datosVolumenesNivel_1, double2 *d_datosVolumenesNivel_2, float *d_vec,
			int num_volx, int num_voly, double epsilon_h, double Hmin, double H, double Q, nd_item<2> item)
{
	int pos, pos_x_hebra, pos_y_hebra;
	double maxheps, factor;
	double h;

	pos_x_hebra = item.get_group(1) * NUM_HEBRASX_EST + item.get_local_id(1);
	pos_y_hebra = item.get_group(0) * NUM_HEBRASY_EST + item.get_local_id(0);

	if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
		pos = pos_y_hebra*num_volx + pos_x_hebra;

		h = d_datosVolumenesNivel_1[pos].x();
		if (h < epsilon_h) {
			maxheps = epsilon_h;
			factor = M_SQRT2*h / sqrt(h*h*h*h + maxheps*maxheps*maxheps*maxheps)*(Q/H);
		}
		else {
			factor = Q/(h*H);
		}
		d_vec[pos] = (float) (d_datosVolumenesNivel_2[pos].y()*factor);
	}
}

void obtenerBatimetriaModificadaNetCDF(double2 *d_datosVolumenesNivel_1, float *d_vec, int num_volx,
			int num_voly, double Hmin, double H, nd_item<2> item)
{
	int pos, pos_x_hebra, pos_y_hebra;
	double prof;

	pos_x_hebra = item.get_group(1) * NUM_HEBRASX_EST + item.get_local_id(1);
	pos_y_hebra = item.get_group(0) * NUM_HEBRASY_EST + item.get_local_id(0);

	if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
		pos = pos_y_hebra*num_volx + pos_x_hebra;

		prof = d_datosVolumenesNivel_1[pos].y();
		d_vec[pos] = (float) ((prof+Hmin)*H);
	}
}

/***********/
/* Metrics */
/***********/

void obtenerEta1MaximaNetCDF(double *d_eta1_maxima, float *d_vec, int num_volx,
			int num_voly, double Hmin, double H, nd_item<2> item)
{
	int pos, pos_x_hebra, pos_y_hebra;
	double eta;

	pos_x_hebra = item.get_group(1) * NUM_HEBRASX_EST + item.get_local_id(1);
	pos_y_hebra = item.get_group(0) * NUM_HEBRASY_EST + item.get_local_id(0);

	if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
		pos = pos_y_hebra*num_volx + pos_x_hebra;

		eta = d_eta1_maxima[pos];
		d_vec[pos] = (float) ((eta < -1e20) ? -9999.0f : (eta-Hmin)*H);
	}
}

void obtenerTiemposLlegadaNetCDF(double *d_tiempos_llegada, float *d_vec, int num_volx,
			int num_voly, double T, nd_item<2> item)
{
	int pos, pos_x_hebra, pos_y_hebra;
	double t;

	pos_x_hebra = item.get_group(1) * NUM_HEBRASX_EST + item.get_local_id(1);
	pos_y_hebra = item.get_group(0) * NUM_HEBRASY_EST + item.get_local_id(0);

	if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
		pos = pos_y_hebra*num_volx + pos_x_hebra;

		t = d_tiempos_llegada[pos];
		d_vec[pos] = (float) ((t < 0.0) ? -1.0f : t*T);
	}
}

