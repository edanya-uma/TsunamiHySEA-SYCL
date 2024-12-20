#include "Metrics.hxx"
#include "Constantes.hxx"
#include <sycl/sycl.hpp>

using namespace sycl;

void inicializarEta1MaximaNivel0GPU(double2 *d_datosVolumenes_1, double *d_eta1_maxima,
                                    int num_volx, int num_voly, double epsilon_h,
                                    nd_item<2> item)
{
        double2 W;
        int pos, pos_x_hebra, pos_y_hebra;

        pos_x_hebra = item.get_group(1) * NUM_HEBRASX_EST + item.get_local_id(1);
        pos_y_hebra = item.get_group(0) * NUM_HEBRASY_EST + item.get_local_id(0);

        if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
                pos = pos_y_hebra*num_volx + pos_x_hebra;

                W = d_datosVolumenes_1[pos];
                d_eta1_maxima[pos] = ((W.x() < epsilon_h) ? -1e30 : W.x() - W.y());
        }
}

void actualizarEta1MaximaNivel0GPU(double2 *d_datosVolumenes_1, double *d_eta1_maxima,
                                   int num_volx, int num_voly, double tiempo_act,
                                   double epsilon_h, nd_item<2> item)
{
        double2 W;
        double val, eta1;
        int pos, pos_x_hebra, pos_y_hebra;

        pos_x_hebra = item.get_group(1) * NUM_HEBRASX_EST + item.get_local_id(1);
        pos_y_hebra = item.get_group(0) * NUM_HEBRASY_EST + item.get_local_id(0);

        if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
                pos = pos_y_hebra*num_volx + pos_x_hebra;

                eta1 = d_eta1_maxima[pos];
                W = d_datosVolumenes_1[pos];
                val = W.x() - W.y();
                if ((val > eta1) && (W.x() > epsilon_h))
                        d_eta1_maxima[pos] = val;
        }
}

void inicializarTiemposLlegadaNivel0GPU(double *d_tiempos_llegada, int num_volx,
                                        int num_voly, nd_item<2> item)
{
        int pos, pos_x_hebra, pos_y_hebra;

        pos_x_hebra = item.get_group(1) * NUM_HEBRASX_EST + item.get_local_id(1);
        pos_y_hebra = item.get_group(0) * NUM_HEBRASY_EST + item.get_local_id(0);

        if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
                pos = pos_y_hebra*num_volx + pos_x_hebra;

                d_tiempos_llegada[pos] = -1.0;
        }
}

void actualizarTiemposLlegadaNivel0GPU(double2 *d_datosVolumenes_1, double *d_tiempos_llegada,
                                      double *d_eta1_inicial, int num_volx, int num_voly,
                                      double tiempo_act, double dif_at, nd_item<2> item)
{
        double2 W;
        double eta_act, eta_ini, t;
        int pos, pos_x_hebra, pos_y_hebra;

        pos_x_hebra = item.get_group(1) * NUM_HEBRASX_EST + item.get_local_id(1);
        pos_y_hebra = item.get_group(0) * NUM_HEBRASY_EST + item.get_local_id(0);

        if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
                pos = pos_y_hebra*num_volx + pos_x_hebra;

                t = d_tiempos_llegada[pos];
                eta_ini = d_eta1_inicial[pos];
                W = d_datosVolumenes_1[pos];
                eta_act = W.x() - W.y();
                if ((t < 0.0) && (abs(eta_act - eta_ini) > dif_at))
                        d_tiempos_llegada[pos] = tiempo_act;
        }
}

