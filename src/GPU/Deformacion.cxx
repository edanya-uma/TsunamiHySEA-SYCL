#include "Deformacion.hxx"
#include "Constantes.hxx"
#include "prtoxy.hxx"
#include "DC3D.hxx"

using namespace sycl;

/******************/
/* Okada standard */
/******************/

void convertirAMenosValorAbsolutoGPU(double *d_in, float *d_out, int n, nd_item<1> item)
{
	int pos = item.get_group(0) * NUM_HEBRAS_PUNTOS + item.get_local_id(0);

	if (pos < n) {
		d_out[pos] = (float) (-fabs(d_in[pos]));
	}
}

void truncarDeformacionGPU(double *d_def, int num_volx, int num_voly, double crop_value, nd_item<2> item)
{
	int pos, pos_x_hebra, pos_y_hebra;
	double U_Z;

	pos_x_hebra = item.get_group(1) * NUM_HEBRASX_EST + item.get_local_id(1);
	pos_y_hebra = item.get_group(0) * NUM_HEBRASY_EST + item.get_local_id(0);

	if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
		pos = pos_y_hebra*num_volx + pos_x_hebra;
		U_Z = d_def[pos];
		if (fabs(U_Z) < crop_value)
			d_def[pos] = 0.0;
	}
}

void sumarDeformacionADatosGPU(double2 *d_datosVolumenes_1, double *d_def, double *d_eta1Inicial,
			int num_volx, int num_voly, nd_item<2> item)
{
	int pos, pos_x_hebra, pos_y_hebra;
	double U_Z;

	pos_x_hebra = item.get_group(1) * NUM_HEBRASX_EST + item.get_local_id(1);
	pos_y_hebra = item.get_group(0) * NUM_HEBRASY_EST + item.get_local_id(0);

	if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
		pos = pos_y_hebra*num_volx + pos_x_hebra;
		U_Z = d_def[pos];
		d_datosVolumenes_1[pos].y() -= U_Z;
		d_eta1Inicial[pos] += U_Z;
	}
}

void aplicarOkadaStandardGPU(double2 *d_datosVolumenes_1, double *d_def, int num_volx, int num_voly, double lon_ini,
		double incx, double lat_ini, double incy, double LON_C_ent, double LAT_C_ent, double DEPTH_C_ent, double FAULT_L,
		double FAULT_W, double STRIKE, double DIP_ent, double RAKE, double SLIP, double H, nd_item<2> item)
{
    double LON_P, LAT_P;
    double LON_C, LAT_C;
    double DEPTH_C, DIP;
    double S_RAKE;
    double C_RAKE;
    double S_STRIKE;
    double C_STRIKE;
    double Z;
    double AL1, AL2, AW1, AW2;
    double X_OKA, Y_OKA;
    double XP, YP;
    double RAD = M_PI/180.0;
    double alfa = 2.0/3.0;
    int i0 = 0;
    int IRET;
    double U_X, U_Y, U_Z, UXX, UYX, UZX, UXY, UYY, UZY, UXZ, UYZ, UZZ;
    double DISL1, DISL2, DISL3;
    int pos, pos_x_hebra, pos_y_hebra;

    pos_x_hebra = item.get_group(1) * NUM_HEBRASX_EST + item.get_local_id(1);
    pos_y_hebra = item.get_group(0) * NUM_HEBRASY_EST + item.get_local_id(0);

    if ((pos_x_hebra < num_volx) && (pos_y_hebra < num_voly)) {
        pos = pos_y_hebra*num_volx + pos_x_hebra;
        LON_C = LON_C_ent;
        LAT_C = LAT_C_ent;
        DEPTH_C = DEPTH_C_ent;
        DIP = DIP_ent;

        LON_P = lon_ini + pos_x_hebra*incx;
        LAT_P = lat_ini + pos_y_hebra*incy;

        S_RAKE = sin(RAD*RAKE);
        C_RAKE = cos(RAD*RAKE);

        S_STRIKE = sin(RAD*STRIKE);
        C_STRIKE = cos(RAD*STRIKE);

        DISL2 = SLIP*S_RAKE;
        DISL1 = SLIP*C_RAKE;
        DISL3 = 0.0;

        Z = 0.0;
        AL1 = -0.5*FAULT_L;
        AL2 = 0.5*FAULT_L;
        AW1 = -0.5*FAULT_W;
        AW2 = 0.5*FAULT_W;

        prtoxy_(&LAT_P, &LON_P, &LAT_C, &LON_C, &XP, &YP, &i0);
        X_OKA = XP*S_STRIKE + YP*C_STRIKE;
        Y_OKA = -XP*C_STRIKE + YP*S_STRIKE;
        dc3d_(&alfa, &X_OKA, &Y_OKA, &Z, &DEPTH_C, &DIP, &AL1, &AL2, &AW1, &AW2, &DISL1, &DISL2, &DISL3,
            &U_X, &U_Y, &U_Z, &UXX, &UYX, &UZX, &UXY, &UYY, &UZY, &UXZ, &UYZ, &UZZ, &IRET);

        U_Z /= H;
        d_def[pos] = U_Z;
    }
}

void aplicarOkada(queue &q, event &ev, double2 *d_datosVolumenes_1, double *d_eta1Inicial, int crop_flag, double crop_value,
		double *d_deltaTVolumenes, float *d_vec, int num_volx, int num_voly, double lon_ini, double incx, double lat_ini,
		double incy, double LON_C, double LAT_C, double DEPTH_C, double FAULT_L, double FAULT_W, double STRIKE, double DIP,
		double RAKE, double SLIP, range<2> blockGridEst, range<2> threadBlockEst, double H)
{
	int num_volumenes = num_volx*num_voly;
    range<1> blockGridVec(iDivUp(num_volumenes, NUM_HEBRAS_PUNTOS));
    range<1> threadBlockVec(NUM_HEBRAS_PUNTOS);
	float def_max;
	double crop_value_final;

	nd_range<2> Krange { blockGridEst*threadBlockEst, threadBlockEst };
	ev = q.parallel_for<class aplicar_okada>( Krange, ev, [=]( nd_item<2> idx )
	{
		aplicarOkadaStandardGPU(d_datosVolumenes_1, d_deltaTVolumenes, num_volx, num_voly, lon_ini, incx,
			lat_ini, incy, LON_C, LAT_C, DEPTH_C, FAULT_L, FAULT_W, STRIKE, DIP, RAKE, SLIP, H, idx);
	});

	if (crop_flag == CROP_RELATIVE) {
		nd_range<1> my_range { blockGridVec*threadBlockVec, threadBlockVec };
		q.parallel_for<class convertir_abs>( my_range, ev, [=]( nd_item<1> idx )
		{
			convertirAMenosValorAbsolutoGPU(d_deltaTVolumenes, d_vec, num_volumenes, idx);
		});

		// Reduction
		size_t size = num_volumenes;
		float minResult = std::numeric_limits<float>::max();
		buffer<float> minBuf { &minResult, 1 };
		q.submit( [d_vec,size,&minBuf,&ev]( handler &h )
		{
			h.depends_on(ev);
			auto minReduction = reduction(minBuf,h,minimum<>());
			h.parallel_for<class reduction_crop>( range<1>(size), minReduction, [=]( id<1> idx, auto &min )
			{
				min.combine(d_vec[idx]);
			});
		});
		def_max = -(minBuf.get_host_access()[0]);
		crop_value_final = crop_value*((double) def_max);

		Krange = nd_range<2> { blockGridEst*threadBlockEst, threadBlockEst };
		ev = q.parallel_for<class truncar_def_rel>( Krange, ev, [=]( nd_item<2> idx )
		{
			truncarDeformacionGPU(d_deltaTVolumenes, num_volx, num_voly, crop_value_final, idx);
		});
	}
	else if (crop_flag == CROP_ABSOLUTE) {
		Krange = nd_range<2> { blockGridEst*threadBlockEst, threadBlockEst };
		ev = q.parallel_for<class truncar_def_abs>( Krange, ev, [=]( nd_item<2> idx )
		{
			truncarDeformacionGPU(d_deltaTVolumenes, num_volx, num_voly, crop_value, idx);
		});
	}

	Krange = nd_range<2> { blockGridEst*threadBlockEst, threadBlockEst };
	ev = q.parallel_for<class sumar_def>( Krange, ev, [=]( nd_item<2> idx )
	{
		sumarDeformacionADatosGPU(d_datosVolumenes_1, d_deltaTVolumenes, d_eta1Inicial, num_volx, num_voly, idx);
	});
}

