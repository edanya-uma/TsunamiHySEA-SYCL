#ifndef CONSTANTES_H
#define CONSTANTES_H

#include <limits>
#include <sycl/sycl.hpp>

using size_t = std::size_t;

constexpr double EPSILON               { std::numeric_limits<double>::epsilon() };
constexpr double EARTH_RADIUS          { 6378136.6 };
constexpr int    MAX_FAULTS            { 5000 };
constexpr size_t SPONGE_SIZE           { 4 };
constexpr double SEA_LEVEL             { 0.0 };
constexpr size_t DEFLATE_LEVEL         { 5 }; 

constexpr int    SEA_SURFACE_FROM_FILE { 0 };
constexpr int    OKADA_STANDARD        { 1 };
constexpr int    NO_CROP               { 0 };
constexpr int    CROP_RELATIVE         { 1 };
constexpr int    CROP_ABSOLUTE         { 2 };

constexpr double sqrt2        { 1.4142135623730950488016887242096980785696718 };

constexpr size_t NUM_HEBRASX_ARI   { 8 };
constexpr size_t NUM_HEBRASY_ARI   { 8 };
constexpr size_t NUM_HEBRASX_EST   { 8 };
constexpr size_t NUM_HEBRASY_EST   { 8 };
constexpr size_t NUM_HEBRAS_PUNTOS { 8 };

constexpr size_t iDivUp( size_t a, size_t b )
{
    return 1 + (a-1)/b;
}

using tipoDatosSubmalla = struct {
	sycl::double2 *areaYCosPhi;
	double *anchoVolumenes;
	double *altoVolumenes;
	double *longitud;
	double *latitud;
};

#endif

