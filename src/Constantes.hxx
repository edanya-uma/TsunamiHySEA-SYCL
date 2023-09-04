#ifndef _CONSTANTES_H_
#define _CONSTANTES_H_

#include <limits>
#include <sycl/sycl.hpp>

using TVec2 = sycl::double2;
using TVec3 = sycl::double3;

using size_t = std::size_t;

constexpr double EPSILON               { std::numeric_limits<double>::epsilon()*10 };
constexpr double EARTH_RADIUS          { 6378136.6 };
constexpr size_t SPONGE_SIZE           { 4 };
constexpr double SEA_LEVEL             { 0.0 };
constexpr size_t DEFLATE_LEVEL         { 5 }; 
constexpr size_t SEA_SURFACE_FROM_FILE { 0 };
constexpr int    OKADA_STANDARD        { 1 }; 


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


typedef struct {
    sycl::double2 *areaYCosPhi;
    double *anchoVolumenes;
    double *altoVolumenes;
    double *longitud;
    double *latitud;
} tipoDatosSubmalla;

#endif

