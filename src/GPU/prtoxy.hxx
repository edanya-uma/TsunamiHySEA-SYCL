#ifndef PRTOXY_H
#define PRTOXY_H

#include <sycl/sycl.hpp>

SYCL_EXTERNAL
int prtoxy_(double *alatdg, double *alngdg, 
            double *alato, double *alngo, double *x, double *y, 
            int *ind);
#endif

