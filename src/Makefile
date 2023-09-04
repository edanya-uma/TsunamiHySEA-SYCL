CXX=icpx
CXXFLAGS=-Wall -Wextra -O3
CPPFLAGS=-I.
SYCLFLAGS=-fsycl -fsycl-targets=nvidia_gpu_sm_86

LD=icpx
LDFLAGS=
LIBS=-lnetcdf

OBJECTS=Problema.o ShallowWater.o \
        GPU/Arista_kernel.o GPU/DC3D.o GPU/Deformacion.o \
        GPU/netcdf.o GPU/netcdfSeries.o GPU/prtoxy.o \
        GPU/ShallowWater.o GPU/Volumen_kernel.o

.PHONY: all clean
.DEFAULT: all

all: $(OBJECTS) TsunamiHySEA

clean:
	rm -f $(OBJECTS) TsunamiHySEA

%.o: %.cxx
	$(CXX) $(SYCLFLAGS) $(CXXFLAGS) $(CPPFLAGS) -MMD -MF $(*D)/$(*F).Po -o $@ -c $<

TsunamiHySEA: $(OBJECTS)
	$(LD) $(SYCLFLAGS) $(LDFLAGS) $(LIBS) -o $@ $^

-include $(OBJECTS:.o=.Po)

