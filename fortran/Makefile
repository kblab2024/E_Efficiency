! Makefile for compiling Fortran optics codes
! Supports OpenMP for CPU parallelism on multi-core CPUs (e.g. Xeon w5-3423)

FC = gfortran
FFLAGS = -O2 -Wall -std=f2008 -fopenmp
LDFLAGS = -fopenmp

# List of modules (order matters for dependencies)
MODULES = multilayer_reflectance.o te_greens.o tm_greens.o bessel_functions.o \
          gauss_quadrature.o split_integration.o te_integrate.o

# Main executables
PROGRAMS = test_multilayer test_te_greens test_tm_greens main_te_integrate

.PHONY: all clean

all: $(PROGRAMS)

# Compilation rules for modules
multilayer_reflectance.o: multilayer_reflectance.f90
	$(FC) $(FFLAGS) -c $< -o $@

te_greens.o: te_greens.f90 multilayer_reflectance.o
	$(FC) $(FFLAGS) -c $< -o $@

tm_greens.o: tm_greens.f90 te_greens.o multilayer_reflectance.o
	$(FC) $(FFLAGS) -c $< -o $@

bessel_functions.o: bessel_functions.f90
	$(FC) $(FFLAGS) -c $< -o $@
gauss_quadrature.o: gauss_quadrature.f90
	$(FC) $(FFLAGS) -c $< -o $@

split_integration.o: split_integration.f90 gauss_quadrature.o
	$(FC) $(FFLAGS) -c $< -o $@

te_integrate.o: te_integrate.f90 te_greens.o bessel_functions.o split_integration.o
	$(FC) $(FFLAGS) -c $< -o $@


# Link executables
main_te_integrate: te_integrate.o split_integration.o gauss_quadrature.o bessel_functions.o \
                   te_greens.o multilayer_reflectance.o
	$(FC) $(FFLAGS) $^ -o $@ $(LDFLAGS)

test_multilayer: multilayer_reflectance.o
	$(FC) $(FFLAGS) $^ -o $@ $(LDFLAGS)

test_te_greens: te_greens.o multilayer_reflectance.o
	$(FC) $(FFLAGS) $^ -o $@ $(LDFLAGS)

test_tm_greens: tm_greens.o te_greens.o multilayer_reflectance.o
	$(FC) $(FFLAGS) $^ -o $@ $(LDFLAGS)

# Clean up
clean:
	rm -f *.o *.mod $(PROGRAMS) *.dat

# Clean intermediate files but keep executables
clean-obj:
	rm -f *.o *.mod
