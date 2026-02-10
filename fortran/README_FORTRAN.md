# Fortran Translation of Optics Code

This directory contains Fortran translations of the Python optics simulation code for multilayer structures.

## Files

### Fortran Source Files
- `multilayer_reflectance.f90` - Core multilayer reflectance calculations (RF/RB recursion)
- `te_greens.f90` - TE polarization Green's function calculations
- `tm_greens.f90` - TM polarization Green's function calculations
- `bessel_functions.f90` - Bessel function implementations (J0, J2)

### Python Source Files (Original)
- `multilayer_reflectance.py` - Python version of reflectance calculations
- `te_greens.py` - Python TE Green's functions
- `tm_greens.py` - Python TM Green's functions
- `te_integrate_plot.py` - Integration and plotting utilities

## Building

Use the provided Makefile:

```bash
# Build all programs
make

# Build specific program
make test_multilayer
make test_te_greens
make test_tm_greens

# Clean up
make clean
```

Or compile manually:

```bash
# Compile multilayer reflectance
gfortran -O2 -o test_multilayer multilayer_reflectance.f90

# Compile TE Green's functions (requires multilayer_reflectance)
gfortran -O2 -c multilayer_reflectance.f90
gfortran -O2 -o test_te_greens te_greens.f90 multilayer_reflectance.o

# Compile TM Green's functions (requires both previous modules)
gfortran -O2 -c te_greens.f90
gfortran -O2 -o test_tm_greens tm_greens.f90 te_greens.o multilayer_reflectance.o
```

## Running

```bash
# Test multilayer reflectance
./test_multilayer
# Output: reflectance.dat

# Test TE Green's functions
./test_te_greens

# Test TM Green's functions
./test_tm_greens
```

## Key Differences from Python

1. **Array Indexing**: Fortran uses 1-based indexing (Python uses 0-based)
2. **Complex Numbers**: Fortran has native complex arithmetic
3. **Module System**: Uses modules instead of Python imports
4. **Type Declarations**: All variables must be declared
5. **Performance**: Fortran code typically runs faster for numerical computations

## Module Structure

### multilayer_reflectance
- `r_te()` - TE Fresnel reflection coefficient
- `r_tm()` - TM Fresnel reflection coefficient
- `RF_multilayer()` - Reflection looking downward
- `RB_multilayer()` - Reflection looking upward

### te_greens
- `compute_q_list()` - Calculate q values for all layers
- `compute_RF_all()` - Compute RF for all layers
- `compute_RB_all()` - Compute RB for all layers
- `TE_f1yf2y_same_layer()` - Source amplitudes
- `propagate_down_TE()` - Downward propagation
- `propagate_up_TE()` - Upward propagation
- `gyy_TE()` - Assemble TE Green's function G_yy

### tm_greens
- `TM_f1xf2x_same_layer()` - TM source amplitudes for x-component
- `TM_f1zf2z_same_layer()` - TM source amplitudes for z-component
- `propagate_down_TM()` - TM downward propagation
- `propagate_up_TM()` - TM upward propagation
- `gxx_TM()` - G_xx component
- `gzx_TM()` - G_zx component

### bessel_functions
- `J0_series()` - Bessel J0 function (series expansion)
- `J2_series()` - Bessel J2 function (series expansion)
- `J0_array()` - Vectorized J0
- `J2_array()` - Vectorized J2

## Notes

- Double precision (15 digits) is used throughout
- Complex square roots handle branch cuts correctly with `cmplx()`
- The code structure follows the 5-level architecture from the Python version
- OpenMP is used for CPU parallelism in the rho-array integration loop
- Set `OMP_NUM_THREADS` to control the number of threads (defaults to all available cores)
- Plotting functionality requires external tools (gnuplot, matplotlib via Python bridge)

## Future Work

- Add CUDA/OpenACC kernels for GPU-accelerated k_parallel integration
- Implement parallel computing with MPI for distributed workloads
- Add more test cases and validation scripts
- Create Python wrapper for easy comparison
