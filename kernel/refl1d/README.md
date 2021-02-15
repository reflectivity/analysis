# Refl1D calculation kernels
Pulled from https://github.com/reflectometry/refl1d.git 2021-02-15

These are all in C++ and are used from Python with a wrapper.

## [reflectivity.cc](reflectivity.cc)
The standard, unpolarized reflectivity kernel.  It uses the matrix formalism, and 
includes Nevot-Croce roughness terms (see line 55)
```cpp
    const Cplx F = (k-k_next)/(k+k_next)*exp(-2.*k*k_next*sigma[next]*sigma[next]);
```


## [magnetic_CD_tempered.cc](magnetic_CD_tempered.cc)
The original library [magnetic.cc](magnetic.cc) re-implements the Fortran algorithm of C. F. Majkrzak and N. F. Berk found in [gepore.f](gepore.f)

The matrix was reformulated in [magnetic_CD_tempered.cc](magnetic_CD_tempered.cc) to avoid issues that occur when the magnetic field has a significant non-zero out-of-plane component 
(divide-by-zero in some of the terms).  The details can be found in https://scripts.iucr.org/cgi-bin/paper?ge5021 and the new kernel is stable for all
values of vector B (magnitude and direction), and takes into account the Zeeman splitting.
