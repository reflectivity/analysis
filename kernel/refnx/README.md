# refnx calculation kernels
Performant kernels for unpolarised reflectivity calculation (polarisation is on
the way).

Pulled from https://github.com/refnx/refnx.git 2021-02-16

## [refcalc.c](refcalc.c)
Uses the Abeles matrix method for calculation, with Nevot-Croce roughness.
Calculations are done in C.
In refnx this code is called from a Python via a Cython based extension
([_creflect.pyx](_creflect.pyx)) and a [C++ wrapper](refcaller.cpp).
The base C-code is vectorised over all Q points, and the C++ wrapper has an
optional argument for thread-based parallelisation.


## [_reflect.py](_reflect.py)
The `abeles` function uses the Abeles matrix method for calculation, with
Nevot-Croce roughness.
Calculations are done in Python. The code is vectorised over all Q points
(reducing overhead).


## [abeles_pyopencl.cl](abeles_pyopencl.cl)
A reflectometry calculation kernel using the Abeles matrix method with
Nevot-Croce roughness.
Calculations are done with pyopencl/openCL, with driver code from
[_reflect.py](_reflect.py).
The code is vectorised/parallelised over all Q points.