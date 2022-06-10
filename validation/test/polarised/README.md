_A series of tests for examining polarised reflectivity calculation
correctness._

Each of the tests is described by a `test*` file in this directory. 
Those test files have a series of header lines that can be used to describe the test, prefixed by #.
Four pieces of information are encoded in this file in the uncommented lines:

1. A string that refers to a file in the `layers` directory,
which specifies a slab-like model for which the reflectivity needs to be
calculated.
1. A string that refers to a file in the `data` directory
containing the validated values against which the reflectivity calculation is
to be checked. These 'datafiles' will eventually be standard ORSO format, but
for the time being they are specified as below in "Data Specification"
1. (Optional) The angle of the guide field with respect to the surface normal, in degrees.
Will be $90^{\circ}$ or $270^{\circ}$ for a guide field in the sample plane (see ../README.md for geometry definition)
1. (Optional) The magnitude of the applied field at the sample position, in units of Tesla.  Only important for spin-flip measurements when the applied field has an out-of-plane component.

## Example test:
```bash
# Zeeman splitting of spin-flip: 
# guide field nearly parallel to Q (5 deg offset)
layers/test0.layers
data/test0.dat
# AGUIDE:
5
# H:
0.244
```

## Data specification: 

| $Q (\textrm{Å}^{-1})$ |  $R_{--}$  | $R_{-+}$ | $R_{+-}$ | $R_{++}$ |
| --------------------- | ---------- | -------- | -------- | -------- |
| 0.00005 | 1.00000 | 0.00000 | 0.00000 | 1.00000 |
| 0.00010 | 1.00000 | 0.00000 | 0.00000 | 1.00000 |
| 0.00015 | 1.00000 | 0.00000 | 0.00000 | 1.00000 |
...

## Layer specification:

| thickness (Å)	| sld | mu | thetaM $({}^\circ)$ | sldm | roughness (Å)|
| --------- | --- | -- | ------ | ---- | --------- |
| _0.000_&dagger; | 0.000 | _0.000_&dagger; | 90.00 | 0.000 | 0.000 |
| 50.00 | 4.000 | 0.000 | 90.00 | 0.000 | 0.000 |
| 825.0 | 9.060 | 0.000 | 90.00 | 0.000 | 0.000 |
| _0.000_&dagger; | 2.070 | 0.000 | 90.00 | 0.000 | 0.000 |
...

**sld**, **mu** and **sldm** (the magnetic sld) have units $1\times 10^{-6} \textrm{Å}^{-2}$.

These values marked with &dagger; are ignored:
 - the fronting and backing media have no
thickness
 - absorption is also ignored in the fronting medium.
