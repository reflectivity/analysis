# Strategies for multi-modal reflectivity measurements

A response to community request (see issue #43), specifically to address issues people have in co-fitting neutron and X-ray reflectivities 
where the smaller (relative) statistical error bars on the X-ray data cause it to dominate fit results.

# Effect of systematic errors on fitting X-ray data

Typically, the statistical error in X-ray data is very small, to the point whereby (for laboratory data at least) systematic errors dominate. Quite often, the error bars or the reflectivity theory does not capture these errors, such that they can dominate a fit – especially if these occur close to the critical edge. This has the effect of keeping the &chi;<sup>2</sup> metric high and causing the resulting depth profile to contort in such a way as to minimise the systematic error – resulting in often non-physical profiles which do not capture the high Q/scattering angle data.

## Systematic errors "as yet not accounted for" in reduction/analysis of laboratory X-ray data

The classic example for laboratory X-ray data is the footprint. This is typically captured by using a geometric correction (such as a square beam or gaussian profile). Either in some form of correction/reduction to the data prior to fitting, or by directly fitting it (fitting parameters such as beam width and sample length). 
However, the basic geometric corrections do not always capture the footprint correction - this could be due to a number of reasons, i.e. :

1.	Is the sample perfectly centred in the beam?
2.	If the sample is not square/rectangular and/or not "squared up" to the incoming beam direction, this seems to have a significant effect - as the basic geometric corrections assume a square beam/square sample or gaussian beam/square sample.
3.	The contribution from the straight-through beam may not be insignificant at low scattering angles - i.e. assuming a gaussian beam profile, at low scattering angles there may be some overspill from the incident beam into the detector, artificially increasing the value of the sub-critical edge region, which would not be captured by footprint corrections or reflectometry theory.


