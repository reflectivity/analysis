# Validation
Tests for reflectivity calculations
(from Andrew Nelson: https://github.com/andyfaff/orso)

## Polarised tests
### Coordinate system: Refl1D
The coordinate system used by Refl1D is based on the book chapter "Polarized Neutron Reflectometry" [^1] by C. F. Majkrzak et al. and is defined as seen in this figure: ![NIST_coords](./NIST_coords.png)

The Sample coordinate system is defined such that the surface normal (parallel to Q) is _z<sub>Sample</sub>_, and then _x<sub>Sample</sub>_  and _y<sub>Sample</sub>_ are in the sample plane.

The Lab coordinate system is defined so that _z<sub>Lab</sub>_ points along the polarisation (guide field = _H_) direction.

Further, _x<sub>Lab</sub>_ is defined to be the same as _x<sub>Sample</sub>_.  Thus _x<sub>Sample</sub>_ is a unit vector in the sample plane (perpendicular to Q) that is also perpendicular to the guide field _H_. _y<sub>Sample</sub>_, _y<sub>Lab</sub>_ are then derived from the right-hand-rule from the _x,z_ components in their respective reference frames, and importantly _H_ is then always in the sample _y,z_ plane.

The angle between _z<sub>Sample</sub>_ and _z<sub>Lab</sub>_ we call _AGUIDE_, and the angle between _x<sub>Sample</sub>_ and the in-plane component of the sample magnetization (_M<sub>IP</sub>_) is _theta<sub>M</sub>_.

Because of the way the reference frames are defined, there are always (at least) two valid choices for the direction of _x<sub>Sample</sub>_.
For some common geometries, here are some corresponding angles:
#### Guide field H in sample plane (perpendicular to Q)
For e.g. a vertical-axis right-handed reflectometer, when the guide field at sample points up

 - _AGUIDE_ = 90&deg;
 - _theta<sub>M</sub>_ = 90&deg; for _M<sub>IP</sub>_ || _H_

#### Guide field H || Q
Where the guide field is normal to the surface:
 - _AGUIDE_ = 0&deg;
 - _theta<sub>M</sub>_ is not constrained (any choice of _x<sub>Sample</sub>_ is equally valid), and does not contribute to the calculation of the scattering


[^1]:Majkrzak, C. F., K. V. O'Donovan, and N. F. Berk. "Polarized neutron reflectometry." In Neutron Scattering from Magnetic Materials, pp. 397-471. Elsevier Science, 2006.